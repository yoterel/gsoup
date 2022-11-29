import torch
import torch.nn as nn
from pathlib import Path
from gsoup import structures
import torch
import torch.nn as nn

def sphere_tracing(
        signed_distance_function,
        ray_positions,
        ray_directions,
        num_iterations,
        convergence_threshold,
        foreground_masks=None,
        bounding_radius=None
):
    """
    Sphere trace an sdf. can be made differentiable using https://arxiv.org/abs/1912.07372
    """
    if foreground_masks is None:
        foreground_masks = torch.all(torch.isfinite(ray_positions), dim=-1, keepdim=True)

    if bounding_radius:
        a = torch.sum(ray_directions * ray_directions, dim=-1, keepdim=True)
        b = 2 * torch.sum(ray_directions * ray_positions, dim=-1, keepdim=True)
        c = torch.sum(ray_positions * ray_positions, dim=-1, keepdim=True) - bounding_radius ** 2
        d = b ** 2 - 4 * a * c
        t = (-b - torch.sqrt(d)) / (2 * a)
        bounded = d >= 0
        ray_positions = torch.where(bounded, ray_positions + ray_directions * t, ray_positions)
        foreground_masks = foreground_masks & bounded

    with torch.no_grad():
        for i in range(num_iterations):
            signed_distances = signed_distance_function(ray_positions)
            if i:
                ray_positions = torch.where(foreground_masks & ~converged,
                                            ray_positions + ray_directions * signed_distances, ray_positions)
            else:
                ray_positions = torch.where(foreground_masks, ray_positions + ray_directions * signed_distances,
                                            ray_positions)
            if bounding_radius:
                bounded = torch.norm(ray_positions, dim=-1, keepdim=True) < bounding_radius
                foreground_masks = foreground_masks & bounded
            converged = torch.abs(signed_distances) < convergence_threshold
            if torch.all(~foreground_masks | converged):
                break
    return ray_positions, converged


def compute_shadows(
        signed_distance_function,
        surface_positions,
        surface_normals,
        light_directions,
        num_iterations,
        convergence_threshold,
        foreground_masks=None,
        bounding_radius=None,
):
    surface_positions, converged = sphere_tracing(
        signed_distance_function=signed_distance_function,
        ray_positions=surface_positions + surface_normals * 1e-3,
        ray_directions=light_directions,
        num_iterations=num_iterations,
        convergence_threshold=convergence_threshold,
        foreground_masks=foreground_masks,
        bounding_radius=bounding_radius,
    )
    return foreground_masks & converged


def compute_normal(
        signed_distance_function,
        surface_positions,
        finite_difference_epsilon,
        use_gradient=False
):
    if use_gradient:
        surface_positions.requires_grad = True
        raw = signed_distance_function(surface_positions)
        d_output = torch.ones_like(raw, requires_grad=False, device=raw.device)
        surface_normals = torch.autograd.grad(
                outputs=raw,
                inputs=surface_positions,
                grad_outputs=d_output,
                create_graph=False,
                retain_graph=False)[0]
    else:
        finite_difference_epsilon_x = surface_positions.new_tensor([finite_difference_epsilon, 0.0, 0.0])
        finite_difference_epsilon_y = surface_positions.new_tensor([0.0, finite_difference_epsilon, 0.0])
        finite_difference_epsilon_z = surface_positions.new_tensor([0.0, 0.0, finite_difference_epsilon])
        surface_normals_x = signed_distance_function(
            surface_positions + finite_difference_epsilon_x) - signed_distance_function(
            surface_positions - finite_difference_epsilon_x)
        surface_normals_y = signed_distance_function(
            surface_positions + finite_difference_epsilon_y) - signed_distance_function(
            surface_positions - finite_difference_epsilon_y)
        surface_normals_z = signed_distance_function(
            surface_positions + finite_difference_epsilon_z) - signed_distance_function(
            surface_positions - finite_difference_epsilon_z)
        surface_normals = torch.cat((surface_normals_x, surface_normals_y, surface_normals_z), dim=-1)
    surface_normals = nn.functional.normalize(surface_normals, dim=-1)
    return surface_normals


def generate_rays(w2v, v2c, resx=512, resy=512, device="cuda:0"):
    """
    generates batch of rays for a batch of cameras
    :param w2v: world to view matrix
    :param v2c: view to clip matrix
    :param resx: resolution x
    :param resy: resolution y
    :param device: device
    """
    y_clip = (torch.arange(resy, dtype=torch.float32, device=device) / resy) * 2 - 1
    x_clip = (torch.arange(resx, dtype=torch.float32, device=device) / resx) * 2 - 1
    xy_clip = torch.stack(torch.meshgrid(x_clip, y_clip, indexing="ij"), dim=-1)
    xy_clip_near = torch.cat((xy_clip, -1*torch.ones_like(xy_clip[:, :, :1]), torch.ones_like(xy_clip[:, :, :1])), dim=-1)
    xy_clip_far = torch.cat((xy_clip, torch.ones_like(xy_clip[:, :, :1]), torch.ones_like(xy_clip[:, :, :1])), dim=-1)
    xy_view_near = (torch.inverse(v2c) @ xy_clip_near.view(-1, 4).T).T.view(resy, resx, 4)
    xy_view_near = xy_view_near / xy_view_near[:, :, 3:]
    rays_d = []
    rays_o = []
    for transform in w2v:
        xy_world_near = (torch.inverse(transform) @ xy_view_near.view(-1, 4).T).T.view(resy, resx, 4)
        ray_o = xy_world_near[:, :, :3]
        camera_position = torch.inverse(transform)[:3 ,3]
        ray_d = nn.functional.normalize(xy_world_near[:, :, :3] - camera_position, dim=-1)
        rays_d.append(ray_d)
        rays_o.append(ray_o)
    return torch.stack(rays_o), torch.stack(rays_d)


def render(p_sdf, ray_positions, ray_directions, num_iterations=2000, convergence_threshold=1e-4, use_gradient=False):
    surface_positions, converged = sphere_tracing(
        signed_distance_function=p_sdf,
        ray_positions=ray_positions,
        ray_directions=ray_directions,
        num_iterations=num_iterations,
        convergence_threshold=convergence_threshold,
        bounding_radius=2.0
    )
    surface_positions = torch.where(converged, surface_positions, torch.zeros_like(surface_positions))
    surface_normals = compute_normal(
        signed_distance_function=p_sdf,
        surface_positions=surface_positions,
        finite_difference_epsilon=1e-4,
        use_gradient=use_gradient
    )
    surface_normals = torch.where(converged, surface_normals, torch.zeros_like(surface_normals))
    image = (surface_normals + 1.0) / 2.0
    image = torch.where(converged, image, torch.zeros_like(image))
    image = torch.concat((image, converged), dim=-1)
    return image
