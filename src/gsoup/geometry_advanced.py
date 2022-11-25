import torch
import numpy as np

def calculate_vertex_incident_scalar(num_vertices, f, per_face_scalar_field, avg=False):
    """
    computes a scalar field per vertex by summing / averaging the incident face scalar field
    """
    device = f.device
    incident_face_areas = torch.zeros([num_vertices, 1], device=device)
    f_unrolled = f.flatten()
    face_indices_repeated_per_vertex = torch.arange(f.shape[0], device=f.device)
    face_indices_repeated_per_vertex = torch.repeat_interleave(face_indices_repeated_per_vertex, repeats=3)
    face_areas_repeated_per_face = per_face_scalar_field[face_indices_repeated_per_vertex].unsqueeze(-1)
    incident_face_areas = torch.index_add(incident_face_areas, dim=0, index=f_unrolled,
                                          source=face_areas_repeated_per_face)
    if avg:
        neighbors = torch.index_add(torch.zeros_like(incident_face_areas), dim=0, index=f_unrolled,
                                    source=torch.ones_like(face_areas_repeated_per_face))
        incident_face_areas = incident_face_areas / neighbors
    return incident_face_areas


def calculate_vertex_incident_vector(num_vertices, f, per_face_vector_field):
    """
    computes a vector field per vertex by summing the incident face vector field
    """
    device = f.device
    incident_face_areas = torch.zeros([num_vertices, 1], device=device)
    f_unrolled = f.flatten()
    face_indices_repeated_per_vertex = torch.arange(f.shape[0], device=f.device)
    face_indices_repeated_per_vertex = torch.repeat_interleave(face_indices_repeated_per_vertex, repeats=3)
    normals_repeated_per_face = per_face_vector_field[face_indices_repeated_per_vertex]
    normals_repeated_per_face = normals_repeated_per_face
    incident_face_vectors = torch.zeros([num_vertices,per_face_vector_field.shape[-1]], device=device)
    incident_face_vectors = torch.index_add(incident_face_vectors, dim=0, index=f_unrolled,
                                            source=normals_repeated_per_face)
    return incident_face_vectors