__version__ = "0.0.8"

from .core import (
    is_np,
    broadcast_batch,
    to_hom,
    homogenize,
    look_at_torch,
    look_at_np,
    compose_rt,
    to_44,
    to_34,
    to_np,
    to_numpy,
    to_torch,
    to_8b,
    to_float,
    rotx,
    roty,
    rotz,
    map_range,
    create_random_cameras_on_unit_sphere,
    random_vectors_on_sphere,
    opengl_c2w_to_opencv_c2w,
    opencv_c2w_to_opengl_c2w,
    opencv_intrinsics_from_opengl_project,
    opengl_project_from_opencv_intrinsics,
    perspective_projection,
    vec2skew,
    batch_vec2skew,
    rotvec2mat,
    batch_rotvec2mat,
    random_qvec,
    qvec2mat,
    batch_qvec2mat,
    mat2qvec,
    batch_mat2qvec,
)

from .geometry_basic import (
    duplicate_faces,
    remove_duplicate_faces,
    get_aspect_ratio,
    normalize_vertices,
    clean_infinite_vertices,
    calc_edges,
    calc_face_normals,
    calc_vertex_normals,
    calc_edge_length,
    get_center_of_attention,
    scale_poses,
    find_princple_componenets,
    qslerp
)

from .geometry_advanced import (
    distribute_field,
    distribute_scalar_field,
    distribute_vector_field,
    qem
)

from .gsoup_io import (
    write_to_json,
    save_image,
    save_images,
    save_animation,
    save_mesh,
    save_meshes,
    save_obj,
    load_image,
    load_images,
    load_obj,
    load_mesh,
)

from .image import (
    alpha_compose,
    draw_text_on_image,
    draw_gizmo_on_image,
    merge_figures_with_line,
    generate_checkerboard,
    generate_voronoi_diagram,
    generate_gray_gradient,
    generate_dot_pattern,
    generate_stripe_pattern,
    generate_concentric_circles,
    generate_lollipop_pattern,
    interpolate_single_channel,
    interpolate_multi_channel,
    image_grid,
    resize_images_naive,
    adjust_contrast_brightness,
    change_brightness,
    linear_to_srgb,
    srgb_to_linear,
    pad_image_to_res,
    mask_regions,
    crop_center
)

from .video import (
    get_video_info,
    get_frame_timestamps,
    load_video,
    save_video,
    reverse_video,
    compress_video,
    video_to_images,
    slice_from_video,
    VideoReader,
    FPS
)

from .procam import (
    warp_image,
    generate_gray_code,
    calibrate_procam,
    pix2pix_correspondence,
    naive_color_compensate,
)

from .sphere_trace import (
    generate_rays,
    render
)

from . import structures