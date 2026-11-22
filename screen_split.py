# Screen-space major-axis split for 3D Gaussian Splatting (one-shot, N = 2^k).
#
# Each selected Gaussian is projected onto the image plane of `camera`.
# Let s1 >= s2 be the stds of the projected 2D Gaussian (major / minor axis).
# The k halvings are distributed greedily, exactly as if the Gaussian were
# split in two k times, each time along its currently longest projected axis:
#
#     repeat k times:  halve the larger of (s1, s2)      (tie -> s1)
#
# which gives a halvings of the major axis and b = k - a of the minor axis.
# All 2^k children are then created at once with projected stds
# (s1 / 2^a, s2 / 2^b). Nothing is re-projected and no per-level loop is used:
# one closed-form 2x2 eigen-decomposition and one 3x3 eigh per PARENT.
#
# Usage:
#
#   from screen_split import screen_major_axis_split
#   new_xyz, new_scale, new_rot, parent = screen_major_axis_split(
#       xyz, scaling, rotation, viewpoint_cam, k=k_per_parent)
#
# Conventions follow the original 3DGS code base:
#   - quaternion order (w, x, y, z)
#   - camera.world_view_transform is the transposed 4x4 world-to-view matrix
#   - camera.FoVx / FoVy / image_width / image_height

import inspect
import math

import torch


def _build_rotation(q):
    q = torch.nn.functional.normalize(q, dim=1)
    r, x, y, z = q.unbind(dim=1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
    ], dim=1).view(-1, 3, 3)
    return R


def _rotmat_to_quat(R):
    """Batched rotation matrix -> quaternion (w, x, y, z)."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    # 4 * q_i^2 for i in (w, x, y, z); pick the largest for numerical stability
    diag = torch.stack([
        1 + m00 + m11 + m22,
        1 + m00 - m11 - m22,
        1 - m00 + m11 - m22,
        1 - m00 - m11 + m22,
    ], dim=1)
    cand = torch.stack([
        torch.stack([diag[:, 0], m21 - m12, m02 - m20, m10 - m01], dim=1),
        torch.stack([m21 - m12, diag[:, 1], m01 + m10, m02 + m20], dim=1),
        torch.stack([m02 - m20, m01 + m10, diag[:, 2], m12 + m21], dim=1),
        torch.stack([m10 - m01, m02 + m20, m12 + m21, diag[:, 3]], dim=1),
    ], dim=1)  # (M, 4, 4)
    best = diag.argmax(dim=1)
    q = cand[torch.arange(R.shape[0], device=R.device), best]
    return torch.nn.functional.normalize(q, dim=1)


def _greedy_halvings(lam1, lam2, k):
    """Closed form of: repeat k times, halve the larger projected std.

    lam1 >= lam2 are the projected variances, k (M,) long.
    Returns (a, b): halvings of the major / minor axis, a + b = k.
    """
    rho = torch.sqrt(lam1 / lam2.clamp_min(1e-300)).clamp(1.0, 2.0 ** 40)
    # number of major halvings until the major std drops below the minor std
    m0 = torch.floor(torch.log2(rho)).long() + 1
    # after that the two axes alternate: minor, major, minor, ...
    a = torch.where(k <= m0, k, m0 + torch.div(k - m0, 2, rounding_mode="floor"))
    return a, k - a


@torch.no_grad()
def screen_major_axis_split(xyz, scaling, rotation, camera, k=1,
                            offset="normal", near=0.2):
    """Split every Gaussian into 2^k children in one shot.

    xyz      (M, 3) centers
    scaling  (M, 3) activated scales (std per local axis), i.e. get_scaling
    rotation (M, 4) quaternions (w, x, y, z), normalized or not
    camera   3DGS Camera
    k        int or (M,) long tensor: number of halvings (children = 2^k)
    offset   where the children are placed:
               "normal": mu + e1 * x1 + e2 * x2 with e1, e2 ~ N(0, 1) drawn
                         independently per child (original 3DGS style);
                         an axis that is not halved gets no offset
               "moment": deterministic grid along the split axes that keeps
                         the combined covariance equal to the parent's
             x1 / x2 are the 3D vectors that project onto the 1-sigma tips of
             the projected major / minor axis.

    Returns new_xyz (C, 3), new_scaling (C, 3, activated), new_rotation (C, 4),
            parent (C,) long: index into the M inputs for every child
            (children of one parent are contiguous).
    Gaussians behind the camera fall back to their two largest 3D axes.
    """
    out_dtype = xyz.dtype
    dev = xyz.device
    M = xyz.shape[0]
    f64 = torch.float64

    if not torch.is_tensor(k):
        k = torch.full((M,), int(k), dtype=torch.long, device=dev)
    k = k.to(device=dev, dtype=torch.long).clamp_min(0)

    R = _build_rotation(rotation).to(f64)
    S = scaling.to(f64)
    A = R * S[:, None, :]                      # R @ diag(S); Sigma = A A^T

    # ---- projection (same linearization as the rasterizer's computeCov2D)
    Wv = camera.world_view_transform.to(f64)
    Rc = Wv[:3, :3].T                          # p_cam = Rc @ p_world + t
    t = Wv[3, :3]
    p = xyz.to(f64) @ Rc.T + t

    tanx = math.tan(camera.FoVx * 0.5)
    tany = math.tan(camera.FoVy * 0.5)
    fx = camera.image_width / (2.0 * tanx)
    fy = camera.image_height / (2.0 * tany)

    z = p[:, 2]
    in_front = z > near
    zs = torch.where(in_front, z, torch.ones_like(z))
    px = torch.clamp(p[:, 0] / zs, -1.3 * tanx, 1.3 * tanx) * zs
    py = torch.clamp(p[:, 1] / zs, -1.3 * tany, 1.3 * tany) * zs

    J = torch.zeros((M, 2, 3), dtype=f64, device=dev)
    J[:, 0, 0] = fx / zs
    J[:, 0, 2] = -fx * px / (zs * zs)
    J[:, 1, 1] = fy / zs
    J[:, 1, 2] = -fy * py / (zs * zs)

    B = (J @ Rc) @ A                           # (M, 2, 3); Sigma2D = B B^T
    cov2d = B @ B.transpose(1, 2)
 
    # ---- closed-form eigen-decomposition of the symmetric 2x2 covariance
    c00, c01, c11 = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    mean = 0.5 * (c00 + c11)
    diff = 0.5 * (c00 - c11)
    rad = torch.sqrt(diff * diff + c01 * c01)
    lam1 = mean + rad                          # major variance
    lam2 = (mean - rad).clamp_min(0.0)         # minor variance
    v1 = torch.where((diff >= 0).unsqueeze(-1),
                     torch.stack([diff + rad, c01], dim=1),
                     torch.stack([c01, rad - diff], dim=1))
    v1n = v1.norm(dim=1, keepdim=True)
    e_x = torch.tensor([1.0, 0.0], dtype=f64, device=dev).expand(M, 2)
    v1 = torch.where(v1n > 0, v1 / v1n.clamp_min(1e-300), e_x)   # isotropic -> x axis
    v2 = torch.stack([-v1[:, 1], v1[:, 0]], dim=1)

    # ---- whitened unit directions: x_i = A u_i projects onto sqrt(lam_i) * v_i
    Bt = B.transpose(1, 2)
    u1 = (Bt @ v1.unsqueeze(-1)).squeeze(-1) / torch.sqrt(lam1.clamp_min(1e-300)).unsqueeze(-1)
    u2 = (Bt @ v2.unsqueeze(-1)).squeeze(-1) / torch.sqrt(lam2.clamp_min(1e-300)).unsqueeze(-1)

    # fallback for Gaussians behind the camera: two largest 3D axes
    used_screen = in_front & (lam1 > 1e-300)
    order = S.argsort(dim=1, descending=True)
    fb1 = torch.nn.functional.one_hot(order[:, 0], 3).to(f64)
    fb2 = torch.nn.functional.one_hot(order[:, 1], 3).to(f64)
    s_sorted = torch.gather(S, 1, order)
    sel = used_screen.unsqueeze(-1)
    u1 = torch.where(sel, u1, fb1)
    u2 = torch.where(sel, u2, fb2)
    lam1 = torch.where(used_screen, lam1, s_sorted[:, 0] ** 2)
    lam2 = torch.where(used_screen, lam2, s_sorted[:, 1] ** 2)
    u1 = torch.nan_to_num(u1)
    u2 = torch.nan_to_num(u2)

    # ---- how many times each projected axis is halved
    a, b = _greedy_halvings(lam1, lam2, k)
    c1 = 1.0 - torch.pow(torch.tensor(0.25, dtype=f64, device=dev), a.to(f64))   # 1 - 4^-a
    c2 = 1.0 - torch.pow(torch.tensor(0.25, dtype=f64, device=dev), b.to(f64))   # 1 - 4^-b

    # ---- child covariance (shared by all children of a parent)
    #   Sigma_child = A (I - c1 u1 u1^T - c2 u2 u2^T) A^T
    #   -> projected stds become sqrt(lam1) / 2^a and sqrt(lam2) / 2^b
    eye = torch.eye(3, dtype=f64, device=dev).expand(M, 3, 3)
    mid = (eye
           - c1[:, None, None] * u1.unsqueeze(-1) * u1.unsqueeze(-2)
           - c2[:, None, None] * u2.unsqueeze(-1) * u2.unsqueeze(-2))
    cov_child = A @ mid @ A.transpose(1, 2)
    cov_child = 0.5 * (cov_child + cov_child.transpose(1, 2))

    evals3, evecs3 = torch.linalg.eigh(cov_child)
    det = torch.linalg.det(evecs3)
    evecs3 = evecs3.clone()
    evecs3[:, :, 0] = evecs3[:, :, 0] * torch.sign(det).unsqueeze(-1)   # proper rotation
    child_scale = torch.sqrt(evals3.clamp_min(1e-20))
    child_quat = _rotmat_to_quat(evecs3)

    x1 = (A @ u1.unsqueeze(-1)).squeeze(-1)    # (M, 3) -> tip of projected major axis
    x2 = (A @ u2.unsqueeze(-1)).squeeze(-1)    # (M, 3) -> tip of projected minor axis

    # ---- children
    n_child = torch.pow(2, k)                                             # (M,)
    parent = torch.arange(M, device=dev).repeat_interleave(n_child)       # (C,)
    C = int(parent.numel())
    a_c, b_c = a[parent], b[parent]

    if offset == "normal":
        e1 = torch.randn(C, dtype=f64, device=dev) * (a_c > 0)
        e2 = torch.randn(C, dtype=f64, device=dev) * (b_c > 0)
    elif offset == "moment":
        # binary grid: level j moves the child by +-sqrt(3)/2 * 2^-j sigma
        first = torch.cumsum(n_child, dim=0) - n_child
        child_idx = torch.arange(C, device=dev) - first[parent]           # 0 .. 2^k - 1
        kmax = max(int(k.max().item()), 1)
        j = torch.arange(kmax, device=dev)
        w = torch.pow(torch.tensor(0.5, dtype=f64, device=dev), j.to(f64))
        beta = math.sqrt(0.75)
        bits1 = torch.bitwise_right_shift(child_idx[:, None], j[None, :]) & 1
        bits2 = torch.bitwise_right_shift(child_idx[:, None], a_c[:, None] + j[None, :]) & 1
        e1 = beta * ((2 * bits1 - 1).to(f64) * w * (j[None, :] < a_c[:, None])).sum(dim=1)
        e2 = beta * ((2 * bits2 - 1).to(f64) * w * (j[None, :] < b_c[:, None])).sum(dim=1)
    else:
        raise ValueError("offset must be 'normal' or 'moment'")

    new_xyz = xyz.to(f64)[parent] + e1[:, None] * x1[parent] + e2[:, None] * x2[parent]
    return (new_xyz.to(out_dtype), child_scale[parent].to(out_dtype),
            child_quat[parent].to(out_dtype), parent)


@torch.no_grad()
def densify_and_split_screen(gaussians, grads, grad_threshold, scene_extent,
                             viewpoint_camera, N=2, offset="normal"):
    """Drop-in replacement for GaussianModel.densify_and_split (N = 2, 4, 8, ...)."""
    if N < 2 or (N & (N - 1)) != 0:
        raise ValueError("N must be a power of two >= 2, got {}".format(N))
    n_init = gaussians.get_xyz.shape[0]
    padded_grad = torch.zeros((n_init,), device=gaussians.get_xyz.device)
    padded_grad[:grads.shape[0]] = grads.squeeze()
    mask = padded_grad >= grad_threshold
    mask = torch.logical_and(
        mask,
        torch.max(gaussians.get_scaling, dim=1).values > gaussians.percent_dense * scene_extent)
    sel_idx = mask.nonzero(as_tuple=True)[0]
    if sel_idx.numel() == 0:
        return 0

    new_xyz, new_scaling_act, new_rotation, parent_local = screen_major_axis_split(
        gaussians.get_xyz[sel_idx], gaussians.get_scaling[sel_idx], gaussians._rotation[sel_idx],
        viewpoint_camera, k=N.bit_length() - 1, offset=offset)
    parent_all = sel_idx[parent_local]

    args = [new_xyz,
            gaussians._features_dc[parent_all],
            gaussians._features_rest[parent_all],
            gaussians._opacity[parent_all],
            gaussians.scaling_inverse_activation(new_scaling_act),
            new_rotation]
    # newer 3DGS versions also carry tmp_radii through densification
    if "new_tmp_radii" in inspect.signature(gaussians.densification_postfix).parameters:
        args.append(gaussians.tmp_radii[parent_all])
    gaussians.densification_postfix(*args)

    prune_filter = torch.cat(
        (mask, torch.zeros(parent_all.numel(), device=mask.device, dtype=torch.bool)))
    gaussians.prune_points(prune_filter)
    return int(sel_idx.numel())
