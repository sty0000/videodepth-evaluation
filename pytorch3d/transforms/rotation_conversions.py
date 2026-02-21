import torch

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    q_abs = torch.stack([
        1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22,
    ], dim=-1)
    quat_by_max = torch.stack([
        torch.stack([q_abs[..., 0], m12 - m21, m20 - m02, m01 - m10], dim=-1),
        torch.stack([m12 - m21, q_abs[..., 1], m01 + m10, m02 + m20], dim=-1),
        torch.stack([m20 - m02, m01 + m10, q_abs[..., 2], m12 + m21], dim=-1),
        torch.stack([m01 - m10, m20 + m02, m12 + m21, q_abs[..., 3]], dim=-1),
    ], dim=-2)
    fltr = torch.argmax(q_abs, dim=-1)
    quat = quat_by_max.gather(-2, fltr[..., None, None].expand(*matrix.shape[:-2], 1, 4)).squeeze(-2)
    return quat / torch.sqrt(q_abs.gather(-1, fltr[..., None]) + 1e-9)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1).clamp(min=1e-9)
    o = torch.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))