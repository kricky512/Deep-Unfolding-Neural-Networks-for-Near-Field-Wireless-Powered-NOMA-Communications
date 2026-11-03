import math
from dataclasses import asdict

import torch


def setup_runtime(config):
    t = config.training
    torch.set_num_threads(t.threads)
    torch.manual_seed(t.seed)
    if t.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA，请检查 GPU 版 PyTorch 或使用 --device cpu。")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(t.seed)
    return torch.device(t.device), getattr(torch, t.dtype)


def sort_channels(H):
    order = H.abs().square().sum(-2).argsort(dim=-1, descending=True, stable=True)
    return H.gather(-1, order.unsqueeze(-2).expand_as(H)), order


def generate_channels(num_samples, cfg, seed):
    
    generator = torch.Generator().manual_seed(seed)

    def uniform(shape, bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
            shape, generator=generator, dtype=torch.float64)

    def positions(shape, distances, angles):
        r = uniform(shape, distances)
        theta = uniform(shape, angles) * math.pi / 180
        return torch.stack((r * theta.cos(), r * theta.sin()), -1)

    B, K, S, N = num_samples, cfg.K, cfg.nlos_paths, cfg.N
    users = positions((B, K), cfg.user_distance_m, cfg.user_angle_deg)
    scatterers = positions((B, K, S), cfg.scatterer_distance_m, cfg.scatterer_angle_deg)
    r0 = torch.linalg.vector_norm(users, dim=-1)
    g0 = cfg.wavelength / (4 * math.pi * r0) * torch.exp(-2j * math.pi * r0 / cfg.wavelength)
    if S:
        path_length = (torch.linalg.vector_norm(users.unsqueeze(-2) - scatterers, dim=-1)
                       + torch.linalg.vector_norm(scatterers, dim=-1))
        weights = path_length.reciprocal()
        weights = weights / torch.linalg.vector_norm(weights, dim=-1, keepdim=True)
        amplitudes = g0.abs().unsqueeze(-1) * math.sqrt(10 ** (cfg.nlos_los_power_db / 10)) * weights
        reflection_phase = uniform((B, K, S), (-math.pi, math.pi))
        gn = amplitudes * torch.exp(1j * (reflection_phase - 2 * math.pi * path_length / cfg.wavelength))
    else:
        path_length = torch.empty(B, K, 0, dtype=torch.float64)
        gn = torch.empty(B, K, 0, dtype=torch.complex128)
    gains = torch.cat((g0.unsqueeze(-1), gn), -1)
    nodes = torch.cat((users.unsqueeze(-2), scatterers), -2)
    antenna_y = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * cfg.antenna_spacing
    antennas = torch.stack((torch.zeros_like(antenna_y), antenna_y), -1)
    distances = torch.linalg.vector_norm(nodes.unsqueeze(-2) - antennas, dim=-1)
    center_distances = torch.linalg.vector_norm(nodes, dim=-1).unsqueeze(-1)
    response = torch.exp(-2j * math.pi * (distances - center_distances) / cfg.wavelength)
    H = (gains.unsqueeze(-1) * response).sum(-2).transpose(-1, -2).contiguous()
    H, order = sort_channels(H)

    def reorder(value):
        shape = (B, K) + (1,) * (value.ndim - 2)
        return value.gather(1, order.reshape(shape).expand_as(value))

    return {"H": H, "users": reorder(users), "scatterers": reorder(scatterers),
            "gains": reorder(gains), "nlos_path_length": reorder(path_length),
            "order": order, "seed": seed, "system": asdict(cfg), "num_samples": B,
            "schema_version": 1}


def normalize_total_power(V, P, previous=None):
    norm = torch.linalg.vector_norm(V, dim=(-2, -1), keepdim=True)
    valid = norm > 1e-12
    if previous is None and not bool(valid.all()):
        raise ValueError("零/近零波束无法归一化。")
    candidate = math.sqrt(P) * V / norm.clamp_min(1e-12)
    return candidate if previous is None else torch.where(valid, candidate, previous)


def initialize_beamformers(H, cfg):
    norms = torch.linalg.vector_norm(H, dim=-2, keepdim=True)
    if bool((norms <= 1e-12).any()):
        raise ValueError("初始化要求每个用户的信道非零。")
    return normalize_total_power(math.sqrt(cfg.P / H.shape[-1]) * H / norms, cfg.P)


def received_energy_power(H, V):
    return (H.mH @ V).abs().square().sum(-1)


def compute_harvested_energy(H, V, cfg):
    return cfg.alpha0 * cfg.eta * cfg.T * received_energy_power(H, V)


def compute_uplink_power(H, V, cfg):
    return cfg.kappa * received_energy_power(H, V)


def effective_noise(V, cfg):
    return cfg.sigma2 / cfg.P * V.abs().square().sum((-2, -1))


def compute_sinr(H, V, U, cfg, homogenized=False):
    p = compute_uplink_power(H, V, cfg)
    weighted = (U.mH @ H).abs().square() * p.unsqueeze(-2)
    desired = weighted.diagonal(dim1=-2, dim2=-1)
    interference = torch.triu(weighted, diagonal=1).sum(-1)
    noise = effective_noise(V, cfg).unsqueeze(-1) if homogenized else cfg.sigma2
    denominator = interference + noise * U.abs().square().sum(-2)
    return desired / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)


def compute_throughput(H, V, U, cfg):
    C = (1 - cfg.alpha0) * cfg.T * torch.log1p(compute_sinr(H, V, U, cfg)) / math.log(2)
    return C, C.sum(-1)


def update_filters(H, V, cfg):
    """广义瑞利商闭式方向；直接求解线性方程，无需训练或递推。"""
    p = compute_uplink_power(H, V, cfg)
    s = effective_noise(V, cfg)
    eye = torch.eye(H.shape[-2], dtype=H.dtype, device=H.device)
    columns = []
    for k in range(H.shape[-1]):
        later = H[..., k + 1:]
        R = (later * p[..., k + 1:].unsqueeze(-2)) @ later.mH + s[..., None, None] * eye
        q = torch.linalg.solve(R, H[..., k:k + 1])
        columns.append(q / torch.linalg.vector_norm(q, dim=-2, keepdim=True))
    return torch.cat(columns, -1)


def update_fp_auxiliaries(H, V, U, cfg):
    G = U.mH @ H
    desired_gain = G.diagonal(dim1=-2, dim2=-1).abs()
    A = math.sqrt(cfg.kappa) * desired_gain.unsqueeze(-1) * (H.mH @ V)
    p = compute_uplink_power(H, V, cfg)
    B = torch.triu(G.abs().square() * p.unsqueeze(-2), diagonal=1).sum(-1)
    B = B + effective_noise(V, cfg).unsqueeze(-1) * U.abs().square().sum(-2)
    desired = A.abs().square().sum(-1)
    tiny = torch.finfo(B.dtype).tiny
    xi = desired / B.clamp_min(tiny)
    y = torch.sqrt(1 + xi).unsqueeze(-1) * A / (desired + B).clamp_min(tiny).unsqueeze(-1)
    return xi, y


def build_beamformer_system(H, U, xi, y, cfg):
    G = U.mH @ H
    q = y.abs().square().sum(-1)
    weights = cfg.kappa * (torch.triu(G.abs().square()) * q.unsqueeze(-1)).sum(-2)
    M = (H * weights.unsqueeze(-2)) @ H.mH
    nu = cfg.sigma2 / cfg.P * (q * U.abs().square().sum(-2)).sum(-1)
    eye = torch.eye(H.shape[-2], dtype=H.dtype, device=H.device)
    M = M + nu[..., None, None] * eye
    M = (M + M.mH) / 2  # 仅消除有限精度下的非厄米误差
    factors = math.sqrt(cfg.kappa) * torch.sqrt(1 + xi) * G.diagonal(dim1=-2, dim2=-1).abs()
    b = H @ (factors.unsqueeze(-1) * y)
    return M, b


def solve_beamformers_exact(M, b):
    """公共缩放不改变解；全部波束作为多个右端项同时求解。"""
    scale = M.diagonal(dim1=-2, dim2=-1).real.mean(-1).clamp_min(torch.finfo(M.real.dtype).tiny)
    return torch.linalg.solve(M / scale[..., None, None], b / scale[..., None, None])


def physical_outputs(H, V, cfg, U=None):
    U = update_filters(H, V, cfg) if U is None else U
    C, total = compute_throughput(H, V, U, cfg)
    return {"V": V, "U": U, "C_user": C, "C_sum": total,
            "p": compute_uplink_power(H, V, cfg),
            "Q": compute_harvested_energy(H, V, cfg),
            "power_relative_error": (V.abs().square().sum((-2, -1)) - cfg.P).abs() / cfg.P}
