import torch
from torch import nn

from system import (build_beamformer_system, initialize_beamformers,
                    normalize_total_power, physical_outputs,
                    solve_beamformers_exact, update_filters, update_fp_auxiliaries)


class UnfoldingLayer(nn.Module):
    def __init__(self, config, exact=False):
        super().__init__()
        self.system_config = config.system
        self.exact = exact
        self.corrections = nn.ParameterDict()
        if not exact:
            N, K = config.system.N, config.system.K
            real_dtype = torch.float64 if config.training.dtype == "complex128" else torch.float32
            for name in ("X", "Y", "Z"):
                real = torch.eye(N, dtype=real_dtype).repeat(K, 1, 1) if name == "X" else torch.zeros(K, N, N, dtype=real_dtype)
                self.corrections[name + "_real"] = nn.Parameter(real)
                self.corrections[name + "_imag"] = nn.Parameter(torch.zeros_like(real))
            self.corrections["o_real"] = nn.Parameter(torch.zeros(K, N, dtype=real_dtype))
            self.corrections["o_imag"] = nn.Parameter(torch.zeros(K, N, dtype=real_dtype))

    def complex_parameter(self, name):
        return torch.complex(self.corrections[name + "_real"], self.corrections[name + "_imag"])

    def forward(self, H, V):
        s = self.system_config
        U = update_filters(H, V, s)
        xi, y = update_fp_auxiliaries(H, V, U, s)
        M, b = build_beamformer_system(H, U, xi, y, s)
        if self.exact:
            candidate = solve_beamformers_exact(M, b)
        else:
            # 按矩阵-向量顺序求值，避免形成 M@Y 这样的矩阵-矩阵乘积。
            rhs = b.transpose(-1, -2).unsqueeze(-1)  # [batch,K,N,1]
            X, Y, Z = (self.complex_parameter(name) for name in ("X", "Y", "Z"))
            diagonal = M.diagonal(dim1=-2, dim2=-1).real
            diagonal = diagonal.clamp_min(torch.finfo(diagonal.dtype).tiny)
            value = (X @ rhs) / diagonal[:, None, :, None]
            value = value + M.unsqueeze(1) @ (Y @ rhs) + Z @ rhs
            value = value.squeeze(-1) + self.complex_parameter("o")
            candidate = value.transpose(-1, -2)
        return normalize_total_power(candidate, s.P, V)


class UnfoldingNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.layers = nn.ModuleList([
            UnfoldingLayer(config, exact=(i == config.training.num_layers - 1))
            for i in range(config.training.num_layers)
        ])

    def forward(self, H, V0=None, return_history=False):
        s = self.config.system
        V = initialize_beamformers(H, s) if V0 is None else normalize_total_power(V0, s.P)
        history = []
        if return_history:
            history.append(physical_outputs(H, V, s)["C_sum"].detach().cpu().tolist())
        for layer in self.layers:
            V = layer(H, V)
            if return_history:
                history.append(physical_outputs(H, V, s)["C_sum"].detach().cpu().tolist())
        result = physical_outputs(H, V, s)  # 最终 V 对应的 U 必须重算
        if return_history:
            result["layer_history"] = history
        return result
