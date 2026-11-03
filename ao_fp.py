"""传统外层 AO、内层 FP。逐样本求解，收敛记录按实际吞吐量计算。"""
import torch

from system import (build_beamformer_system, compute_throughput,
                    initialize_beamformers, normalize_total_power,
                    physical_outputs, solve_beamformers_exact,
                    update_filters, update_fp_auxiliaries)


@torch.no_grad()
def solve_ao_fp(H, config, V0=None):
    if H.shape[0] != 1:
        raise ValueError("AO 基线逐样本运行，请传入 [1,N,K]，以便独立判断收敛。")
    s, a = config.system, config.ao
    V = initialize_beamformers(H, s) if V0 is None else normalize_total_power(V0, s.P)
    U = update_filters(H, V, s)
    current = float(compute_throughput(H, V, U, s)[1].item())
    history, inner_history, inner_counts = [current], [], []
    converged = False
    for outer in range(a.max_outer_iter):
        old_outer = current
        # U 在初始化/上一轮末尾已与 V 匹配，不重复求解同一滤波器。
        trace = [current]
        accepted = 0
        for _ in range(a.max_inner_iter):
            xi, y = update_fp_auxiliaries(H, V, U, s)
            M, b = build_beamformer_system(H, U, xi, y, s)
            candidate = normalize_total_power(solve_beamformers_exact(M, b), s.P, V)
            score = float(compute_throughput(H, candidate, U, s)[1].item())
            if not torch.isfinite(candidate).all() or not torch.isfinite(torch.tensor(score)):
                raise FloatingPointError("AO–FP 出现非有限数值。")
            if score < current:
                tolerance = max(1e-9, 64 * torch.finfo(H.real.dtype).eps) * max(1.0, abs(current))
                if current - score > tolerance:
                    raise RuntimeError("精确 FP 更新使吞吐量下降，请检查公式或数值精度。")
                break  # 舍入造成的小幅下降不接受
            difference = score - current
            V, current = candidate, score
            trace.append(current)
            accepted += 1
            if difference <= a.inner_tol * max(1.0, abs(current)):
                break
        # 与网络一样，外层结果使用最终波束匹配的闭式滤波器。
        U = update_filters(H, V, s)
        current = float(compute_throughput(H, V, U, s)[1].item())
        history.append(current)
        inner_history.append(trace)
        inner_counts.append(accepted)
        if abs(current - old_outer) <= a.outer_tol * max(1.0, abs(old_outer)):
            converged = True
            break
    result = physical_outputs(H, V, s, U)
    result.update(history=history, inner_history=inner_history,
                  outer_iters=len(history) - 1, inner_iters=sum(inner_counts),
                  inner_counts=inner_counts, converged=converged)
    return result
