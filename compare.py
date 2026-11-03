"""同一信道和初始化下评估 AO–FP 与展开网络，并保存结果和图片。"""
import argparse
import csv
import json
import platform
import statistics
import time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ao_fp import solve_ao_fp
from config import add_common_arguments, config_from_arguments, config_to_dict
from system import compute_throughput, initialize_beamformers, setup_runtime
from unfolding import UnfoldingNetwork


def evaluate_methods(config, samples=None, repeats=1):
    if repeats < 1 or (samples is not None and samples < 1):
        raise ValueError("samples/repeats 必须为正。")
    device, dtype = setup_runtime(config)
    test_path = config.data_dir / "test.pt"
    model_path = config.checkpoint_dir / "best_model.pt"
    if not test_path.exists() or not model_path.exists():
        raise FileNotFoundError("请先使用相同的 --smoke 设置运行 train.py，生成测试集和模型。")
    saved = torch.load(test_path, map_location="cpu", weights_only=True)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if saved["system"] != asdict(config.system) or checkpoint["config"]["system"] != asdict(config.system):
        raise ValueError("测试集/模型与当前系统配置不同，请使用对应配置或重新训练。")
    trained = checkpoint["config"]["training"]
    if trained["dtype"] != config.training.dtype:
        raise ValueError("评估精度必须与训练精度一致。")
    config.training.num_layers = trained["num_layers"]
    model = UnfoldingNetwork(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    count = len(saved["H"]) if samples is None else min(samples, len(saved["H"]))
    channels = saved["H"][:count].to(device=device, dtype=dtype)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed(function):
        durations, output = [], None
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            output = function()
            synchronize()
            durations.append((time.perf_counter() - start) * 1000)
        return output, statistics.mean(durations)

    rows, first_trace = [], None
    with torch.inference_mode():
        warm_H = channels[:1]
        warm_V = initialize_beamformers(warm_H, config.system)
        # 两种方法均预热；初始化和数据传输不计入求解/推理时间。
        solve_ao_fp(warm_H, config, warm_V)
        model(warm_H, warm_V)
        synchronize()
        for index in range(count):
            H = channels[index:index + 1]
            V0 = initialize_beamformers(H, config.system)
            functions = {"ao": lambda: solve_ao_fp(H, config, V0),
                         "net": lambda: model(H, V0)}
            outputs, times = {}, {}
            for name in (("ao", "net") if index % 2 == 0 else ("net", "ao")):
                outputs[name], times[name] = timed(functions[name])
            record = {"sample": index}
            for name, output in outputs.items():
                C, total = compute_throughput(H, output["V"], output["U"], config.system)
                record[name + "_C_sum"] = float(total.item())
                record[name + "_ms"] = times[name]
                record[name + "_power_error"] = float(output["power_relative_error"].item())
                for k, value in enumerate(C[0].tolist()):
                    record[f"{name}_C_user_{k + 1}"] = value
            record["ao_outer_iters"] = outputs["ao"]["outer_iters"]
            record["ao_inner_iters"] = outputs["ao"]["inner_iters"]
            record["ao_converged"] = outputs["ao"]["converged"]
            rows.append(record)
            if first_trace is None:
                trace = model(H, V0, return_history=True)["layer_history"]
                first_trace = {"ao": outputs["ao"]["history"], "net": [v[0] for v in trace]}
            print(f"Sample {index + 1}/{count}: AO={record['ao_C_sum']:.6f}, NET={record['net_C_sum']:.6f}")
    summary = {"num_samples": count, "timing_repeats": repeats,
               "device": str(device), "dtype": str(dtype), "torch": torch.__version__,
               "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
               "checkpoint_epoch": checkpoint["epoch"], "checkpoint_val_C_sum": checkpoint["val_C_sum"],
               "config": config_to_dict(config),
               "timing_scope": "batch=1; same device/dtype/H/V0; excludes IO, initialization, training and plotting"}
    for name in ("ao", "net"):
        values = [r[name + "_C_sum"] for r in rows]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[name] = {"mean_C_sum": mean, "std_C_sum": std,
                         "mean_ms": statistics.mean(r[name + "_ms"] for r in rows),
                         "max_power_error": max(r[name + "_power_error"] for r in rows)}
    differences = [r["net_C_sum"] - r["ao_C_sum"] for r in rows]
    summary["paired_mean_net_minus_ao"] = statistics.mean(differences)
    summary["ao_converged_fraction"] = statistics.mean(float(r["ao_converged"]) for r in rows)
    config.ensure_dirs()
    with (config.results_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (config.results_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (config.results_dir / "iteration_traces.json").write_text(json.dumps(first_trace, indent=2), encoding="utf-8")
    return {"rows": rows, "summary": summary, "traces": first_trace}


def plot_results(results, config):
    rows, summary, traces = results["rows"], results["summary"], results["traces"]
    names = ("ao", "net")
    labels = ("AO-FP", "Unfolding")
    colors = ("#2563a6", "#df8a24")

    def save(fig, filename):
        fig.savefig(config.results_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].bar(labels, [summary[n]["mean_C_sum"] for n in names], color=colors)
    axes[0].set_ylabel("Mean sum throughput (bit/Hz per frame)")
    axes[1].bar(labels, [summary[n]["mean_ms"] for n in names], color=colors)
    axes[1].set_ylabel("Mean online time (ms/sample)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    save(fig, "performance.png")

    fig, ax = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)
    x, y = [r["ao_C_sum"] for r in rows], [r["net_C_sum"] for r in rows]
    ax.scatter(x, y, s=22, alpha=0.7)
    low, high = min(x + y), max(x + y)
    padding = max(1e-6, (high - low) * 0.05)
    ax.plot([low - padding, high + padding], [low - padding, high + padding], "k--", linewidth=1)
    ax.set(xlabel="AO-FP sum throughput", ylabel="Unfolding sum throughput")
    ax.grid(alpha=0.25)
    save(fig, "paired_throughput.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, name, xlabel in zip(axes, names, ("AO outer iteration", "Network layer")):
        ax.plot(range(len(traces[name])), traces[name], "o-", markersize=3)
        ax.set(xlabel=xlabel, ylabel="Sum throughput", title=labels[names.index(name)])
        ax.grid(alpha=0.25)
    save(fig, "iterations_and_layers.png")

    history_path = config.results_dir / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
        fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
        ax.plot([r["epoch"] for r in history], [r["val_C_sum"] for r in history], label="Validation")
        train = [r for r in history if r["train_C_sum"] is not None]
        ax.plot([r["epoch"] for r in train], [r["train_C_sum"] for r in train], label="Training minibatches")
        ax.set(xlabel="Epoch", ylabel="Mean sum throughput")
        ax.legend()
        ax.grid(alpha=0.25)
        save(fig, "training_curve.png")


def main():
    parser = argparse.ArgumentParser(description="AO–FP 与深度展开网络的配对对比")
    add_common_arguments(parser)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    config = config_from_arguments(args)
    if args.check:
        setup_runtime(config)
        print(json.dumps({"config": config_to_dict(config), "inputs": {
            "test_data_exists": (config.data_dir / "test.pt").is_file(),
            "best_model_exists": (config.checkpoint_dir / "best_model.pt").is_file()}}, indent=2))
        return
    results = evaluate_methods(config, args.samples, args.repeats)
    plot_results(results, config)
    print(json.dumps(results["summary"], indent=2))
    print(f"Saved figures and results: {config.results_dir}")


if __name__ == "__main__":
    main()
