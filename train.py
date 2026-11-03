import argparse
import json
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import add_common_arguments, config_from_arguments, config_to_dict
from system import generate_channels, setup_runtime
from unfolding import UnfoldingNetwork


def prepare_data(config, regenerate=False):
    """独立种子生成三个数据集；默认复用匹配的数据，不静默覆盖。"""
    config.ensure_dirs()
    t = config.training
    specs = [("train", t.num_train, t.seed), ("val", t.num_val, t.seed + 1),
             ("test", t.num_test, t.seed + 2)]
    if not regenerate:
        for name, count, seed in specs:
            path = config.data_dir / f"{name}.pt"
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if (saved["system"] != asdict(config.system) or saved["seed"] != seed
                        or saved["num_samples"] != count):
                    raise ValueError(f"{path} 与配置不匹配。确认后使用 --regenerate-data 重新生成。")
    for name, count, seed in specs:
        path = config.data_dir / f"{name}.pt"
        if regenerate or not path.exists():
            torch.save(generate_channels(count, config.system, seed), path)
            print(f"Generated {name}: {count} samples -> {path}")


def load_channels(path, dtype):
    return torch.load(path, map_location="cpu", weights_only=True)["H"].to(dtype=dtype)


@torch.no_grad()
def validation_score(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for (H,) in loader:
        values = model(H.to(device))["C_sum"]
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("验证吞吐量出现非有限值。")
        total += float(values.sum().item())
        count += len(H)
    return total / count


def train_network(config):
    device, dtype = setup_runtime(config)
    t = config.training
    H_train = load_channels(config.data_dir / "train.pt", dtype)
    H_val = load_channels(config.data_dir / "val.pt", dtype)
    generator = torch.Generator().manual_seed(t.seed + 100)
    train_loader = DataLoader(TensorDataset(H_train), batch_size=t.batch_size,
                              shuffle=True, generator=generator)
    val_loader = DataLoader(TensorDataset(H_val), batch_size=t.batch_size)
    model = UnfoldingNetwork(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t.learning_rate)
    best_score = validation_score(model, val_loader, device)
    history = [{"epoch": 0, "train_C_sum": None, "val_C_sum": best_score}]
    checkpoint_path = config.checkpoint_dir / "best_model.pt"

    def save_best(epoch, score):
        torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": config_to_dict(config), "epoch": epoch,
                    "val_C_sum": score, "schema_version": 1}, checkpoint_path)

    save_best(0, best_score)  # 训练未改善时，保留初始化模型并如实记录 epoch=0
    print(f"Initial validation C_sum={best_score:.6f}; device={device}; dtype={dtype}")
    for epoch in range(1, t.num_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for (H,) in train_loader:
            H = H.to(device)
            optimizer.zero_grad(set_to_none=True)
            C_sum = model(H)["C_sum"]
            loss = -C_sum.mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("训练损失非有限，请减小学习率并检查配置。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total += float(C_sum.detach().sum().item())
            count += len(H)
        score = validation_score(model, val_loader, device)
        history.append({"epoch": epoch, "train_C_sum": total / count, "val_C_sum": score})
        if score > best_score:
            best_score = score
            save_best(epoch, score)
        print(f"Epoch {epoch:03d}: train C_sum={total / count:.6f}, val={score:.6f}, best={best_score:.6f}")
        (config.results_dir / "training_history.json").write_text(
            json.dumps({"config": config_to_dict(config), "history": history}, indent=2), encoding="utf-8")
    print(f"Saved best model: {checkpoint_path}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="近场 NOMA 无监督展开网络训练")
    add_common_arguments(parser)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--regenerate-data", action="store_true", help="显式重新生成数据集")
    args = parser.parse_args()
    config = config_from_arguments(args)
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
        config.validate()
    device, dtype = setup_runtime(config)
    if args.check:
        model = UnfoldingNetwork(config).to(device)
        H = generate_channels(2, config.system, config.training.seed)["H"].to(device=device, dtype=dtype)
        with torch.no_grad():
            output = model(H)
        print(json.dumps({"status": "forward_check_passed", "config": config_to_dict(config),
                          "mean_C_sum": float(output["C_sum"].mean().item()),
                          "max_power_relative_error": float(output["power_relative_error"].max().item()),
                          "real_trainable_parameters": sum(p.numel() for p in model.parameters()),
                          "cuda_available": torch.cuda.is_available()}, indent=2))
        return
    prepare_data(config, regenerate=args.regenerate_data)
    train_network(config)


if __name__ == "__main__":
    main()
