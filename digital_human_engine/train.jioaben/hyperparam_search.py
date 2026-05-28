from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import FaceParamDataset, collect_samples, collate_fn, compute_stats
from loss import compute_multimodal_loss
from model import MultiModalFaceFormer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 GPU，请确认 CUDA 环境已正确安装")
    return torch.device("cuda")

def split_samples(samples: List[Dict[str, str]], train_ratio: float, seed: int) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    train_count = int(len(indices) * train_ratio)
    train_count = max(1, min(train_count, len(indices) - 1))
    train_samples = [samples[i] for i in indices[:train_count]]
    val_samples = [samples[i] for i in indices[train_count:]]
    return train_samples, val_samples


def run_epoch(
    model: MultiModalFaceFormer,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    sum_total = 0.0
    sum_vertex = 0.0
    sum_temporal = 0.0
    sum_emotion = 0.0
    steps = 0
    for batch in loader:
        audio = batch["audio"].to(device, non_blocking=True)
        emotion = batch["emotion"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            pred = model(audio, emotion, valid_mask=mask)
            loss, metrics = compute_multimodal_loss(pred, target, emotion, mask=mask)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        sum_total += float(metrics["loss_total"])
        sum_vertex += float(metrics["loss_vertex"])
        sum_temporal += float(metrics["loss_temporal"])
        sum_emotion += float(metrics["loss_emotion"])
        steps += 1
    if steps == 0:
        raise RuntimeError("DataLoader 未产生有效 batch")
    return {
        "loss_total": sum_total / steps,
        "loss_vertex": sum_vertex / steps,
        "loss_temporal": sum_temporal / steps,
        "loss_emotion": sum_emotion / steps,
    }


def parse_list_float(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_list_int(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def generate_grid_trials(args: argparse.Namespace) -> List[Dict[str, float | int | bool]]:
    combos = itertools.product(
        parse_list_float(args.learning_rates),
        parse_list_int(args.batch_sizes),
        parse_list_int(args.num_layers_list),
        parse_list_float(args.dropouts),
        parse_list_float(args.weight_decays),
        parse_list_float(args.scheduler_factors),
    )
    trials = []
    for lr, bs, layers, drop, wd, sch_factor in combos:
        trials.append(
            {
                "learning_rate": lr,
                "batch_size": bs,
                "num_layers": layers,
                "dropout": drop,
                "weight_decay": wd,
                "scheduler_factor": sch_factor,
                "use_cross_attention": args.use_cross_attention,
            }
        )
    return trials


def sample_random_trial(args: argparse.Namespace, rng: random.Random) -> Dict[str, float | int | bool]:
    return {
        "learning_rate": rng.choice(parse_list_float(args.learning_rates)),
        "batch_size": rng.choice(parse_list_int(args.batch_sizes)),
        "num_layers": rng.choice(parse_list_int(args.num_layers_list)),
        "dropout": rng.choice(parse_list_float(args.dropouts)),
        "weight_decay": rng.choice(parse_list_float(args.weight_decays)),
        "scheduler_factor": rng.choice(parse_list_float(args.scheduler_factors)),
        "use_cross_attention": args.use_cross_attention,
    }


def generate_bayes_trials(args: argparse.Namespace) -> List[Dict[str, float | int | bool]]:
    try:
        import optuna  # type: ignore
    except Exception:
        rng = random.Random(args.seed)
        return [sample_random_trial(args, rng) for _ in range(args.num_trials)]

    storage = f"sqlite:///{(Path(args.output_dir) / 'optuna_study.db').as_posix()}"
    study = optuna.create_study(study_name=args.study_name, direction="minimize", storage=storage, load_if_exists=True)
    lrs = parse_list_float(args.learning_rates)
    bss = parse_list_int(args.batch_sizes)
    layers = parse_list_int(args.num_layers_list)
    drops = parse_list_float(args.dropouts)
    wds = parse_list_float(args.weight_decays)
    schf = parse_list_float(args.scheduler_factors)
    trials: List[Dict[str, float | int | bool]] = []
    for _ in range(args.num_trials):
        t = study.ask()
        p = {
            "learning_rate": t.suggest_categorical("learning_rate", lrs),
            "batch_size": t.suggest_categorical("batch_size", bss),
            "num_layers": t.suggest_categorical("num_layers", layers),
            "dropout": t.suggest_categorical("dropout", drops),
            "weight_decay": t.suggest_categorical("weight_decay", wds),
            "scheduler_factor": t.suggest_categorical("scheduler_factor", schf),
            "use_cross_attention": args.use_cross_attention,
            "_optuna_trial_id": t.number,
        }
        trials.append(p)
    return trials


def build_dataloaders_for_trial(args: argparse.Namespace, batch_size: int) -> tuple[DataLoader, DataLoader, Dict[str, Dict[str, np.ndarray]]]:
    all_samples = collect_samples(args.root_dir)
    if len(all_samples) < 2:
        raise RuntimeError("样本数量不足，至少需要 2 条")
    train_samples, val_samples = split_samples(all_samples, args.train_ratio, args.seed)
    stats = compute_stats(train_samples) if args.normalize else None
    train_dataset = FaceParamDataset(args.root_dir, samples=train_samples, normalize=args.normalize, stats=stats)
    val_dataset = FaceParamDataset(args.root_dir, samples=val_samples, normalize=args.normalize, stats=stats)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, stats if stats is not None else {}


def get_trial_dir(output_dir: str | Path, trial_index: int) -> Path:
    return Path(output_dir) / "trials" / f"trial_{trial_index:04d}"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_trial_manifest(output_dir: Path, indexed_trials: Sequence[tuple[int, Dict[str, float | int | bool]]]) -> None:
    payload: List[Dict[str, Any]] = []
    generated_at = datetime.now().isoformat(timespec="seconds")
    for trial_index, params in indexed_trials:
        trial_dir = get_trial_dir(output_dir, trial_index)
        payload.append(
            {
                "trial_index": trial_index,
                "params": params,
                "trial_dir": str(trial_dir),
                "best_model_path": str(trial_dir / "best_model.pth"),
                "history_path": str(trial_dir / "history.json"),
                "summary_path": str(trial_dir / "trial_summary.json"),
                "generated_at": generated_at,
            }
        )
    write_json(output_dir / "trial_manifest.json", {"generated_at": generated_at, "trials": payload})


def run_single_trial(
    trial_index: int,
    params: Dict[str, float | int | bool],
    args_dict: Dict[str, object],
) -> Dict[str, object]:
    args = argparse.Namespace(**args_dict)
    set_seed(args.seed + trial_index)
    device = get_device()
    train_loader, val_loader, _ = build_dataloaders_for_trial(args, int(params["batch_size"]))
    model = MultiModalFaceFormer(
        dropout=float(params["dropout"]),
        num_layers=int(params["num_layers"]),
        use_cross_attention=bool(params["use_cross_attention"]),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(params["scheduler_factor"]),
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    trial_dir = get_trial_dir(args.output_dir, trial_index)
    trial_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = trial_dir / "best_model.pth"
    history_path = trial_dir / "history.json"
    summary_path = trial_dir / "trial_summary.json"
    write_json(
        trial_dir / "params.json",
        {
            "trial_index": trial_index,
            "params": params,
            "trial_dir": str(trial_dir),
            "best_model_path": str(best_model_path),
            "history_path": str(history_path),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    start = time.time()
    history: List[Dict[str, object]] = []
    if not args.quiet:
        print(
            f"[trial {trial_index:04d}] start "
            f"lr={params['learning_rate']} bs={params['batch_size']} layers={params['num_layers']} "
            f"dropout={params['dropout']} wd={params['weight_decay']} sch={params['scheduler_factor']}",
            flush=True,
        )
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step(val_metrics["loss_total"])
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": current_lr})
        improved = (best_val - val_metrics["loss_total"]) > args.early_stop_min_delta
        if improved:
            best_val = val_metrics["loss_total"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "params": params,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                },
                best_model_path,
            )
            if not args.quiet:
                print(
                    f"[trial {trial_index:04d}] new_best epoch={epoch} val_total={best_val:.6f}",
                    flush=True,
                )
        else:
            no_improve += 1
        if (not args.quiet) and (epoch == 1 or epoch % max(1, int(args.log_every)) == 0 or epoch == int(args.epochs)):
            print(
                f"[trial {trial_index:04d}] epoch={epoch:03d} "
                f"train_total={train_metrics['loss_total']:.6f} "
                f"val_total={val_metrics['loss_total']:.6f} "
                f"val_vertex={val_metrics['loss_vertex']:.6f} "
                f"val_temporal={val_metrics['loss_temporal']:.6f} "
                f"val_emotion={val_metrics['loss_emotion']:.6f} "
                f"lr={current_lr:.7f}",
                flush=True,
            )
        if no_improve >= args.early_stop_patience:
            if not args.quiet:
                print(f"[trial {trial_index:04d}] early_stop at epoch={epoch}", flush=True)
            break
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(
        summary_path,
        {
            "trial_index": trial_index,
            "params": params,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "duration_sec": round(time.time() - start, 2),
            "trial_dir": str(trial_dir),
            "best_model_path": str(best_model_path),
            "history_path": str(history_path),
            "status": "ok",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    if not args.quiet:
        print(
            f"[trial {trial_index:04d}] done best_val={best_val:.6f} best_epoch={best_epoch} "
            f"duration={round(time.time() - start, 2)}s",
            flush=True,
        )
    return {
        "trial_index": trial_index,
        "params": params,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "duration_sec": round(time.time() - start, 2),
        "trial_dir": str(trial_dir),
        "best_model_path": str(best_model_path),
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "status": "ok",
    }


def load_completed_trials(result_file: Path) -> Dict[int, Dict[str, object]]:
    done: Dict[int, Dict[str, object]] = {}
    if not result_file.exists():
        return done
    with result_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            done[int(item["trial_index"])] = item
    return done


def append_result(result_file: Path, item: Dict[str, object]) -> None:
    with result_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def persist_current_best(output_dir: Path, best: Dict[str, object]) -> None:
    payload = {
        "best_trial_index": best["trial_index"],
        "best_val": best["best_val"],
        "best_epoch": best["best_epoch"],
        "best_params": best["params"],
        "trial_dir": best["trial_dir"],
        "best_model_path": best["best_model_path"],
    }
    (output_dir / "current_best.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_reports(output_dir: Path, all_results: Sequence[Dict[str, object]]) -> None:
    sorted_results = sorted(all_results, key=lambda x: float(x["best_val"]))
    leaderboard = output_dir / "leaderboard.csv"
    with leaderboard.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "trial_index",
                "best_val",
                "best_epoch",
                "duration_sec",
                "learning_rate",
                "batch_size",
                "num_layers",
                "dropout",
                "weight_decay",
                "scheduler_factor",
                "use_cross_attention",
                "best_model_path",
            ]
        )
        for idx, r in enumerate(sorted_results, start=1):
            p = r["params"]
            writer.writerow(
                [
                    idx,
                    r["trial_index"],
                    r["best_val"],
                    r["best_epoch"],
                    r["duration_sec"],
                    p["learning_rate"],
                    p["batch_size"],
                    p["num_layers"],
                    p["dropout"],
                    p["weight_decay"],
                    p["scheduler_factor"],
                    p["use_cross_attention"],
                    r["best_model_path"],
                ]
            )
    best = sorted_results[0]
    best_trial_dir = Path(best["trial_dir"])
    shutil.copy2(best_trial_dir / "best_model.pth", output_dir / "best_model.pth")
    (output_dir / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "total_trials": len(all_results),
        "best_trial_index": best["trial_index"],
        "best_val": best["best_val"],
        "best_epoch": best["best_epoch"],
        "best_params": best["params"],
        "best_model_path": str(output_dir / "best_model.pth"),
        "source_trial_best_model_path": best["best_model_path"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_config(config_path: str | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件必须是 JSON 对象")
    return data


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config_data = load_json_config(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--root_dir", type=str, default=r"d:\ai大模型应用开发\第十七界服创\数据清理\data_feature\NoXI_expert")
    parser.add_argument("--output_dir", type=str, default=r"d:\ai大模型应用开发\第十七界服创\数据清理\checkpoints\hpo")
    parser.add_argument("--search_method", type=str, default="grid", choices=["grid", "bayes", "random"])
    parser.add_argument("--study_name", type=str, default="multimodal_hpo")
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_false", dest="normalize")
    parser.add_argument("--learning_rates", type=str, default="1e-4,5e-5,2e-4")
    parser.add_argument("--batch_sizes", type=str, default="4,8")
    parser.add_argument("--num_layers_list", type=str, default="2,4,6")
    parser.add_argument("--dropouts", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--weight_decays", type=str, default="0.0,1e-5,1e-4")
    parser.add_argument("--scheduler_factors", type=str, default="0.5,0.8")
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--use_cross_attention", action="store_true")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    if config_data:
        parser.set_defaults(**config_data)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "results.jsonl"
    completed = load_completed_trials(result_file)

    if args.search_method == "grid":
        trials = generate_grid_trials(args)
    elif args.search_method == "bayes":
        trials = generate_bayes_trials(args)
    else:
        rng = random.Random(args.seed)
        trials = [sample_random_trial(args, rng) for _ in range(args.num_trials)]

    indexed_trials = list(enumerate(trials))
    write_trial_manifest(output_dir, indexed_trials)
    pending = [(i, p) for i, p in indexed_trials if i not in completed]
    all_results = list(completed.values())
    total_trial_count = len(indexed_trials)
    done_count = len(all_results)
    if not args.quiet:
        print(
            f"hpo_start total={total_trial_count} completed={done_count} pending={len(pending)} "
            f"workers={args.max_workers} method={args.search_method}",
            flush=True,
        )

    current_best = min(all_results, key=lambda x: float(x["best_val"])) if all_results else None
    if current_best is not None:
        persist_current_best(output_dir, current_best)
        if not args.quiet:
            print(
                f"resume_best trial={current_best['trial_index']} "
                f"val={float(current_best['best_val']):.6f}",
                flush=True,
            )

    if args.max_workers <= 1:
        for i, p in pending:
            result = run_single_trial(i, p, vars(args))
            append_result(result_file, result)
            all_results.append(result)
            done_count += 1
            if current_best is None or float(result["best_val"]) < float(current_best["best_val"]):
                current_best = result
                persist_current_best(output_dir, current_best)
                print(
                    f"global_best_update trial={result['trial_index']} "
                    f"val={float(result['best_val']):.6f}",
                    flush=True,
                )
            if not args.quiet:
                print(f"hpo_progress {done_count}/{total_trial_count}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(run_single_trial, i, p, vars(args)) for i, p in pending]
            for fut in as_completed(futures):
                result = fut.result()
                append_result(result_file, result)
                all_results.append(result)
                done_count += 1
                if current_best is None or float(result["best_val"]) < float(current_best["best_val"]):
                    current_best = result
                    persist_current_best(output_dir, current_best)
                    print(
                        f"global_best_update trial={result['trial_index']} "
                        f"val={float(result['best_val']):.6f}",
                        flush=True,
                    )
                if not args.quiet:
                    print(f"hpo_progress {done_count}/{total_trial_count}", flush=True)

    if not all_results:
        raise RuntimeError("没有可用试验结果")
    write_reports(output_dir, all_results)
    best = min(all_results, key=lambda x: float(x["best_val"]))
    print(f"total_trials={len(all_results)}")
    print(f"best_trial={best['trial_index']}")
    print(f"best_val={float(best['best_val']):.6f}")
    print(f"best_params={json.dumps(best['params'], ensure_ascii=False)}")
    print(f"best_model={output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
