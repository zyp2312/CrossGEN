"""Isolated ST-FiT runner; baseline, VAE-transfer, or joint candidate.

Repair V2 fixes augmented batch offsets in every method. Joint restores the
source backbone and adds zero-gated sparse corrections. Historical windows,
stochastic evaluation, and baseline/V10 synthetic loss scales remain disclosed.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:128")
for _key, _cap in (("OMP_NUM_THREADS", 2), ("MKL_NUM_THREADS", 2),
                   ("OPENBLAS_NUM_THREADS", 1), ("NUMEXPR_NUM_THREADS", 1)):
    os.environ[_key] = str(min(int(os.environ.get(_key, _cap)), _cap))

import argparse
import copy
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parent
_VENDOR_METHOD = None


def configure_runtime(threads=2):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass  # Can only be set before the first parallel operation.


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def log_event(directory, event, **fields):
    record = {"event": event, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
    with (Path(directory) / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def historical_ranges(length):
    # Exactly the historical half-open ranges, including boundary overlap.
    a, b = int(0.6 * length), int(0.8 * length)
    result = {"train": range(0, a - 24), "val": range(a - 24, b - 24),
              "test": range(b - 24, length - 24)}
    if any(len(r) == 0 for r in result.values()):
        raise ValueError("Sequence too short for the historical 24-step windows")
    return result


def fit_scaler(flow, train_nodes, chunk_size=2048):
    """Train-node/train-window weighted population moments, never test values.

    NumPy reductions are single threaded. Float64 accumulation avoids building
    the historical repeated float32 window tensor; final scaler is float32.
    This is the same weighting with a documented floating-point roundoff change.
    """
    starts = historical_ranges(len(flow))["train"]
    n = len(starts)
    t = np.arange(n + 23, dtype=np.int64)
    weights = np.minimum(t, n - 1) - np.maximum(0, t - 23) + 1
    count = int(weights.sum()) * len(train_nodes)
    total = 0.0
    for offset in range(0, len(weights), chunk_size):
        w = weights[offset:offset + chunk_size]
        x = np.asarray(flow[offset:offset + len(w), train_nodes, 0], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite value in training scaler inputs")
        total += float(np.sum(x.astype(np.float64) * w[:, None], dtype=np.float64))
    mean = total / count
    variance = 0.0
    for offset in range(0, len(weights), chunk_size):
        w = weights[offset:offset + chunk_size]
        x = np.asarray(flow[offset:offset + len(w), train_nodes, 0], dtype=np.float64)
        variance += float(np.sum((x - mean) ** 2 * w[:, None], dtype=np.float64))
    std = math.sqrt(variance / count)
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Historical normalization requires a positive finite train std")
    return torch.tensor([mean, std], dtype=torch.float32).reshape(2, 1, 1, 1)


class LazyWindows(Dataset):
    def __init__(self, flow, nodes, starts, scaler):
        self.flow, self.nodes, self.starts = flow, np.asarray(nodes, dtype=np.int64), starts
        self.mean = np.float32(scaler[0].item())
        self.std = np.float32(scaler[1].item())

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        x = np.array(self.flow[start:start + 24, self.nodes, :], dtype=np.float32, copy=True)
        x = ((x - self.mean) / self.std).transpose(1, 2, 0).copy()
        return torch.from_numpy(x)


class BatchLimit:
    def __init__(self, loader, limit):
        self.loader, self.limit = loader, limit

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.limit)

    def __len__(self):
        return min(len(self.loader), self.limit)


def make_loader(flow, nodes, starts, scaler, batch_size, shuffle=False, limit=None):
    loader = DataLoader(LazyWindows(flow, nodes, starts, scaler), batch_size=batch_size,
                        shuffle=shuffle, num_workers=0, drop_last=False, pin_memory=False)
    return BatchLimit(loader, limit) if limit is not None else loader


def load_nodes(path, node_count, ratio):
    payload = json_read(path)
    nodes = {}
    for name in ("train_nodes", "val_nodes", "test_nodes"):
        values = payload[name]
        if not values or any(type(v) is not int for v in values):
            raise ValueError(f"{name} must contain integer positional IDs")
        if len(set(values)) != len(values) or min(values) < 0 or max(values) >= node_count:
            raise ValueError(f"Invalid/duplicate {name}")
        nodes[name] = values
    if len(nodes["train_nodes"]) != int(ratio * node_count):
        raise ValueError("Fixed train split size differs from historical floor(ratio*N)")
    if nodes["val_nodes"] != nodes["train_nodes"]:
        raise ValueError("val_node_ratio=0 requires the same ordered train/val node IDs")
    if set(nodes["test_nodes"]) != set(range(node_count)):
        raise ValueError("Test nodes must include every node exactly once")
    return nodes


def load_vendor(method):
    global _VENDOR_METHOD
    if _VENDOR_METHOD is not None and _VENDOR_METHOD != method:
        raise RuntimeError("Run each method in a separate process to isolate src imports")
    vendor_name = "joint" if method == "joint" else method
    vendor = ROOT / f"vendor_{vendor_name}"
    if _VENDOR_METHOD is None:
        if "src" in sys.modules:
            raise RuntimeError("Another src package is already imported")
        sys.path.insert(0, str(vendor))
    model_path = "src.model.stgcn_n.joint_stgcn" if method == "joint" else "src.model.stgcn_n.stgcn"
    model_module = importlib.import_module(model_path)
    base = importlib.import_module("src.trainer.train").Trainer
    if method in ("transfer", "joint"):
        spec = importlib.util.spec_from_file_location("historical_v10", vendor / "train_v10_seed.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base = module.V10SeedTrainer
    _VENDOR_METHOD = method
    return model_module, base


def model_from_fixed_split(model_module, cfg, nodes, adjacency):
    class ReadOnlyTrimer:
        def __init__(self, unused_cfg):
            pass

        def __call__(self):
            train, val = nodes["train_nodes"], nodes["val_nodes"]
            adj = torch.as_tensor(np.asarray(adjacency, dtype=np.float32))
            # Retain the historical unused rectangular test_graph. Dynamic
            # graph learning is fixed on, so all forward graphs are learned.
            graphs = [adj[train][:, train], adj[val][:, val], adj[val][:, train]]
            # Original DataTrimer restores these RNGs before model init.
            torch.manual_seed(cfg["seed"])
            random.seed(cfg["seed"])
            return [len(train), len(val), len(adj)], [train, val], graphs

    original = model_module.DataTrimer
    model_module.DataTrimer = ReadOnlyTrimer
    try:
        return model_module.STGCN_n(cfg=cfg)
    finally:
        model_module.DataTrimer = original


def load_source_vae(model, checkpoint):
    whole = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(whole, dict) or not all(isinstance(v, torch.Tensor) for v in whole.values()):
        raise ValueError("Source checkpoint must be a plain tensor state_dict")
    source = {k: v.detach().clone() for k, v in whole.items() if k.startswith("vae.")}
    target = {k: v for k, v in model.state_dict().items() if k.startswith("vae.")}
    if not source or set(source) != set(target):
        raise ValueError("Source and target VAE key sets must match exactly")
    for key, value in source.items():
        if value.shape != target[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"Incompatible or nonfinite source VAE tensor: {key}")
    state = model.state_dict()
    state.update(source)
    model.load_state_dict(state, strict=True)
    return source, {"tensor_count": len(source), "parameter_elements": sum(v.numel() for v in source.values()),
                    "source_full_elements": sum(v.numel() for v in whole.values())}


class MetricSums:
    """Streaming historical unmasked MAE/RMSE and y>=1 masked MAPE."""
    def __init__(self, groups):
        self.groups = groups
        self.sums = {g: {h: np.zeros(5, np.float64) for h in (3, 6, 12)} for g in groups}

    def update(self, y, prediction):
        for group, indices in self.groups.items():
            for horizon in (3, 6, 12):
                truth = y[:, indices, :horizon].detach().cpu()
                pred = prediction[:, indices, :horizon].detach().cpu()
                error = (truth - pred).abs()
                mask = truth >= 1
                # Float32 element operations match historical metrics; sums
                # use float64 for stable streaming aggregation across batches.
                self.sums[group][horizon] += np.asarray([
                    error.sum(dtype=torch.float64).item(),
                    error.square().sum(dtype=torch.float64).item(), error.numel(),
                    (error[mask] / truth[mask]).sum(dtype=torch.float64).item(), mask.sum().item()])

    def result(self):
        output = {}
        for group, horizons in self.sums.items():
            output[group] = {}
            for horizon, (ae, se, count, ape, masked) in horizons.items():
                output[group][str(horizon)] = {
                    "mae": ae / count if count else None,
                    "rmse": math.sqrt(se / count) if count else None,
                    "mape": ape / masked if masked else None,
                    "mape_percent": 100 * ape / masked if masked else None,
                    "elements": int(count), "mape_elements": int(masked)}
        return output


def make_trainer(base, cfg, model, scaler, train_loader, val_loader, source):
    class HistoricalLoop(base):
        def __init__(self):
            # Bypass only the original wandb/loader construction, not losses.
            self.configs, self.device, self.model = cfg, cfg["device"], model.to(cfg["device"])
            self.horizon, self.window_size = 12, 24
            self.scaler = scaler.to(self.device)
            self.train_iterator, self.val_iterator = train_loader, val_loader
            if cfg.get('repair_joint'):
                slow, fast = [], []
                for name, parameter in self.model.named_parameters():
                    is_slow = name.startswith(('graph_learner.', 'st_block_')) and '.residual_' not in name
                    (slow if is_slow else fast).append(parameter)
                groups = [{'params': fast, 'lr': cfg['lr'], 'role': 'head_vae_residual'},
                          {'params': slow, 'lr': cfg['lr'] * 0.1, 'role': 'source_backbone'}]
                self.opt = torch.optim.Adam(groups, weight_decay=cfg['weight_decay'])
                def protect_source(optimizer, args, kwargs):
                    if getattr(self, 'current_epoch', 0) < 10:
                        for parameter in optimizer.param_groups[1]['params']:
                            parameter.grad = None
                self.opt.register_step_pre_hook(protect_source)
            else:
                self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=cfg["milestones"], gamma=cfg["gamma"])
            self.vae_source_state = {k: v.detach().clone() for k, v in source.items()}
            self.lambda_l2 = 0.001
            self.predictor_source_state = {}
            self.lambda_predictor_l2 = float(cfg.get("lambda_predictor_l2", 0.0))

        @torch.no_grad()
        def evaluate_stream(self, loader, groups=None):
            self.model.eval()
            absolute, count = 0.0, 0
            metrics = MetricSums(groups) if groups is not None else None
            for values in loader:
                values = values.to(self.device)
                prediction, _, _, _ = self._iter(values=values)
                prediction = prediction * self.scaler[1][..., 0, :] + self.scaler[0][..., 0, :]
                truth = values[..., -12:] * self.scaler[1] + self.scaler[0]
                truth = truth[..., 0, :]
                if not torch.isfinite(prediction).all() or not torch.isfinite(truth).all():
                    raise FloatingPointError("Nonfinite validation/test values")
                # The original get_mae is sum(abs)/numel with no mask.
                absolute += (truth - prediction).abs().sum(dtype=torch.float64).item()
                count += truth.numel()
                if metrics is not None:
                    metrics.update(truth, prediction)
            if count == 0:
                raise ValueError("Empty evaluation loader")
            return absolute / count, metrics.result() if metrics is not None else None

    trainer = HistoricalLoop()
    trainer.successful_steps = 0
    def report_step(optimizer, args, kwargs):
        trainer.successful_steps += 1
        if trainer.successful_steps == 1 or trainer.successful_steps % 200 == 0:
            log_event(cfg["save_dir"], "optimizer_progress", steps=trainer.successful_steps,
                      epoch=getattr(trainer, "current_epoch", None),
                      gpu_allocated_bytes=torch.cuda.memory_allocated() if str(cfg["device"]).startswith("cuda") else 0)
    trainer.opt.register_step_post_hook(report_step)
    return trainer


def train_epoch_checked(trainer, epoch):
    trainer.current_epoch = epoch
    augmentation = bool(trainer.configs.get("aug_node"))
    trainer.aug_node_epoch = augmentation and epoch % 2 == 0
    trainer.aug_node_n_epoch = augmentation and epoch % 2 == 1
    checked = []

    def first_forward(module, inputs):
        if not checked:
            if not all(p.requires_grad for p in module.parameters()):
                raise RuntimeError("Historical epoch did not restore all requires_grad flags")
            checked.append(True)

    handle = trainer.model.register_forward_pre_hook(first_forward)
    try:
        loss = float(trainer._train_epoch(epoch))  # unchanged vendor function
    finally:
        handle.remove()
    if not checked or not math.isfinite(loss):
        raise FloatingPointError("Empty epoch or nonfinite training loss")
    for name, parameter in trainer.model.named_parameters():
        expected = True if not augmentation else (("vae" in name) if epoch % 2 == 0 else ("vae" not in name))
        if parameter.requires_grad != expected:
            raise RuntimeError(f"Unexpected freeze state after epoch {epoch}: {name}")
    return loss


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(),
            numpy_state[2], numpy_state[3], numpy_state[4]], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Cannot exactly resume CUDA RNG state on a CPU-only host")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def snapshot(trainer, epoch, best, best_epoch, best_loss, patience, fingerprint, scaler, done=False):
    return {"version": 1, "fingerprint": fingerprint, "epoch": epoch,
            "model": cpu_state(trainer.model), "optimizer": trainer.opt.state_dict(),
            "scheduler": trainer.scheduler.state_dict(), "rng": rng_state(),
            "requires_grad": {n: p.requires_grad for n, p in trainer.model.named_parameters()},
            "best": best, "best_epoch": best_epoch, "best_loss": best_loss,
            "patience": patience, "scaler": scaler.cpu(), "training_complete": done,
            "final_result": None}


def restore_training(trainer, state):
    trainer.model.load_state_dict(state["model"], strict=True)
    trainer.opt.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    for name, parameter in trainer.model.named_parameters():
        parameter.requires_grad = state["requires_grad"][name]
    restore_rng(state["rng"])


def resolve_config(payload, device, output, smoke=False):
    method = payload["method"]
    if method not in ("baseline", "transfer", "joint"):
        raise ValueError("method must be baseline, transfer, or joint")
    city = payload.get("city", payload.get("dataset"))
    if not city:
        raise ValueError("run config requires city or dataset")
    ratio = float(payload.get("ratio", payload.get("train_node_ratio", 0)))
    if ratio in (10, 25):
        ratio /= 100
    if ratio not in (0.1, 0.25):
        raise ValueError("Only train ratios 0.1 and 0.25 are supported")
    if int(payload.get("seed", 0)) != 0 or int(payload.get("node_seed", 0)) != 0:
        raise ValueError("This isolated experiment is fixed at seed=node_seed=0")
    for key, expected in (("epochs", 100), ("batch_size", 4), ("tolerance", 20),
                          ("history_steps", 12), ("horizon_steps", 12),
                          ("lambda_l2", 0.001 if method in ("transfer", "joint") else 0.0)):
        if key in payload and payload[key] != expected:
            raise ValueError(f"Registered {key} must remain {expected}")
    predictor_lambda = float(payload.get("lambda_predictor_l2", 1e-5 if method == "joint" else 0.0))
    expected_predictor_lambda = 1e-5 if method == "joint" else 0.0
    if not math.isfinite(predictor_lambda) or predictor_lambda < 0:
        raise ValueError("lambda_predictor_l2 must be finite and nonnegative")
    if method != "joint" and predictor_lambda != expected_predictor_lambda:
        raise ValueError(f"Registered lambda_predictor_l2 must remain {expected_predictor_lambda}")
    prepared = Path(payload["prepared_dir"]).resolve()
    split = Path(payload["split_path"]).resolve()
    cfg = yaml.safe_load((ROOT / "templates" / f"target_{int(ratio * 100)}.yml").read_text(encoding="utf-8"))
    cfg.update(dataset_name=city, device=device, seed=0, node_seed=0, train_node_ratio=ratio,
               data_root=str(prepared) + os.sep, networks_name="adj.npy", save_dir=str(output),
               batch_size=4, epochs=2 if smoke else 100, tolerance=20, wandb=False,
               split_ratio=[0.6, 0.2, 0.2], val_node_ratio=0, input_size=1, window_size=24, horizon=12)
    cfg["lambda_predictor_l2"] = predictor_lambda
    cfg['protocol_version'] = 'repair_v3'
    cfg['repair_joint'] = method == 'joint'
    if method == "joint":
        cfg.setdefault('graph', {}).update(correction_topk_per_type=4, correction_virtual_budget=1/3,
                                           residual_initial_scale=0.0, source_lr_factor=0.1,
                                           source_warmup_epochs=10)
    cfg["model"]["in_channels"] = cfg["vae"]["input_dim"] = cfg["aug_model"]["input_dim"] = 1
    cfg["eval_batch_size"] = int(payload.get("eval_batch_size", 2))
    if cfg["eval_batch_size"] < 1:
        raise ValueError("eval_batch_size must be positive")
    return method, prepared, split, cfg


def execute(args):
    execute_started = time.perf_counter()
    configure_runtime(2)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("--resume requires last.pt")
    if not args.resume and any((output / n).exists() for n in ("last.pt", "result.json", "events.jsonl")):
        raise FileExistsError("Run output already exists; use --resume for an interrupted run")
    payload = json_read(args.run_config)
    method, prepared, split, cfg = resolve_config(payload, args.device, output, args.smoke)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    if args.device.startswith("cuda"):
        # Leave physical VRAM for the desktop; fail visibly rather than rely on WDDM spill.
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.cuda.reset_peak_memory_stats()
    manifest = json_read(prepared / "manifest.json")
    flow = np.load(prepared / "flow.npy", mmap_mode="r", allow_pickle=False)
    if flow.ndim != 3 or flow.shape[2] != 1:
        raise ValueError("Prepared flow must be [T,N,1]")
    adjacency = np.load(prepared / "adj.npy", allow_pickle=False)
    if adjacency.shape != (flow.shape[1], flow.shape[1]) or not np.isfinite(adjacency).all():
        raise ValueError("Adjacency must be finite and match all prepared nodes")
    cfg["num_nodes"] = cfg["model"]["num_nodes"] = int(flow.shape[1])
    nodes = load_nodes(split, flow.shape[1], cfg["train_node_ratio"])
    source_path = Path(args.source_ckpt or payload.get("source_ckpt", ROOT / "source/p04_full_1ch.pt")).resolve() if method in ("transfer", "joint") else None
    identity = {"config": cfg, "method": method, "smoke": args.smoke,
                "smoke_batches": args.smoke_batches if args.smoke else None,
                "prepared_manifest_sha256": sha256(prepared / "manifest.json"), "split_sha256": sha256(split),
                "source_sha256": sha256(source_path) if source_path else None,
                "vendor": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / f"vendor_{'joint' if method == 'joint' else method}").rglob("*.py"))},
                "runner_sha256": sha256(__file__)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    state = torch.load(output / "last.pt", map_location="cpu", weights_only=True) if args.resume else None
    if state and state["fingerprint"] != fingerprint:
        raise ValueError("Resume configuration, code, source, or split differs from saved run")
    if state and state.get("final_result") is not None:
        atomic_json(output / "result.json", state["final_result"])
        print("Run already complete; final result restored without training/testing", flush=True)
        return state["final_result"]
    if args.resume and (output / "result.json").exists():
        raise ValueError("Result exists without a matching finalized checkpoint")
    scaler = state["scaler"] if state else fit_scaler(flow, nodes["train_nodes"])
    starts = historical_ranges(len(flow))
    limit = args.smoke_batches if args.smoke else None
    train_loader = make_loader(flow, nodes["train_nodes"], starts["train"], scaler, cfg["batch_size"], True, limit)
    val_loader = make_loader(flow, nodes["val_nodes"], starts["val"], scaler, cfg["eval_batch_size"], False, limit)
    seed_all(0)
    model_module, base = load_vendor(method)
    model = model_from_fixed_split(model_module, cfg, nodes, adjacency)
    source, source_info = load_source_vae(model, source_path) if source_path else ({}, None)
    predictor_source = {}
    if method == "joint":
        from src.model.stgcn_n.fewnode_stgcn import load_transferable_state
        source_state = torch.load(source_path, map_location="cpu", weights_only=True)
        predictor_info = load_transferable_state(model, source_state)
        normalized_source = {k.removeprefix("module."): v for k, v in source_state.items()}
        predictor_source = {key: normalized_source[key].detach().clone()
                            for key in predictor_info["loaded_keys"]}
        source_info = {"vae": source_info, "predictor": predictor_info}
    trainer = make_trainer(base, cfg, model, scaler, train_loader, val_loader, source)
    trainer.predictor_source_state = predictor_source
    best, best_epoch, best_loss, patience, first_epoch = None, None, math.inf, 0, 0
    if state:
        restore_training(trainer, state)
        best, best_epoch, best_loss, patience = (state[k] for k in ("best", "best_epoch", "best_loss", "patience"))
        first_epoch = state["epoch"] + 1
    atomic_json(output / "config.json", {"identity": identity, "fingerprint": fingerprint, "prepared_manifest": manifest,
                 "split_path": str(split), "source_path": str(source_path) if source_path else None,
                 "source_info": source_info, "scaler": scaler.flatten().tolist(),
                 "scaler_accumulation": "float64 weighted population moments; float32 output",
                 "window_ranges": {k: [r.start, r.stop] for k, r in starts.items()},
                 "historical_caveats": ["Boundary-overlapping windows retained", "Augmented graph batch offsets FIXED in all methods",
                     "Baseline synthetic MAE is denormalized; V10 synthetic MAE is normalized",
                     "Evaluation remains stochastic; omitted per-epoch test changes historical RNG consumption",
                     "Train batch4 (real-data 8GB capacity amendment); eval/test batch2; streaming float64 metric sums"]})
    log_event(output, "resume" if state else "start", method=method, smoke=args.smoke,
              next_epoch=first_epoch, scaler=scaler.flatten().tolist(), source_info=source_info)
    if not (state and state["training_complete"]):
        for epoch in range(first_epoch, cfg["epochs"]):
            train_loss = train_epoch_checked(trainer, epoch)
            eval_loss = None
            if epoch % 2 == 1:
                trainer.scheduler.step()
                eval_loss, _ = trainer.evaluate_stream(val_loader)
                if eval_loss < best_loss:
                    best_loss, best_epoch, patience = eval_loss, epoch, 0
                    best = cpu_state(trainer.model)
                    atomic_torch(output / "best.pt", best)
                else:
                    patience += 1
            done = epoch + 1 >= cfg["epochs"] or patience >= cfg["tolerance"]
            state = snapshot(trainer, epoch, best, best_epoch, best_loss, patience, fingerprint, scaler, done)
            atomic_torch(output / "last.pt", state)
            phase = ("vae" if epoch % 2 == 0 else "predictor_graph") if cfg.get("aug_node") else "joint_predictor"
            log_event(output, "epoch", epoch=epoch, phase=phase,
                      train_loss=train_loss, eval_loss=eval_loss, best_epoch=best_epoch,
                      best_val_loss=None if best is None else best_loss, patience=patience,
                      lr=trainer.opt.param_groups[0]["lr"], freeze_restored=True)
            if method == 'joint':
                log_event(output, 'repair_diagnostics', epoch=epoch,
                          residual_scales=[float(torch.tanh(block.residual_scale).detach().cpu())
                                           for block in (trainer.model.st_block_1, trainer.model.st_block_2)],
                          source_frozen=epoch < 10,
                          group_lrs=[group['lr'] for group in trainer.opt.param_groups])
            if done:
                break
    if best is None:
        raise RuntimeError("No finite validation-selected checkpoint")
    atomic_torch(output / "best.pt", best)  # Repair possible interrupted best/last write ordering.
    result = {"status": "smoke_complete" if args.smoke else "complete", "formal_complete": not args.smoke,
              "dataset": cfg["dataset_name"], "method": method, "ratio": cfg["train_node_ratio"],
              "seed": 0, "node_seed": 0, "best_epoch": best_epoch, "best_val_loss": best_loss,
              "scaler": scaler.flatten().tolist(), "fingerprint": fingerprint,
              "best_checkpoint_sha256": sha256(output / "best.pt"),
              "test_evaluations": 0, "metrics": None}
    if not args.smoke:
        # This is the only place a test dataset/iterator is constructed.
        trainer.model.load_state_dict(best, strict=True)
        test_nodes = nodes["test_nodes"]
        groups = {"all": list(range(len(test_nodes)))}
        test_loader = make_loader(flow, test_nodes, starts["test"], scaler, cfg["eval_batch_size"])
        test_loss, metrics = trainer.evaluate_stream(test_loader, groups)
        result.update(test_evaluations=1, test_loss=test_loss, metrics=metrics)
    result["resources"] = {
        "device": args.device, "cpu_threads": torch.get_num_threads(),
        "current_invocation_wall_seconds": time.perf_counter() - execute_started,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else 0,
        "gpu_allocator_limit_fraction": 0.8 if args.device.startswith("cuda") else None,
        "gpu_allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
        "torch": str(torch.__version__), "python": sys.version,
        "note": "Invocation time excludes earlier attempts after a resume; queue ledger preserves attempt resource records."
    }
    state["final_result"] = result
    atomic_torch(output / "last.pt", state)
    atomic_json(output / "result.json", result)
    log_event(output, "finished", **result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-ckpt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=1)
    args = parser.parse_args()
    if args.smoke_batches < 1:
        parser.error("--smoke-batches must be positive")
    execute(args)


if __name__ == "__main__":
    main()
