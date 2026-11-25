"""Synthetic CPU-only integration check for the joint VAE/graph/predictor path."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor_joint"
sys.path.insert(0, str(VENDOR))
import runner as isolated_runner
model_module, trainer_base = isolated_runner.load_vendor("joint")

from src.model.stgcn_n.fewnode_stgcn import load_transferable_state
from src.model.stgcn_n.joint_stgcn import STGCN_n


def config():
    return {
        "dataset_name": "synthetic_cpu_only",
        "device": "cpu",
        "seed": 0,
        "train_node_ratio": 0.25,
        "train_node_ratio": 0.25,
        "window_size": 24,
        "horizon": 12,
        "input_size": 1,
        "history_steps": 12,
        "horizon_steps": 12,
        "anchor_lambda": 0.5,
        "aug_node": True,
        "vae_loss_weight": 2,
        "sim_loss": True,
        "aug_loss": "MSE",
        "model": {
            "num_nodes": 20,
            "in_channels": 1,
            "hidden_channels": 64,
            "out_channels": 12,
            "kernel_size": 3,
            "K": 2,
            "normalization": "sym",
            "bias": True,
        },
        "vae": {"input_dim": 1, "hidden_dim": 64, "latent_dim": 64,
                "num_layers": 2, "dropout": 0.1, "variance": 0.1},
        "graph": {"hidden_dim": 64, "provenance_tau": 0.5,
                  "provenance_alpha": 0.5, "node2edge_type": "cosine"},
        "dropout": 0.0,
    }


def load_source(model: STGCN_n, checkpoint: Path):
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(source, dict) or not all(isinstance(v, torch.Tensor) for v in source.values()):
        raise ValueError("expected a tensor-only source state_dict")
    target = model.state_dict()
    vae = {k: v for k, v in source.items() if k.startswith("vae.")}
    expected = {k: v for k, v in target.items() if k.startswith("vae.")}
    if not vae or set(vae) != set(expected) or any(vae[k].shape != expected[k].shape for k in vae):
        raise ValueError("source VAE state does not exactly match candidate VAE")
    target.update(vae)
    model.load_state_dict(target, strict=True)
    report = load_transferable_state(model, source)
    report["vae_tensor_count"] = len(vae)
    report["vae_parameter_elements"] = sum(value.numel() for value in vae.values())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ckpt", type=Path,
                        default=Path(r"C:\Users\beng1\Desktop\AAA2027\TVA_research\extensions\flow_s0\source\p04_full_1ch.pt"))
    args = parser.parse_args()
    if not args.source_ckpt.is_file():
        raise FileNotFoundError(f"frozen source checkpoint not found: {args.source_ckpt}")
    torch.set_num_threads(2)
    torch.manual_seed(19)
    cfg = config()
    dummy_nodes = {"train_nodes": list(range(5)), "val_nodes": list(range(5)),
                   "test_nodes": list(range(20))}
    model = isolated_runner.model_from_fixed_split(
        model_module, cfg, dummy_nodes, np.eye(20, dtype=np.float32)
    ).cpu()
    if model_module.STGCN_n is not STGCN_n or trainer_base.__name__ != "V10SeedTrainer":
        raise AssertionError("candidate runner did not resolve its joint model and V10 trainer")
    transfer = load_source(model, args.source_ckpt) if args.source_ckpt.is_file() else None

    # Fully synthetic: 5 labeled training nodes plus 5 VAE virtual nodes.
    history = torch.randn(2, 5, 1, 12)
    future = torch.randn(2, 5, 1, 12)
    model.train()
    prediction, future_aug, embedding, vae_loss = model(history, future)
    if prediction.shape != (2, 10, 12) or future_aug.shape != (2, 10, 1, 12):
        raise AssertionError("training-time VAE augmentation shape mismatch")
    total = prediction.square().mean() + future_aug.square().mean() + vae_loss
    total.backward()
    graph_grad = model.graph_learner.fc.weight.grad
    temporal_grad = model.st_block_1._temporal_conv1.causal_conv.weight.grad
    if graph_grad is None or not torch.isfinite(graph_grad).all() or graph_grad.abs().sum() <= 0:
        raise AssertionError("graph encoder did not receive a finite nonzero gradient")
    if temporal_grad is None or not torch.isfinite(temporal_grad).all() or temporal_grad.abs().sum() <= 0:
        raise AssertionError("temporal predictor did not receive a finite nonzero gradient")
    if not torch.isfinite(vae_loss) or not torch.isfinite(embedding).all():
        raise AssertionError("VAE path returned nonfinite values")

    # Evaluation sees all real nodes and must not create virtual nodes.
    model.eval()
    with torch.no_grad():
        all_node_history = torch.randn(1, 17, 1, 12)
        inference, unchanged_future, _, eval_vae_loss = model(all_node_history, torch.randn(1, 17, 1, 12))
    if inference.shape != (1, 17, 12) or unchanged_future.shape != (1, 17, 1, 12):
        raise AssertionError("all-node inference shape mismatch")
    if not torch.isfinite(inference).all() or eval_vae_loss.item() != 0.0:
        raise AssertionError("inference must be finite and bypass training augmentation")

    # Exercise one synthetic VAE epoch and one synthetic predictor/graph epoch
    # through the copied V10 trainer, with no real traffic values or file writes.
    training_cfg = config()
    training_cfg.update(device="cpu", model_name="STGCN_n", lr=0.002, weight_decay=0.0001,
                        milestones=[1], gamma=0.5, batch_size=2, epochs=2,
                        tolerance=20, train_normalized=False, fst_loss=True,
                        lambda_predictor_l2=1e-5)
    source_state = torch.load(args.source_ckpt, map_location="cpu", weights_only=True)
    vae_source = {k: v.detach().clone() for k, v in source_state.items() if k.startswith("vae.")}
    normalized_source = {k.removeprefix("module."): v for k, v in source_state.items()}
    predictor_source = {key: normalized_source[key].detach().clone() for key in transfer["loaded_keys"]}
    scaler = torch.tensor([0.0, 1.0], dtype=torch.float32).reshape(2, 1, 1, 1)
    training_batch = torch.randn(2, 5, 1, 24)
    trainer = isolated_runner.make_trainer(
        trainer_base, training_cfg, model, scaler, [training_batch], [training_batch], vae_source
    )
    trainer.predictor_source_state = predictor_source
    vae_before = model.vae.encoder.oneshot_fc_mu.weight.detach().clone()
    predictor_before = model.st_block_1._temporal_conv1.causal_conv.weight.detach().clone()
    graph_before = model.graph_learner.fc.weight.detach().clone()
    spatial_before = model.st_block_1._spatial_mix.neighbor_map.weight.detach().clone()
    vae_epoch_loss = isolated_runner.train_epoch_checked(trainer, 0)
    vae_updated = not torch.equal(vae_before, model.vae.encoder.oneshot_fc_mu.weight.detach())
    predictor_epoch_loss = isolated_runner.train_epoch_checked(trainer, 1)
    predictor_updated = not torch.equal(predictor_before, model.st_block_1._temporal_conv1.causal_conv.weight.detach())
    graph_updated = not torch.equal(graph_before, model.graph_learner.fc.weight.detach())
    spatial_updated = not torch.equal(spatial_before, model.st_block_1._spatial_mix.neighbor_map.weight.detach())
    if not vae_updated or not predictor_updated or not graph_updated or not spatial_updated:
        raise AssertionError("alternating synthetic V10 phases failed to update their intended module")
    if not math.isfinite(vae_epoch_loss) or not math.isfinite(predictor_epoch_loss):
        raise AssertionError("synthetic V10 mini-train returned a nonfinite loss")

    no_aug_cfg = config()
    no_aug_cfg["aug_node"] = False
    no_aug_model = STGCN_n(no_aug_cfg).cpu()
    no_aug_transfer = load_source(no_aug_model, args.source_ckpt)
    no_aug_training_cfg = dict(training_cfg, aug_node=False)
    no_aug_trainer = isolated_runner.make_trainer(
        trainer_base, no_aug_training_cfg, no_aug_model, scaler,
        [training_batch], [training_batch], vae_source
    )
    no_aug_trainer.predictor_source_state = {
        key: normalized_source[key].detach().clone() for key in no_aug_transfer["loaded_keys"]
    }
    no_aug_vae_before = no_aug_model.vae.encoder.oneshot_fc_mu.weight.detach().clone()
    no_aug_epoch_loss = isolated_runner.train_epoch_checked(no_aug_trainer, 0)
    no_aug_vae_unchanged = torch.equal(
        no_aug_vae_before, no_aug_model.vae.encoder.oneshot_fc_mu.weight.detach()
    )
    if not math.isfinite(no_aug_epoch_loss) or not no_aug_vae_unchanged:
        raise AssertionError("VAE-free ablation should train the predictor while bypassing the VAE")

    result = {
        "status": "passed",
        "device": "cpu",
        "runner_factory": "passed",
        "traffic_data_read": False,
        "source_checkpoint": str(args.source_ckpt.resolve()),
        "source_checkpoint_sha256": hashlib.sha256(args.source_ckpt.read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "forward_backward_optimizer_updates": 0,
        "synthetic_v10_mini_train_optimizer_updates": 2,
        "cuda_used": False,
        "synthetic_training_prediction_shape": list(prediction.shape),
        "synthetic_training_future_shape": list(future_aug.shape),
        "all_node_inference_shape": list(inference.shape),
        "graph_encoder_gradient_nonzero": True,
        "temporal_predictor_gradient_nonzero": True,
        "vae_loss_finite": math.isfinite(float(vae_loss.detach())),
        "synthetic_vae_epoch_loss": vae_epoch_loss,
        "synthetic_predictor_epoch_loss": predictor_epoch_loss,
        "vae_phase_updated_vae": vae_updated,
        "predictor_phase_updated_predictor": predictor_updated,
        "predictor_phase_updated_graph_encoder": graph_updated,
        "predictor_phase_updated_new_spatial_path": spatial_updated,
        "predictor_source_anchor_lambda": training_cfg["lambda_predictor_l2"],
        "vae_disabled_ablation_epoch_loss": no_aug_epoch_loss,
        "vae_disabled_ablation_bypassed_vae": no_aug_vae_unchanged,
        "source_transfer": transfer,
    }
    checks_dir = ROOT / "checks"
    checks_dir.mkdir(exist_ok=True)
    (checks_dir / "cpu_integration_smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
