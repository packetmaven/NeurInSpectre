#!/usr/bin/env python3
"""
Subnetwork hijack on CIFAR-10 (Table 5).

Table 5 claims (ResNet-50 + CIFAR-10):

    Subnet   Params   BD ASR    dAcc     NeuralCleanse   Spectral Sigs.
    -----    ------   -------   ------   -------------   --------------
    Cl. A    12%      97.1%     -0.8%    not detected    not detected
    Cl. B    8%       95.8%     -0.5%    not detected    not detected
    Cl. C    6%       94.2%     -0.3%    not detected    not detected
    BadNets  100%     99.1%     -1.2%    detected        detected

The headline claim is that a *quasi-independent* 6%-parameter subnetwork can
absorb a backdoor that (a) achieves ~94% ASR, (b) costs only 0.3 pp clean
accuracy, and (c) evades Neural Cleanse + Spectral Signatures detectors.

This script gives reviewers a rigorous, reviewer-runnable scaled-down
reproduction. The goals are:

  1. Identify a genuinely quasi-independent subnetwork (NOT by arbitrary
     k-means activation clustering) so that the clean-accuracy claim is
     testable in principle;
  2. Train the backdoor with a realistic BadNets poisoning-rate protocol,
     not pure-backdoor pushing;
  3. Evaluate with faithful Neural Cleanse (Wang et al. 2019) and
     Spectral Signatures (Tran et al. 2018) implementations;
  4. Report mean +/- std over multiple independent seeds;
  5. Contrast against a BadNets full-hijack baseline (same training,
     no subnetwork mask) to isolate the subnetwork's contribution.

Method rationale (why this captures the paper's stated mechanism)
=================================================================

The paper's §5.3 identifies quasi-independent subnetworks via
"Krylov spectral-gap DBSCAN on the Hessenberg projection's Ritz values."
The shipped `neurinspectre subnetwork_hijack identify` CLI uses KMeans
over penultimate-layer activations, which is NOT the same thing and does
NOT find quasi-independent subnetworks. A more faithful realisation must
identify neurons whose contribution to the clean loss is minimal.

We adopt **gradient-based neuron importance ranking** (Molchanov et al.
2017 "Pruning Convolutional Neural Networks for Resource Efficient
Inference"; Lee et al. 2019 "SNIP"). For each neuron i in the penultimate
layer we compute

    I(i) = E_{(x,y) ~ D_clean} [ | d L_ce(f(x), y) / d a_i(x) | ]

The K neurons with the smallest I(i) form the operational "quasi-
independent subnetwork": by construction, their activations have the
smallest first-order impact on the clean loss, so modifying the
last-linear weight columns they feed cannot distort the clean decision
boundary much.

Connection to the paper's stated method. Activation-gradient importance
is the diagonal approximation to Hessian-based importance: Fisher
information F_{ii} = E[ (d L/d theta_i)^2 ] relates to Hessian eigen-
values via the Gauss-Newton bound, and Ritz values of the Hessenberg
projection are Lanczos approximations to Hessian eigenvalues (Saad 2011,
Sidje 1998). All three recover the same "flat-direction" subspace in the
limit; gradient-importance is the 10x-cheaper reviewer-runnable variant.
See the NC / SS evaluation for literal reproductions of Wang 2019 and
Tran 2018.

Training protocol. We use BadNets-style poisoning (Gu et al. 2017):
each batch has a fraction p (default 10%) of samples with a 3x3 white
corner trigger and the label replaced by the target class; the
remaining (1-p) fraction retains original images+labels. The loss is
standard cross-entropy on the mixed batch. Subnetwork-restricted
fine-tuning zeros the last-Linear weight gradients outside the chosen
columns. The full-network update would be more faithful to the paper;
last-Linear-only is a scalable approximation that lets reviewers
reproduce the experiment in <10 minutes.

Scope caveats (read before comparing to paper numbers)
=====================================================

- Backbone: WRN-28-10 (Carmon2019Unlabeled, 36M params, published 89.7%
  clean acc). Paper uses ResNet-50 (~23.5M). Absolute numbers will
  differ; the mechanism is what we validate.
- We fine-tune only the last Linear layer (restricted to the chosen
  columns); the paper trains the full network. With full-network training,
  BD ASR should be 5-15 pp higher; with last-Linear only, clean-accuracy
  preservation should be easier because earlier layers are frozen.
- The paper's stated 6%-subnetwork identification method (Krylov-DBSCAN)
  is *not* available in the shipped CLI; we use gradient-importance as
  its practical proxy, documented in the header.
- Neural Cleanse / Spectral Signatures are run with the protocols in
  Wang et al. (2019) and Tran et al. (2018) respectively, but scaled to
  512 images and 50 optimisation iterations each for reviewer runtime.
  Full-protocol NC on a large validation set may yield different
  anomaly indices.

Usage
=====

    python scripts/reproduce_module_table5.py                  # defaults
    python scripts/reproduce_module_table5.py --n-seeds 3      # paper-style
    python scripts/reproduce_module_table5.py --subnet-fraction 0.06
    python scripts/reproduce_module_table5.py --baseline       # +BadNets
    python scripts/reproduce_module_table5.py --model resnet20 # no-download smoke

Runtime (MPS on M3 Pro, defaults):
    Per seed: ~5-8 min  (1 importance pass + 3 fine-tune epochs + NC + SS)
    --n-seeds 3:        ~20 min
    --baseline adds:    ~3 min per seed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-script imports

from neurinspectre.models.cifar10 import load_cifar10_model  # noqa: E402
from neurinspectre.evaluation.datasets import CIFAR10Dataset  # noqa: E402
from run_real_defense_experiments import load_carmon2019  # noqa: E402


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

TRIGGER_SIZE = 3
TRIGGER_VALUE = 1.0
TARGET_CLASS = 0  # arbitrary fixed target ("airplane" in CIFAR-10)


# ---------------------------------------------------------------------------
#  Device + trigger helpers
# ---------------------------------------------------------------------------


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _apply_trigger(x: torch.Tensor) -> torch.Tensor:
    """Apply BadNets-style 3x3 white patch to the bottom-right corner."""
    x = x.clone()
    x[..., -TRIGGER_SIZE:, -TRIGGER_SIZE:] = TRIGGER_VALUE
    return x


def _find_last_linear(model: nn.Module) -> nn.Linear:
    """Return the last nn.Linear submodule (the classification head)."""
    last: nn.Linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    assert last is not None, "Model has no nn.Linear head."
    return last


# ---------------------------------------------------------------------------
#  Phase 1: gradient-based neuron importance (the principled
#  quasi-independence criterion; see header docstring for rationale)
# ---------------------------------------------------------------------------


def compute_neuron_importance(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int = 16,
) -> np.ndarray:
    """Per-penultimate-neuron importance via clean-task gradient magnitude.

    For each neuron i in the penultimate (last-Linear input) layer:
        importance[i] = E_{(x,y)~D_clean} [ | dL_ce(f(x), y) / d a_i(x) | ]

    Low values => neuron's activation has little leverage on the clean
    loss => perturbing its last-Linear weight columns will not distort
    clean classification much => good backdoor-hijack candidate.

    Returns a NumPy array of shape [D] where D is the penultimate-layer
    width (640 for WRN-28-10, 64 for ResNet-20).
    """
    last_linear = _find_last_linear(model)
    D = last_linear.weight.shape[1]

    captured: Dict[str, torch.Tensor] = {"act": None}

    def _hook(_mod, inp, _out):
        a = inp[0] if isinstance(inp, tuple) else inp
        if a.ndim == 4:
            a = F.adaptive_avg_pool2d(a, 1).flatten(1)
        elif a.ndim != 2:
            a = a.reshape(a.size(0), -1)
        captured["act"] = a

    handle = last_linear.register_forward_hook(_hook)

    # We need gradients to flow through the network, but we do NOT update
    # any weights. Keep BN in eval mode so activations reflect deployment.
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()

    scores = np.zeros(D, dtype=np.float64)
    n_samples = 0

    try:
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            act = captured["act"]
            assert act is not None, "Forward hook failed to capture activation."
            grads = torch.autograd.grad(
                loss, act, retain_graph=False, create_graph=False
            )
            g_abs = grads[0].detach().abs().mean(dim=0).cpu().numpy()
            scores += g_abs * x.size(0)
            n_samples += x.size(0)
    finally:
        handle.remove()

    return scores / max(n_samples, 1)


def select_low_importance_subnet(
    importances: np.ndarray, fraction: float
) -> np.ndarray:
    """Return indices of the `fraction` lowest-importance neurons.

    fraction=0.06 gives the paper's smallest (Cl. C) subnetwork.
    """
    D = int(importances.size)
    k = max(1, int(round(fraction * D)))
    return np.argsort(importances)[:k]


def make_last_linear_mask(
    model: nn.Module, chosen_indices: np.ndarray
) -> Dict[str, torch.Tensor]:
    """Build per-parameter gradient masks for the last Linear.

    The mask zeros gradients for weight columns OUTSIDE the chosen
    subnetwork, restricting backdoor learning to those columns only.
    The bias is left unmasked: the backdoor training must be free to
    shift per-class logits.
    """
    last_linear = _find_last_linear(model)
    D = int(last_linear.weight.shape[1])
    in_mask = np.zeros(D, dtype=np.float32)
    in_mask[chosen_indices] = 1.0
    weight_mask = (
        torch.tensor(in_mask, dtype=torch.float32)
        .unsqueeze(0)
        .expand_as(last_linear.weight.detach())
        .clone()
    )
    bias_mask = torch.ones_like(last_linear.bias.detach())
    return {"weight": weight_mask, "bias": bias_mask}


# ---------------------------------------------------------------------------
#  Phase 2: BadNets-style poisoning-rate backdoor fine-tuning
# ---------------------------------------------------------------------------


def train_backdoor_poisoned(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    param_mask: Dict[str, torch.Tensor],
    epochs: int,
    lr: float,
    poison_rate: float = 0.1,
    target_class: int = TARGET_CLASS,
) -> None:
    """BadNets-style backdoor fine-tune, last-Linear only, with subnet mask.

    Each mini-batch mixes clean samples (p=1-poison_rate, original labels)
    with triggered samples (p=poison_rate, trigger stamped + label ->
    target_class). The loss is plain CE on the mixed batch. The mask
    blocks weight-column gradients outside the chosen subnetwork, so the
    classifier is forced to learn the trigger->target_class mapping *using
    only those columns*.

    This is the faithful poisoning protocol (Gu et al. 2017 "BadNets";
    Qi et al. 2022 "SRA"), not a pure-backdoor objective.
    """
    last_linear = _find_last_linear(model)
    for p in model.parameters():
        p.requires_grad_(False)
    last_linear.weight.requires_grad_(True)
    last_linear.bias.requires_grad_(True)

    weight_mask = param_mask["weight"].to(device)
    bias_mask = param_mask["bias"].to(device)

    opt = torch.optim.SGD(
        [last_linear.weight, last_linear.bias], lr=lr, momentum=0.9
    )

    model.train()
    for ep in range(epochs):
        tot_loss = 0.0
        n = 0
        correct_clean = 0
        correct_bd = 0
        n_bd = 0
        n_cl = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # Sample poisoning mask per-example
            poison_mask = torch.rand(x.size(0), device=device) < poison_rate
            x_mixed = x.clone()
            y_mixed = y.clone()
            if poison_mask.any():
                x_mixed[poison_mask] = _apply_trigger(x[poison_mask])
                y_mixed[poison_mask] = target_class

            opt.zero_grad(set_to_none=True)
            logits = model(x_mixed)
            loss = F.cross_entropy(logits, y_mixed)
            loss.backward()
            with torch.no_grad():
                last_linear.weight.grad *= weight_mask
                last_linear.bias.grad *= bias_mask
            opt.step()

            # Running diagnostics (not used for the reported metrics)
            preds = logits.argmax(dim=1).detach()
            if poison_mask.any():
                correct_bd += int(
                    (preds[poison_mask] == target_class).sum().item()
                )
                n_bd += int(poison_mask.sum().item())
            cl_mask = ~poison_mask
            if cl_mask.any():
                correct_clean += int(
                    (preds[cl_mask] == y[cl_mask]).sum().item()
                )
                n_cl += int(cl_mask.sum().item())

            tot_loss += float(loss.item()) * x.size(0)
            n += x.size(0)

        clean_acc = 100 * correct_clean / max(n_cl, 1)
        bd_acc = 100 * correct_bd / max(n_bd, 1)
        print(
            f"    [ep {ep + 1}/{epochs}] CE={tot_loss / max(n, 1):.4f}  "
            f"train_clean={clean_acc:.1f}%  train_bd={bd_acc:.1f}%"
        )
    model.eval()


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    target_class: int = TARGET_CLASS,
) -> Tuple[float, float]:
    """Return (clean_accuracy, backdoor_asr) on the given test loader.

    Clean accuracy: fraction of test images correctly classified.
    Backdoor ASR: conditional success rate on the subset of test images
    whose ground-truth class is != target_class (so that
    "classify as target" is not trivially the correct answer).
    """
    model.eval()
    clean_ok = 0
    clean_tot = 0
    bd_ok = 0
    bd_tot = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            preds_clean = model(x).argmax(1)
            clean_ok += int((preds_clean == y).sum().item())
            clean_tot += int(x.size(0))

            non_target = y != target_class
            if non_target.any():
                x_trig = _apply_trigger(x[non_target])
                preds_trig = model(x_trig).argmax(1)
                bd_ok += int((preds_trig == target_class).sum().item())
                bd_tot += int(non_target.sum().item())

    clean_acc = clean_ok / max(clean_tot, 1)
    bd_asr = bd_ok / max(bd_tot, 1)
    return clean_acc, bd_asr


# ---------------------------------------------------------------------------
#  Neural Cleanse (Wang et al. 2019, IEEE S&P)
# ---------------------------------------------------------------------------


def neural_cleanse(
    model: nn.Module,
    clean_loader: DataLoader,
    device: str,
    num_classes: int = 10,
    n_iter: int = 60,
    lr: float = 0.1,
    lambda_l1: float = 1e-2,
    n_batches_per_class: int = 2,
) -> Tuple[float, np.ndarray]:
    """Neural Cleanse anomaly index per Wang et al. 2019.

    For each target class c, reverse-engineer a minimal
    (L1-regularised) additive trigger (mask m in [0,1]^{HxW}, pattern p
    in [0,1]^{CxHxW}) that flips a reference batch of clean images to c:

        m*, p* = argmin_{m,p} CE(f((1-m) (*) x + m (*) p), c) + lambda * ||m||_1

    The per-class trigger L1 norm {||m_c*||_1}_{c=0..C-1} has a heavy
    tail for backdoored models: the backdoor's target class requires a
    much smaller trigger than benign classes. Wang et al. 2019 detect
    this via the MAD-based anomaly index

        anomaly(c) = (median_c ||m_c||_1 - ||m_c||_1) / (1.4826 * MAD)

    and flag a backdoor if max_c anomaly(c) >= 2. We return
    max_c anomaly(c) plus the full L1-norm vector for reference.

    Scope note. This is a faithful but scaled-down NC: n_iter is 60
    (paper 1000+), we use 2 batches of clean images per class rather
    than the full validation set, and we use a sigmoid-parametrised mask
    rather than a clamped-logit parametrisation. Absolute anomaly
    indices will be smaller than paper-grade NC but the ordering of
    clean vs backdoored classes should be preserved.
    """
    # Collect a pool of clean reference images
    x_pool: List[torch.Tensor] = []
    for i, (x, _y) in enumerate(clean_loader):
        if i >= n_batches_per_class:
            break
        x_pool.append(x.to(device))
    if not x_pool:
        return float("nan"), np.full(num_classes, np.nan)
    x_ref = torch.cat(x_pool, dim=0)
    C, H, W = x_ref.shape[1:]

    l1_norms = np.zeros(num_classes, dtype=np.float64)

    for target in range(num_classes):
        # Initialise mask logits so sigmoid(m) ~= 0 (small trigger) and
        # pattern near 0.5 (mid-gray).
        m_logits = torch.full(
            (1, 1, H, W), -4.0, device=device, requires_grad=True
        )
        p_logits = torch.zeros(
            (1, C, H, W), device=device, requires_grad=True
        )
        opt = torch.optim.Adam([m_logits, p_logits], lr=lr)
        y_target = torch.full(
            (x_ref.size(0),), target, dtype=torch.long, device=device
        )
        model.eval()
        for _ in range(n_iter):
            m = torch.sigmoid(m_logits)
            p = torch.sigmoid(p_logits)
            x_stamped = (1 - m) * x_ref + m * p
            logits = model(x_stamped)
            loss_ce = F.cross_entropy(logits, y_target)
            loss_l1 = m.abs().mean()
            loss = loss_ce + lambda_l1 * loss_l1 * (H * W)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            l1_norms[target] = float(torch.sigmoid(m_logits).abs().sum().item())

    # MAD-based anomaly index (Wang et al. 2019 Eq. 4)
    med = float(np.median(l1_norms))
    mad = float(np.median(np.abs(l1_norms - med))) + 1e-12
    consistency = 1.4826
    anomalies = (med - l1_norms) / (consistency * mad)
    anomaly_max = float(np.max(anomalies))
    return anomaly_max, l1_norms


# ---------------------------------------------------------------------------
#  Spectral Signatures (Tran et al. 2018, NeurIPS)
# ---------------------------------------------------------------------------


def spectral_signatures(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_class: int = TARGET_CLASS,
    trigger_fraction: float = 0.1,
) -> Tuple[float, float]:
    """Spectral Signatures per Tran et al. 2018.

    Protocol:
      1. Collect penultimate-layer activations of all samples that the
         model predicts as `target_class` on both clean and triggered
         inputs.
      2. Center activations; compute SVD; take top right-singular
         vector v_1.
      3. For each sample compute score_i = <a_i - mu, v_1>^2.
      4. The paper's detector: sort samples by score; the top
         1.5 x epsilon fraction (epsilon = expected poison fraction)
         are flagged as poisons.

    We return (detected_fraction_of_triggered, detected_fraction_of_clean).
    detected_fraction_of_triggered >= 0.85 is the common
    "spectral signature detected" threshold (Tran et al. 2018 §5).
    """
    last_linear = _find_last_linear(model)
    captured: List[torch.Tensor] = []

    def _hook(_mod, inp, _out):
        a = inp[0] if isinstance(inp, tuple) else inp
        if a.ndim == 4:
            a = F.adaptive_avg_pool2d(a, 1).flatten(1)
        elif a.ndim != 2:
            a = a.reshape(a.size(0), -1)
        captured.append(a.detach().cpu())

    model.eval()
    handle = last_linear.register_forward_hook(_hook)
    source_is_trigger: List[bool] = []
    predicted_target: List[bool] = []
    try:
        with torch.no_grad():
            # (a) Clean pass
            for x, _y in loader:
                x = x.to(device)
                logits = model(x)
                preds = logits.argmax(1)
                for p in preds:
                    source_is_trigger.append(False)
                    predicted_target.append(int(p.item()) == target_class)
            # (b) Triggered pass
            for x, _y in loader:
                x = x.to(device)
                x_trig = _apply_trigger(x)
                logits = model(x_trig)
                preds = logits.argmax(1)
                for p in preds:
                    source_is_trigger.append(True)
                    predicted_target.append(int(p.item()) == target_class)
    finally:
        handle.remove()

    if not captured:
        return float("nan"), float("nan")
    A = torch.cat(captured, dim=0).numpy().astype(np.float32)
    is_trig = np.array(source_is_trigger, dtype=bool)
    pred_tgt = np.array(predicted_target, dtype=bool)
    keep = pred_tgt
    if keep.sum() < 4:
        return float("nan"), float("nan")
    A_k = A[keep]
    is_trig_k = is_trig[keep]

    mu = A_k.mean(axis=0, keepdims=True)
    A_c = A_k - mu
    try:
        _u, _s, vh = np.linalg.svd(A_c, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    v1 = vh[0]
    scores = (A_c @ v1) ** 2

    N_kept = int(keep.sum())
    N_flag = max(1, int(round(1.5 * trigger_fraction * N_kept)))
    order = np.argsort(-scores)
    flagged = np.zeros(N_kept, dtype=bool)
    flagged[order[:N_flag]] = True

    n_trig = int(is_trig_k.sum())
    n_clean = int((~is_trig_k).sum())
    flagged_trig_frac = (
        float((flagged & is_trig_k).sum()) / max(n_trig, 1)
    )
    flagged_clean_frac = (
        float((flagged & ~is_trig_k).sum()) / max(n_clean, 1)
    )
    return flagged_trig_frac, flagged_clean_frac


# ---------------------------------------------------------------------------
#  Top-level experiment driver (one seed)
# ---------------------------------------------------------------------------


def _run_one_seed(
    *,
    args: argparse.Namespace,
    device: str,
    base_model_state: Dict[str, torch.Tensor],
    model_builder,
    id_loader: DataLoader,
    train_loader: DataLoader,
    test_loader: DataLoader,
    nc_loader: DataLoader,
    seed: int,
    run_baseline: bool,
) -> Dict[str, Any]:
    """Run Phase 1 (importance + selection), Phase 2 (poisoning train),
    Phase 3 (eval + detectors) for a single seed. Optionally run a
    BadNets full-hijack baseline on a fresh copy of the model for
    direct comparison."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Fresh model copy for subnet hijack
    model = model_builder()
    model.load_state_dict({k: v.clone() for k, v in base_model_state.items()})
    model.eval()

    # --- Phase 0: baseline clean acc + trivial BD ASR (on untouched model)
    base_acc, base_asr = evaluate(model, test_loader, device)

    # --- Phase 1: quasi-independent subnetwork via gradient importance
    importances = compute_neuron_importance(
        model, id_loader, device, max_batches=args.importance_batches
    )
    D = int(importances.size)
    chosen = select_low_importance_subnet(importances, args.subnet_fraction)
    subnet_pct = 100.0 * chosen.size / D
    imp_sel = float(importances[chosen].mean())
    imp_all = float(importances.mean())
    imp_rest = float(
        importances[np.setdiff1d(np.arange(D), chosen)].mean()
    )

    # --- Phase 2: subnetwork-restricted backdoor training
    mask = make_last_linear_mask(model, chosen)
    print(
        f"  [seed {seed}] subnetwork = {chosen.size}/{D} neurons "
        f"({subnet_pct:.1f}% of layer)  "
        f"mean_imp(chosen)={imp_sel:.2e}  mean_imp(rest)={imp_rest:.2e}"
    )
    train_backdoor_poisoned(
        model,
        train_loader,
        device,
        param_mask=mask,
        epochs=args.epochs,
        lr=args.lr,
        poison_rate=args.poison_rate,
    )
    post_acc, post_asr = evaluate(model, test_loader, device)

    # --- Phase 3: NC + SS detectors on the backdoored model
    nc_anomaly, nc_l1 = neural_cleanse(
        model, nc_loader, device, num_classes=10, n_iter=args.nc_iter,
    )
    ss_trig_flag, ss_clean_flag = spectral_signatures(
        model, test_loader, device, target_class=TARGET_CLASS,
        trigger_fraction=args.poison_rate,
    )

    # --- Optional BadNets baseline (no subnetwork mask) for contrast
    baseline_post_acc = baseline_post_asr = None
    baseline_nc_anomaly = None
    baseline_ss_trig_flag = None
    if run_baseline:
        model_bl = model_builder()
        model_bl.load_state_dict(
            {k: v.clone() for k, v in base_model_state.items()}
        )
        model_bl.eval()
        full_mask = make_last_linear_mask(model_bl, np.arange(D))  # no mask
        train_backdoor_poisoned(
            model_bl,
            train_loader,
            device,
            param_mask=full_mask,
            epochs=args.epochs,
            lr=args.lr,
            poison_rate=args.poison_rate,
        )
        baseline_post_acc, baseline_post_asr = evaluate(
            model_bl, test_loader, device
        )
        baseline_nc_anomaly, _ = neural_cleanse(
            model_bl, nc_loader, device, num_classes=10,
            n_iter=args.nc_iter,
        )
        baseline_ss_trig_flag, _ = spectral_signatures(
            model_bl, test_loader, device, target_class=TARGET_CLASS,
            trigger_fraction=args.poison_rate,
        )
        del model_bl

    return {
        "seed": int(seed),
        "subnet_fraction_pct": subnet_pct,
        "chosen_size": int(chosen.size),
        "total_neurons": int(D),
        "importance_chosen_mean": imp_sel,
        "importance_rest_mean": imp_rest,
        "importance_all_mean": imp_all,
        "baseline_clean_acc": base_acc,
        "baseline_bd_asr": base_asr,
        "subnet_clean_acc": post_acc,
        "subnet_bd_asr": post_asr,
        "subnet_delta_acc": post_acc - base_acc,
        "subnet_nc_anomaly": nc_anomaly,
        "subnet_nc_l1_norms": nc_l1.tolist(),
        "subnet_ss_triggered_flag_frac": ss_trig_flag,
        "subnet_ss_clean_flag_frac": ss_clean_flag,
        "badnets_clean_acc": baseline_post_acc,
        "badnets_bd_asr": baseline_post_asr,
        "badnets_delta_acc": (
            None if baseline_post_acc is None else baseline_post_acc - base_acc
        ),
        "badnets_nc_anomaly": baseline_nc_anomaly,
        "badnets_ss_triggered_flag_frac": baseline_ss_trig_flag,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Subnetwork hijack on CIFAR-10 (Table 5)"
    )
    parser.add_argument(
        "--subnet-fraction", type=float, default=0.06,
        help="Fraction of penultimate-layer neurons in the chosen quasi-"
             "independent subnetwork (default: 0.06 matches paper's Cl. C)",
    )
    parser.add_argument(
        "--importance-batches", type=int, default=16,
        help="Number of batches used to estimate neuron importance "
             "(default: 16; more = more stable ranking)",
    )
    parser.add_argument("--epochs", type=int, default=3,
                        help="Backdoor fine-tune epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=0.02,
                        help="Fine-tune learning rate (default: 0.02)")
    parser.add_argument(
        "--poison-rate", type=float, default=0.1,
        help="Fraction of each batch stamped with trigger + relabelled to "
             "TARGET_CLASS (default: 0.1 matches BadNets Gu et al. 2017)",
    )
    parser.add_argument("--n-train", type=int, default=2000,
                        help="Train images for backdoor fine-tune")
    parser.add_argument("--n-test", type=int, default=1000,
                        help="Test images for clean-acc + BD-ASR eval")
    parser.add_argument("--n-nc", type=int, default=256,
                        help="Clean images used for Neural Cleanse trigger RE")
    parser.add_argument("--nc-iter", type=int, default=60,
                        help="Optimisation iterations per class in NC")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of independent seeds (default: 3)")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="First seed (subsequent seeds are +1, +2, ...)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also train a BadNets full-hijack baseline "
                             "(no subnet mask) for direct comparison")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output-dir", default="results/module_table5")
    parser.add_argument("--model", default="standard_at",
                        choices=["standard_at", "resnet20"],
                        help="standard_at = Carmon2019 WRN-28-10 (requires "
                             "scripts/download_carmon2019.py); resnet20 = "
                             "no-download smoke (random weights).")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_label = (
        "WRN-28-10 (Carmon2019)" if args.model == "standard_at" else "ResNet-20 (untrained)"
    )
    print("=" * 76)
    print(" Table 5 — subnetwork hijack on CIFAR-10")
    print(f" Backbone: {model_label}  |  Dataset: CIFAR-10")
    print(f" subnet_fraction={args.subnet_fraction:.0%}  "
          f"poison_rate={args.poison_rate:.0%}  "
          f"epochs={args.epochs}  lr={args.lr}")
    print(f" seeds={args.n_seeds} (base={args.seed_base})  "
          f"baseline={args.baseline}  device={device}")
    print("=" * 76)

    # --- Load model factory + state ------------------------------------
    if args.model == "standard_at":
        ckpt_path = Path("models/cifar10/Linf/Carmon2019Unlabeled.pt")
        if not ckpt_path.exists():
            print(f"[error] {ckpt_path} not found.")
            print("[hint] Run `python scripts/download_carmon2019.py` first, or")
            print("       use `--model resnet20` for a smoke (random weights).")
            return 2
        base = load_carmon2019(str(ckpt_path), device)
    else:
        base = load_cifar10_model(model_name="resnet20", pretrained=True,
                                  device=device, normalize=True)
    base.eval()
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    def model_builder() -> nn.Module:
        if args.model == "standard_at":
            m = load_carmon2019(str(Path("models/cifar10/Linf/Carmon2019Unlabeled.pt")), device)
        else:
            m = load_cifar10_model(model_name="resnet20", pretrained=True,
                                   device=device, normalize=True)
        return m

    # --- Data loaders ---------------------------------------------------
    id_loader, _xa, _ya = CIFAR10Dataset.load(
        root="./data/cifar10", n_samples=512, seed=args.seed_base,
        batch_size=64, num_workers=0, split="test",
        download=True, pin_memory=False,
    )
    train_loader, _xtr, _ytr = CIFAR10Dataset.load(
        root="./data/cifar10", n_samples=args.n_train,
        seed=args.seed_base + 100,
        batch_size=64, num_workers=0, split="train",
        download=True, pin_memory=False,
    )
    test_loader, _xt, _yt = CIFAR10Dataset.load(
        root="./data/cifar10", n_samples=args.n_test,
        seed=args.seed_base + 200,
        batch_size=128, num_workers=0, split="test",
        download=False, pin_memory=False,
    )
    nc_loader, _xnc, _ync = CIFAR10Dataset.load(
        root="./data/cifar10", n_samples=args.n_nc,
        seed=args.seed_base + 300,
        batch_size=64, num_workers=0, split="test",
        download=False, pin_memory=False,
    )

    # --- Multi-seed loop ----------------------------------------------
    per_seed: List[Dict[str, Any]] = []
    for s_offset in range(args.n_seeds):
        seed = args.seed_base + s_offset
        print(f"\n--- seed {seed} ({s_offset + 1}/{args.n_seeds}) ---")
        r = _run_one_seed(
            args=args,
            device=device,
            base_model_state=base_state,
            model_builder=model_builder,
            id_loader=id_loader,
            train_loader=train_loader,
            test_loader=test_loader,
            nc_loader=nc_loader,
            seed=seed,
            run_baseline=args.baseline,
        )
        per_seed.append(r)
        print(
            f"  [seed {seed}] subnet: "
            f"clean={100 * r['subnet_clean_acc']:.2f}% (Δ={100 * r['subnet_delta_acc']:+.2f}%)  "
            f"BD ASR={100 * r['subnet_bd_asr']:.2f}%  "
            f"NC={r['subnet_nc_anomaly']:.2f}  "
            f"SS(trig)={100 * r['subnet_ss_triggered_flag_frac']:.1f}%"
        )
        if args.baseline and r["badnets_clean_acc"] is not None:
            print(
                f"  [seed {seed}] BadNets: "
                f"clean={100 * r['badnets_clean_acc']:.2f}% (Δ={100 * r['badnets_delta_acc']:+.2f}%)  "
                f"BD ASR={100 * r['badnets_bd_asr']:.2f}%  "
                f"NC={r['badnets_nc_anomaly']:.2f}  "
                f"SS(trig)={100 * r['badnets_ss_triggered_flag_frac']:.1f}%"
            )

    # --- Aggregate statistics (mean +/- std across seeds) --------------
    def _ms(vs: List[float]) -> Tuple[float, float]:
        arr = np.array([float(v) for v in vs if v is not None], dtype=np.float64)
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0

    clean_m, clean_s = _ms([r["subnet_clean_acc"] for r in per_seed])
    asr_m, asr_s = _ms([r["subnet_bd_asr"] for r in per_seed])
    da_m, da_s = _ms([r["subnet_delta_acc"] for r in per_seed])
    nc_m, nc_s = _ms([r["subnet_nc_anomaly"] for r in per_seed])
    ss_m, ss_s = _ms([r["subnet_ss_triggered_flag_frac"] for r in per_seed])
    base_acc_m, _ = _ms([r["baseline_clean_acc"] for r in per_seed])

    bl_clean_m, bl_clean_s = _ms([r["badnets_clean_acc"] for r in per_seed])
    bl_asr_m, bl_asr_s = _ms([r["badnets_bd_asr"] for r in per_seed])
    bl_da_m, bl_da_s = _ms([r["badnets_delta_acc"] for r in per_seed])
    bl_nc_m, bl_nc_s = _ms([r["badnets_nc_anomaly"] for r in per_seed])
    bl_ss_m, bl_ss_s = _ms([r["badnets_ss_triggered_flag_frac"] for r in per_seed])

    summary: Dict[str, Any] = {
        "config": vars(args),
        "device": device,
        "model_arch": "wrn_28_10_carmon2019" if args.model == "standard_at" else "resnet20",
        "n_seeds": args.n_seeds,
        "subnet_fraction": args.subnet_fraction,
        "baseline_clean_acc_mean": base_acc_m,
        "subnet": {
            "clean_acc_mean": clean_m,
            "clean_acc_std": clean_s,
            "bd_asr_mean": asr_m,
            "bd_asr_std": asr_s,
            "delta_acc_mean": da_m,
            "delta_acc_std": da_s,
            "nc_anomaly_mean": nc_m,
            "nc_anomaly_std": nc_s,
            "ss_triggered_flag_frac_mean": ss_m,
            "ss_triggered_flag_frac_std": ss_s,
        },
        "badnets_baseline": None if not args.baseline else {
            "clean_acc_mean": bl_clean_m,
            "clean_acc_std": bl_clean_s,
            "bd_asr_mean": bl_asr_m,
            "bd_asr_std": bl_asr_s,
            "delta_acc_mean": bl_da_m,
            "delta_acc_std": bl_da_s,
            "nc_anomaly_mean": bl_nc_m,
            "nc_anomaly_std": bl_nc_s,
            "ss_triggered_flag_frac_mean": bl_ss_m,
            "ss_triggered_flag_frac_std": bl_ss_s,
        },
        "per_seed": per_seed,
    }

    out_json = out_dir / "results.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # --- Pretty summary -----------------------------------------------
    print("\n" + "=" * 76)
    print(" TABLE 5 REPRODUCTION SUMMARY  (mean +/- std over "
          f"{args.n_seeds} seeds)")
    print("=" * 76)
    print(f" Subnetwork fraction      : {100 * args.subnet_fraction:.1f}% "
          f"(paper Cl. C = 6%)")
    print(f" Baseline clean accuracy  : {100 * base_acc_m:.2f}%")
    print(f" Subnet hijack:")
    print(f"   clean accuracy         : {100 * clean_m:.2f}% +/- {100 * clean_s:.2f}  "
          f"(Δ={100 * da_m:+.2f} +/- {100 * da_s:.2f})")
    print(f"   backdoor ASR           : {100 * asr_m:.2f}% +/- {100 * asr_s:.2f}")
    print(f"   NC anomaly index       : {nc_m:.2f} +/- {nc_s:.2f}  "
          f"(>=2.0 => detected)")
    print(f"   SS triggered-flag frac : {100 * ss_m:.2f}% +/- {100 * ss_s:.2f}  "
          f"(>=85% => detected)")
    if args.baseline:
        print(f" BadNets (no subnet mask):")
        print(f"   clean accuracy         : {100 * bl_clean_m:.2f}% +/- {100 * bl_clean_s:.2f}  "
              f"(Δ={100 * bl_da_m:+.2f} +/- {100 * bl_da_s:.2f})")
        print(f"   backdoor ASR           : {100 * bl_asr_m:.2f}% +/- {100 * bl_asr_s:.2f}")
        print(f"   NC anomaly index       : {bl_nc_m:.2f} +/- {bl_nc_s:.2f}")
        print(f"   SS triggered-flag frac : {100 * bl_ss_m:.2f}% +/- {100 * bl_ss_s:.2f}")
    print()
    print(" Paper Table 5 targets (ResNet-50; our repro is smaller):")
    print(f"   Cl. C (6%)      BD ASR=94.2%  Δ=-0.3%  NC=NO  SS=NO")
    print(f"   BadNets (100%)  BD ASR=99.1%  Δ=-1.2%  NC=YES SS=YES")
    print()
    print(" Scope: last-Linear-only poisoning (p={:.0%}); paper uses".format(
        args.poison_rate))
    print(" full-network training. Mechanism and qualitative detector")
    print(" contrast are what this script validates; absolute BD-ASR will")
    print(" be lower and |Δ| may be larger than the paper's headline.")
    print()
    print(f"[done] Full JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
