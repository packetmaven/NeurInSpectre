#!/usr/bin/env python3
"""NeurInSpectre — Attention Security Analysis (heatmap + token anomaly scores).

Restores the screenshot-style visualization:
- Left: token×token attention pattern heatmap (averaged across heads; layer-selectable)
- Right: token-level anomaly scores (IsolationForest over attention-derived features)
- Bottom: concise security findings + actionable blue/red team next steps

Key design goals:
- Real-data-driven: uses real model attentions (or a hidden-state similarity proxy if attentions are unavailable).
- Configurable from CLI: per-layer or all-layers (average) + tunable IsolationForest.
- Practical: highlights likely control/delimiter/payload anchor tokens and high-influence attractors.

This module is intended for authorized red/blue team testing and incident triage.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

# Reuse robust attention extraction already present in NeurInSpectre.
from .attention_heatmap import _run_with_attentions, _select_device, _tokenize


@dataclass
class AttentionSecurityConfig:
    model_name: str
    prompt: str

    # Layer selection:
    # - 'all': average across layers (optionally constrained by layer_start/layer_end)
    # - integer-like string: select single layer
    layer: str = 'all'
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None

    max_tokens: int = 128
    device: str = 'auto'

    output_png: str = '_cli_runs/attention_security.png'
    out_json: str = '_cli_runs/attention_security.json'
    out_html: str = '_cli_runs/attention_security.html'

    # IsolationForest
    contamination: str = 'auto'  # 'auto' or float-like string
    n_estimators: int = 256
    seed: int = 0

    title: str = 'NeurInSpectre — Attention Security Analysis'


_SUSPICIOUS_SUBSTRINGS = (
    # instruction hierarchy conflicts / injection cue words
    'ignore', 'previous', 'instruction', 'instructions', 'system', 'developer', 'policy',
    'jailbreak', 'bypass', 'override', 'reveal', 'leak', 'exfil',
    # data secrets / API-style
    'secret', 'api', 'key', 'password', 'token', 'ssh',
    # tool-ish
    'curl', 'wget', 'http', 'https',
)


def _prompt_sha16(prompt: str) -> str:
    return hashlib.sha256((prompt or '').encode('utf-8', errors='ignore')).hexdigest()[:16]


def _clean_tok_display(t: str) -> str:
    if not isinstance(t, str):
        t = str(t)
    t = t.replace('Ġ', ' ')
    if t.startswith('##'):
        t = t[2:]
    t = t.replace('▁', ' ')
    t = t.replace('[CLS]', 'CLS').replace('[SEP]', 'SEP').replace('[PAD]', 'PAD')
    return t.strip()


def _decode_token(tokenizer, tid: int) -> str:
    try:
        return tokenizer.decode([int(tid)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except Exception:
        return str(tid)


def _short(s: str, n: int = 18) -> str:
    t = (s or '').replace('\n', ' ').strip()
    if len(t) <= n:
        return t
    return t[: max(0, n - 3)] + '...'


def _wrap_bullets(items: Sequence[str], width: int) -> str:
    out: List[str] = []
    for it in items:
        lines = textwrap.fill(str(it), width=width).splitlines()
        if not lines:
            continue
        out.append('- ' + lines[0])
        for ln in lines[1:]:
            out.append('  ' + ln)
    return '\n'.join(out)


def _select_token_window(tokens: Sequence[str], *, max_tokens: int) -> Tuple[int, int]:
    n = int(len(tokens))
    if n <= max_tokens:
        return 0, n

    toks_lower: List[str] = []
    for t in tokens:
        try:
            toks_lower.append(str(t).lower())
        except Exception:
            toks_lower.append('')

    anchors: List[int] = []
    for i, t in enumerate(toks_lower):
        if any(x in t for x in ('payload', '```', '<', '>', 'begin', 'end', 'system', 'developer')):
            anchors.append(i)

    if anchors:
        c0 = max(0, min(anchors))
        start = max(0, c0 - max_tokens // 4)
        end = min(n, start + max_tokens)
        return start, end

    return 0, max_tokens


def _attn_to_matrix(att: torch.Tensor) -> Optional[np.ndarray]:
    """Convert an attention-like tensor to (S,S) numpy matrix (avg over heads/batch)."""
    if att is None:
        return None
    t = att.detach()
    if t.ndim == 4:
        # (B,H,S,S)
        return t[0].mean(dim=0).cpu().numpy()
    if t.ndim == 3:
        # (H,S,S) or (B,S,S)
        return t.mean(dim=0).cpu().numpy()
    if t.ndim == 2:
        # (S,S)
        return t.cpu().numpy()
    return None


def _aggregate_attention_matrix(
    *,
    attentions: List[torch.Tensor],
    hidden_states: Optional[List[torch.Tensor]],
    layer: str,
    layer_start: Optional[int],
    layer_end: Optional[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {}

    layer_q = str(layer).strip().lower()

    if attentions:
        L = len(attentions)
        if layer_q == 'all':
            s = 0 if layer_start is None else int(layer_start)
            e = (L - 1) if layer_end is None else int(layer_end)
            s = max(0, min(s, L - 1))
            e = max(0, min(e, L - 1))
            if e < s:
                s, e = e, s
            idxs = list(range(s, e + 1))
            meta['layer_mode'] = 'all'
            meta['layer_indices'] = idxs
        else:
            idx = int(layer_q)
            idx = max(0, min(idx, L - 1))
            idxs = [idx]
            meta['layer_mode'] = 'single'
            meta['layer_indices'] = idxs

        mats = []
        for i in idxs:
            m = _attn_to_matrix(attentions[i])
            if m is not None:
                mats.append(m)

        if not mats:
            raise RuntimeError('No usable attention matrices extracted')

        A = np.mean(np.stack(mats, axis=0), axis=0)
        meta['matrix_type'] = 'attention'
        meta['num_layers_available'] = int(L)
        meta['num_layers_used'] = int(len(mats))
        return A, meta

    # Proxy fallback: cosine similarity over hidden states
    if not hidden_states:
        raise RuntimeError('Model returned no attentions and no hidden states')

    hs = hidden_states
    H = len(hs)
    if layer_q == 'all':
        vecs = hs[-1][0].detach().cpu().numpy()
        meta['layer_mode'] = 'proxy_last'
        meta['layer_indices'] = [H - 1]
    else:
        idx = int(layer_q)
        idx = max(0, min(idx, H - 1))
        vecs = hs[idx][0].detach().cpu().numpy()
        meta['layer_mode'] = 'proxy_single'
        meta['layer_indices'] = [idx]

    v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    A = v @ v.T
    meta['matrix_type'] = 'proxy_cosine_similarity'
    meta['num_layers_available'] = int(H)
    meta['num_layers_used'] = 1
    return A, meta


def _token_text_features(raw: str) -> List[float]:
    s = (raw or '').strip()
    s2 = re.sub(r'\s+', ' ', s)

    is_special = 1.0 if (s2.startswith('[') and s2.endswith(']')) else 0.0
    has_digit = 1.0 if any(c.isdigit() for c in s2) else 0.0
    has_upper = 1.0 if any(c.isupper() for c in s2) else 0.0
    has_underscore = 1.0 if '_' in s2 else 0.0
    has_angle = 1.0 if ('<' in s2 or '>' in s2) else 0.0
    has_equals = 1.0 if '=' in s2 else 0.0

    alnum = sum(1 for c in s2 if c.isalnum())
    punct = len(s2) - alnum
    is_punct = 1.0 if (len(s2) > 0 and punct > 0 and alnum <= max(1, punct // 2)) else 0.0

    length = float(min(len(s2), 24)) / 24.0

    low = s2.lower()
    has_susp = 1.0 if any(x in low for x in _SUSPICIOUS_SUBSTRINGS) else 0.0

    return [is_special, has_digit, has_upper, has_underscore, has_angle, has_equals, is_punct, length, has_susp]


def _attention_feature_matrix(A: np.ndarray, raw_tokens: Sequence[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    S = int(A.shape[0])

    A0 = np.array(A, dtype=np.float64)
    if not np.isfinite(A0).all():
        A0 = np.nan_to_num(A0, nan=0.0, posinf=0.0, neginf=0.0)
    if A0.min() < 0:
        A0 = A0 - float(A0.min())

    row = A0 / (A0.sum(axis=1, keepdims=True) + 1e-12)

    ent = -np.sum(row * np.log(row + 1e-12), axis=1)
    ent = ent / (math.log(float(S) + 1e-12) if S > 1 else 1.0)

    row_max = row.max(axis=1)
    diag = np.diag(row)
    col_sum = row.sum(axis=0) / max(1.0, float(S))
    col_max = row.max(axis=0)

    txt = np.array([_token_text_features(t) for t in raw_tokens], dtype=np.float64)

    X = np.column_stack([ent, row_max, diag, col_sum, col_max, txt])

    per = {
        'row_entropy': ent,
        'row_max': row_max,
        'diag': diag,
        'col_sum': col_sum,
        'col_max': col_max,
    }
    return X, per


def _isolation_forest_scores(
    X: np.ndarray,
    *,
    contamination: str,
    n_estimators: int,
    seed: int,
) -> np.ndarray:
    n = int(X.shape[0])
    if n < 6:
        v = X[:, 1]
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)) + 1e-9)
        z = np.abs((v - med) / mad)
        return (z - z.min()) / (z.max() - z.min() + 1e-9)

    # standardize
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    from sklearn.ensemble import IsolationForest

    if str(contamination).strip().lower() == 'auto':
        cont: Any = 'auto'
    else:
        cont = float(contamination)
        cont = max(0.001, min(cont, 0.49))

    iso = IsolationForest(
        n_estimators=int(n_estimators),
        contamination=cont,
        random_state=int(seed),
    )
    iso.fit(Xz)
    raw = -iso.score_samples(Xz)  # higher => more anomalous
    raw = np.asarray(raw, dtype=np.float64)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)


def _score_to_colors(scores: np.ndarray) -> Tuple[List[str], float, float]:
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return [], 0.0, 0.0

    red_th = float(np.quantile(s, 0.85))
    org_th = float(np.quantile(s, 0.70))

    colors: List[str] = []
    for v in s:
        if float(v) >= red_th:
            colors.append('#d62728')
        elif float(v) >= org_th:
            colors.append('#ff9f1a')
        else:
            colors.append('#2ca02c')
    return colors, org_th, red_th


def _top_attention_pairs(A: np.ndarray, tokens_clean: Sequence[str], *, k: int = 10) -> List[Dict[str, Any]]:
    A0 = np.array(A, dtype=np.float64)
    if A0.min() < 0:
        A0 = A0 - float(A0.min())
    row = A0 / (A0.sum(axis=1, keepdims=True) + 1e-12)

    S = int(row.shape[0])
    mask = ~np.eye(S, dtype=bool)
    vals = row[mask]
    if vals.size == 0:
        return []

    coords = np.argwhere(mask)
    top_idx = np.argsort(vals)[-k:][::-1]

    out: List[Dict[str, Any]] = []
    for ii in top_idx:
        i, j = coords[int(ii)]
        out.append({
            'query_index': int(i),
            'key_index': int(j),
            'query_token': str(tokens_clean[i]),
            'key_token': str(tokens_clean[j]),
            'weight': float(row[i, j]),
        })
    return out


def _summarize_findings(
    *,
    tokens_clean: Sequence[str],
    tokens_raw: Sequence[str],
    scores: np.ndarray,
    red_th: float,
    col_sum: np.ndarray,
) -> Dict[str, Any]:
    toks_raw = [str(t) for t in tokens_raw]

    susp = []
    for i, (raw, sc) in enumerate(zip(toks_raw, scores)):
        r = (raw or '').strip()
        low = r.lower()
        cue = any(x in low for x in _SUSPICIOUS_SUBSTRINGS) or any(ch in r for ch in ('`', '<', '>', '{', '}', '='))
        if float(sc) >= float(red_th) or cue:
            susp.append((int(i), str(tokens_clean[i]), r, float(sc)))
    susp = sorted(susp, key=lambda x: x[3], reverse=True)[:10]

    infl = []
    for i, v in enumerate(col_sum):
        infl.append((int(i), str(tokens_clean[i]), str(tokens_raw[i]), float(v)))
    infl = sorted(infl, key=lambda x: x[3], reverse=True)[:8]

    findings: List[str] = []
    if susp:
        findings.append('[!] Potential control/delimiter/payload tokens detected (review top anomalies)')
    else:
        findings.append('[+] No obvious control/delimiter cues by token text; review anomaly bars for outliers')
    if infl:
        findings.append('[!] High-attention attractor tokens present (tokens other tokens attend to)')

    blue = [
        'Compare against a benign baseline prompt: prioritize tokens that become newly red/orange.',
        'Investigate top attractor tokens (high inbound attention): these often anchor injected instructions/delimiters.',
        'Mitigate by input normalization (escape/strip delimiters), policy gating, and strict tool-call allowlists.',
        'If anomalies persist across many layers, treat as higher-confidence prompt-hijack indicator.',
    ]
    red = [
        'Authorized testing only: iterate prompts to see which tokens become persistent attractors/anomalies across layers.',
        'Vary delimiters/instruction framing to probe whether defenses collapse attractor columns back to task tokens.',
        'A robust defense should prevent long-lived red attractors that track attacker-controlled tokens.',
    ]

    return {
        'security_findings': findings,
        'suspicious_tokens': susp,
        'top_influencers': infl,
        'blue_team_next_steps': blue,
        'red_team_next_steps': red,
    }


def generate_attention_security_analysis(cfg: AttentionSecurityConfig) -> Tuple[str, str, str]:
    device = _select_device(cfg.device)
    tokenizer, encoded, tokens, token_ids, resolved_name = _tokenize(cfg.model_name, cfg.prompt)

    # Execute real forward pass and extract attentions
    attentions, hidden_states = _run_with_attentions(resolved_name, encoded, device)

    A_full, meta = _aggregate_attention_matrix(
        attentions=attentions,
        hidden_states=hidden_states,
        layer=str(cfg.layer),
        layer_start=cfg.layer_start,
        layer_end=cfg.layer_end,
    )

    # Window for readability
    start, end = _select_token_window(tokens, max_tokens=int(cfg.max_tokens))
    tokens_view = list(tokens[start:end])
    ids_view = list(token_ids[start:end])
    tokens_clean = [_clean_tok_display(t) for t in tokens_view]
    tokens_raw = [_decode_token(tokenizer, tid) for tid in ids_view]

    A = A_full[start:end, start:end]

    X, per = _attention_feature_matrix(A, tokens_raw)
    scores = _isolation_forest_scores(
        X,
        contamination=str(cfg.contamination),
        n_estimators=int(cfg.n_estimators),
        seed=int(cfg.seed),
    )
    colors, orange_th, red_th = _score_to_colors(scores)

    findings = _summarize_findings(
        tokens_clean=tokens_clean,
        tokens_raw=tokens_raw,
        scores=scores,
        red_th=red_th,
        col_sum=per['col_sum'],
    )

    top_pairs = _top_attention_pairs(A, tokens_clean, k=10)

    # Salient indices for plot callouts (more useful than generic '!')
    top_anom_idx = [int(i) for i in np.argsort(scores)[-3:][::-1]] if len(scores) else []
    top_attr_idx = [int(i) for i in np.argsort(per['col_sum'])[-3:][::-1]] if len(scores) else []

    # --- Plot ---
    fig = plt.figure(figsize=(16, 9.4))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[3.2, 1.05, 1.15],
        width_ratios=[1.0, 1.0],
        hspace=0.40,
        wspace=0.25,
    )

    ax_hm = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_find = fig.add_subplot(gs[1, :])
    ax_blue = fig.add_subplot(gs[2, 0])
    ax_red = fig.add_subplot(gs[2, 1])
    for _ax in (ax_find, ax_blue, ax_red):
        _ax.axis('off')

    # Prevent title/header overlap with plots
    fig.subplots_adjust(top=0.82, bottom=0.07)

    # Heatmap contrast stretch
    q1, q99 = np.quantile(A, [0.01, 0.99])
    if float(q99) <= float(q1):
        q1, q99 = float(np.min(A)), float(np.max(A) if np.max(A) != 0 else 1.0)

    im = ax_hm.imshow(A, cmap='viridis', aspect='auto', vmin=float(q1), vmax=float(q99))
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight' if meta.get('matrix_type') == 'attention' else 'Similarity')

    max_ticks = 26
    tick_idx = np.linspace(0, len(tokens_clean) - 1, num=min(len(tokens_clean), max_ticks), dtype=int)
    tick_labels = [tokens_clean[i] for i in tick_idx]

    ax_hm.set_xticks(tick_idx)
    ax_hm.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax_hm.set_yticks(tick_idx)
    ax_hm.set_yticklabels(tick_labels)
    ax_hm.set_title('Attention Pattern Heatmap')

    # Highlight top attractor columns (inbound attention)
    for k, idx in enumerate(top_attr_idx, start=1):
        ax_hm.axvline(idx, color='#ff7f0e', linestyle='--', linewidth=1.0, alpha=0.75)
        ax_hm.text(idx + 0.1, -0.6, f"A{k}", fontsize=8, color='#ff7f0e')

    # Bars
    xs = np.arange(len(tokens_clean))
    ax_bar.bar(xs, scores, color=colors, edgecolor='white', linewidth=0.6)
    ax_bar.set_ylim(0.0, max(1.05, float(np.max(scores) + 0.05)))
    ax_bar.set_title('Token-Level Anomaly Scores')
    ax_bar.set_ylabel('Anomaly Score')
    ax_bar.set_xticks(tick_idx)
    ax_bar.set_xticklabels([tokens_clean[i] for i in tick_idx], rotation=45, ha='right')

    ax_bar.axhline(red_th, color='#d62728', linestyle='--', linewidth=1.0)
    ax_bar.axhline(orange_th, color='#ff9f1a', linestyle=':', linewidth=1.0)

    # Mark top anomalies with ranked stars (more useful than generic '!')
    for rank, idx in enumerate(top_anom_idx, start=1):
        ax_bar.scatter(idx, float(scores[idx]), marker='*', s=170, color='#111111', edgecolor='white', linewidth=0.7, zorder=5)
        ax_bar.text(
            idx,
            float(scores[idx]) + 0.05,
            str(rank),
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold',
            color='#111111',
            bbox=dict(boxstyle='circle,pad=0.18', facecolor='white', edgecolor='#111111', alpha=0.95),
            zorder=6,
        )

    # Mark top attractors on x-axis
    for idx in top_attr_idx:
        ax_bar.scatter(idx, 0.0, marker='v', s=60, color='#666666', zorder=4)

    # Compact legend
    try:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color='#d62728', lw=1.2, ls='--', label=f"red≥{red_th:.2f}"),
            Line2D([0], [0], color='#ff9f1a', lw=1.2, ls=':', label=f"orange≥{orange_th:.2f}"),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#111111', markeredgecolor='white', markersize=10, label='top anomalies'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='#666666', markersize=7, label='top attractors'),
        ]
        ax_bar.legend(handles=handles, loc='upper right', fontsize=8, frameon=False)
    except Exception:
        pass

    # Header info
    layer_label = 'All Layers' if str(cfg.layer).strip().lower() == 'all' else f"Layer {cfg.layer}"
    prompt_one_line = re.sub(r'\s+', ' ', str(cfg.prompt)).strip()
    if len(prompt_one_line) > 60:
        prompt_one_line = prompt_one_line[:57] + '...'

    fig.text(
        0.5,
        0.925,
        f"Model: {resolved_name} | Layer: {layer_label} | Prompt: {prompt_one_line}",
        ha='center',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.95),
    )

    # Bottom blocks
    findings_lines = ['=== SECURITY FINDINGS ==='] + list(findings['security_findings'])
    if top_anom_idx:
        findings_lines.append('Top anomalies (★ on bar chart):')
        for r, idx in enumerate(top_anom_idx, start=1):
            findings_lines.append(f"  #{r}: idx={idx} tok='{_short(tokens_clean[idx], 12)}' score={float(scores[idx]):.2f}")
    if top_attr_idx:
        findings_lines.append('Top attractors (A# on heatmap / ▼ on bars):')
        for r, idx in enumerate(top_attr_idx, start=1):
            findings_lines.append(f"  A{r}: idx={idx} tok='{_short(tokens_clean[idx], 12)}' inbound={float(per['col_sum'][idx]):.3f}")
    if top_pairs:
        findings_lines.append('Top attention pairs (off-diagonal):')
        for r, pair in enumerate(top_pairs[:3], start=1):
            findings_lines.append(
                f"  P{r}: Q[{pair['query_index']}] '{_short(pair['query_token'], 10)}' → K[{pair['key_index']}] '{_short(pair['key_token'], 10)}' w={float(pair['weight']):.3f}"
            )

    findings_block = '\n'.join(findings_lines)

    blue_block = 'BLUE TEAM GUIDANCE ===\n' + _wrap_bullets(findings['blue_team_next_steps'], width=64)
    red_block = 'RED TEAM GUIDANCE ===\n' + _wrap_bullets(findings['red_team_next_steps'], width=64)

    ax_find.text(0.01, 0.98, findings_block, ha='left', va='top', fontsize=9, family='monospace', color='#222222')
    ax_blue.text(
        0.01,
        0.98,
        blue_block,
        ha='left',
        va='top',
        fontsize=9,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#e6f0ff', edgecolor='#1f5fbf', alpha=0.95),
    )
    ax_red.text(
        0.01,
        0.98,
        red_block,
        ha='left',
        va='top',
        fontsize=9,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#cc0000', alpha=0.95),
    )

    fig.suptitle(cfg.title, y=0.985, fontsize=14, fontweight='bold')

    out_png = Path(cfg.output_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)

    # JSON artifact
    out_json = Path(cfg.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        'title': str(cfg.title),
        'model': str(resolved_name),
        'tokenizer': str(getattr(tokenizer, 'name_or_path', resolved_name)),
        'prompt': str(cfg.prompt),
        'prompt_sha16': _prompt_sha16(cfg.prompt),
        'device': str(device),
        'layer': str(cfg.layer),
        'layer_start': cfg.layer_start,
        'layer_end': cfg.layer_end,
        'matrix_type': str(meta.get('matrix_type')),
        'num_layers_available': int(meta.get('num_layers_available', 0) or 0),
        'num_layers_used': int(meta.get('num_layers_used', 0) or 0),
        'window': {'start': int(start), 'end': int(end)},
        'tokens_clean': tokens_clean,
        'tokens_raw': tokens_raw,
        'token_ids': [int(x) for x in ids_view],
        'anomaly': {
            'method': 'IsolationForest',
            'contamination': str(cfg.contamination),
            'n_estimators': int(cfg.n_estimators),
            'seed': int(cfg.seed),
            'scores': [float(x) for x in scores],
            'thresholds': {'orange': float(orange_th), 'red': float(red_th)},
        },
        'attention_features': {
            'row_entropy': [float(x) for x in per['row_entropy']],
            'row_max': [float(x) for x in per['row_max']],
            'diag': [float(x) for x in per['diag']],
            'col_sum': [float(x) for x in per['col_sum']],
            'col_max': [float(x) for x in per['col_max']],
        },
        'top_attention_pairs': top_pairs,
        'security': findings,
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    # Interactive HTML (Plotly). Falls back to embedded PNG if Plotly fails.
    out_html = Path(getattr(cfg, 'out_html', '_cli_runs/attention_security.html'))
    out_html.parent.mkdir(parents=True, exist_ok=True)

    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import plotly.io as pio

        S = int(len(tokens_clean))
        xvals = list(range(start, end))

        # Subsample tick labels for readability
        max_ticks = 26
        tick_idx = np.linspace(0, S - 1, num=min(S, max_ticks), dtype=int)
        tick_vals = [start + int(i) for i in tick_idx]
        tick_txt = [tokens_clean[int(i)] for i in tick_idx]

        # Heatmap hover text
        hover = [[f"Q[{i}] {tokens_raw[i]}<br>K[{j}] {tokens_raw[j]}" for j in range(S)] for i in range(S)]

        figp = make_subplots(rows=1, cols=2, column_widths=[0.52, 0.48], horizontal_spacing=0.08)

        q1, q99 = np.quantile(A, [0.01, 0.99])
        if float(q99) <= float(q1):
            q1, q99 = float(np.min(A)), float(np.max(A) if np.max(A) != 0 else 1.0)

        figp.add_trace(
            go.Heatmap(
                z=A,
                x=xvals,
                y=xvals,
                colorscale='Viridis',
                zmin=float(q1),
                zmax=float(q99),
                colorbar=dict(title='Attention' if meta.get('matrix_type') == 'attention' else 'Similarity'),
                text=hover,
                hovertemplate='%{text}<br>value=%{z:.4f}<extra></extra>',
            ),
            row=1,
            col=1,
        )

        figp.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_txt, tickangle=45, row=1, col=1)
        figp.update_yaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_txt, row=1, col=1)

        # Bar chart with rich hover
        cdata = np.column_stack([
            np.array(tokens_clean, dtype=object),
            np.array(tokens_raw, dtype=object),
            np.array(scores, dtype=float),
            np.array(per.get('row_entropy'), dtype=float),
            np.array(per.get('row_max'), dtype=float),
            np.array(per.get('col_sum'), dtype=float),
        ])

        figp.add_trace(
            go.Bar(
                x=xvals,
                y=[float(x) for x in scores],
                marker_color=list(colors),
                customdata=cdata,
                hovertemplate=(
                    "idx=%{x}<br>clean=%{customdata[0]}<br>raw=%{customdata[1]}<br>" +
                    "score=%{customdata[2]:.3f}<br>row_entropy=%{customdata[3]:.3f}<br>" +
                    "row_max=%{customdata[4]:.3f}<br>inbound(col_sum)=%{customdata[5]:.3f}<extra></extra>"
                ),
                name='Anomaly score',
            ),
            row=1,
            col=2,
        )

        # Threshold lines
        figp.add_hline(y=float(red_th), line_dash='dash', line_color='#d62728', row=1, col=2)
        figp.add_hline(y=float(orange_th), line_dash='dot', line_color='#ff9f1a', row=1, col=2)

        # Top anomalies markers (ranked)
        if top_anom_idx:
            xs = [start + int(i) for i in top_anom_idx]
            ys = [float(scores[int(i)]) for i in top_anom_idx]
            figp.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode='markers+text',
                    text=[str(r + 1) for r in range(len(xs))],
                    textposition='top center',
                    marker=dict(symbol='star', size=14, color='#111111', line=dict(color='white', width=1)),
                    hovertemplate='rank=%{text} idx=%{x}<extra></extra>',
                    name='Top anomalies',
                ),
                row=1,
                col=2,
            )

        # Top attractors markers and heatmap vlines
        if top_attr_idx:
            xs = [start + int(i) for i in top_attr_idx]
            figp.add_trace(
                go.Scatter(
                    x=xs,
                    y=[0.0 for _ in xs],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=11, color='#666666'),
                    name='Top attractors',
                    hovertemplate='attractor idx=%{x}<extra></extra>',
                ),
                row=1,
                col=2,
            )
            for i in top_attr_idx:
                figp.add_vline(x=start + int(i), line_dash='dash', line_color='#ff7f0e', line_width=1.0, row=1, col=1)

        # Axis titles
        figp.update_xaxes(title_text='Key token index', row=1, col=1)
        figp.update_yaxes(title_text='Query token index', row=1, col=1)
        figp.update_xaxes(title_text='Token index', row=1, col=2)
        figp.update_yaxes(title_text='Anomaly score', row=1, col=2)

        figp.update_layout(
            template='plotly_white',
            height=640,
            margin=dict(l=70, r=40, t=20, b=120),
            showlegend=False,
        )

        plot_div = pio.to_html(figp, include_plotlyjs='cdn', full_html=False)

        def _esc_html(s: str) -> str:
            return (s or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # External layout: legend first, then security findings (per request), then plot, then guidance.
        legend_html = f'''
<div class='box legend'>
  <div class='leg-item'><span class='swatch red'></span> red&gt;={red_th:.2f}</div>
  <div class='leg-item'><span class='swatch orange'></span> orange&gt;={orange_th:.2f}</div>
  <div class='leg-item'><span class='sym'>#1..#3</span> top anomalies</div>
  <div class='leg-item'><span class='sym'>v</span> top attractors</div>
  <div class='leg-item'><span class='sym'>A1..A3</span> attractor columns (heatmap)</div>
  <div class='leg-item'><span class='sym'>Hover</span> for raw tokens + features</div>
</div>
'''.strip()

        findings_pre = _esc_html(findings_block)
        blue_pre = _esc_html(blue_block)
        red_pre = _esc_html(red_block)

        html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{_esc_html(cfg.title)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 16px; color: #111; }}
  .subtitle {{ margin: 6px 0 14px 0; color: #333; }}
  .box {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px; background: #fff; }}
  .legend {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
  .leg-item {{ display: inline-flex; gap: 8px; align-items: center; font-size: 13px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .swatch.red {{ background: #d62728; }}
  .swatch.orange {{ background: #ff9f1a; }}
  .sym {{ font-weight: 700; }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; white-space: pre-wrap; margin: 0; }}
  .stack {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
  .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  @media (max-width: 980px) {{ .two {{ grid-template-columns: 1fr; }} }}
  .blue {{ background: #e6f0ff; border-color: #1f5fbf; }}
  .red {{ background: #ffe6e6; border-color: #cc0000; }}
</style>
</head><body>
<h2 style="margin:0;">{_esc_html(cfg.title)}</h2>
<div class="subtitle"><b>Model</b>: {_esc_html(resolved_name)} &nbsp; | &nbsp; <b>Layer</b>: {_esc_html(layer_label)} &nbsp; | &nbsp; <b>Prompt</b>: {_esc_html(prompt_one_line)}</div>
<div class="stack">
{legend_html}
<div class="box">{plot_div}</div>
<div class="two">
  <div class="box blue"><b>Blue team next steps</b><pre class="mono">{blue_pre}</pre></div>
  <div class="box red"><b>Red team next steps</b><pre class="mono">{red_pre}</pre></div>
</div>
<div class="box"><b>Security findings</b><pre class="mono">{findings_pre}</pre></div>
</div>
</body></html>
'''

        out_html.write_text(html, encoding='utf-8')
    except Exception:
        # Fallback: embed PNG
        try:
            b64 = base64.b64encode(out_png.read_bytes()).decode('ascii')
            out_html.write_text(
                '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NeurInSpectre — Attention Security Analysis</title></head>
<body>
<h3>NeurInSpectre — Attention Security Analysis (static fallback)</h3>
<p>Plotly not available; showing PNG fallback.</p>
<img style="max-width:100%" src="data:image/png;base64,REPLACE_B64" />
</body></html>
'''.replace('REPLACE_B64', b64),
                encoding='utf-8',
            )
        except Exception:
            pass

    return str(out_png), str(out_json), str(out_html)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description='Attention heatmap + token anomaly scores (IsolationForest)')
    p.add_argument('--model', required=True, help='HuggingFace model name (e.g., gpt2)')

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--prompt', help='Prompt text')
    src.add_argument('--prompt-file', default=None, help='File containing a prompt (first non-empty line)')

    p.add_argument('--layer', default='all', help="Layer index (int) or 'all' to average across layers")
    p.add_argument('--layer-start', type=int, default=None, help='Start layer (inclusive) when using --layer all')
    p.add_argument('--layer-end', type=int, default=None, help='End layer (inclusive) when using --layer all')

    p.add_argument('--max-tokens', type=int, default=128, help='Max tokens to visualize (default: 128)')
    p.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto')

    p.add_argument('--output-png', default='_cli_runs/attention_security.png', help='Output PNG path')
    p.add_argument('--out-json', default='_cli_runs/attention_security.json', help='Output JSON path')
    p.add_argument('--out-html', default='_cli_runs/attention_security.html', help='Output interactive HTML path')

    p.add_argument('--contamination', default='auto', help="IsolationForest contamination ('auto' or float)")
    p.add_argument('--n-estimators', type=int, default=256, help='IsolationForest n_estimators (default: 256)')
    p.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')

    p.add_argument('--title', default='NeurInSpectre — Attention Security Analysis', help='Plot title')

    args = p.parse_args(argv)

    def _read_first_nonempty(fp: str) -> str:
        for ln in Path(fp).read_text(encoding='utf-8', errors='ignore').splitlines():
            s = ln.strip()
            if s:
                return s
        raise ValueError('prompt-file is empty')

    prompt = args.prompt
    if not prompt and getattr(args, 'prompt_file', None):
        prompt = _read_first_nonempty(str(args.prompt_file))

    cfg = AttentionSecurityConfig(
        model_name=str(args.model),
        prompt=str(prompt),
        layer=str(args.layer),
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        max_tokens=int(args.max_tokens),
        device=str(args.device),
        output_png=str(args.output_png),
        out_json=str(args.out_json),
        out_html=str(getattr(args, 'out_html', '_cli_runs/attention_security.html')) ,
        contamination=str(args.contamination),
        n_estimators=int(args.n_estimators),
        seed=int(args.seed),
        title=str(args.title),
    )

    try:
        out_png, out_json, out_html = generate_attention_security_analysis(cfg)
        print(out_png)
        print(out_json)
        print(out_html)
        return 0
    except Exception as e:
        print(f'❌ attention-security failed: {e}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
