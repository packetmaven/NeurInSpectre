#!/usr/bin/env python3
"""
Attention Heatmap CLI
Generates a token×token attention heatmap for a given prompt and model, with
clear axis labels tailored for security interpretability (red/blue teams).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


@dataclass
class AttentionConfig:
    model_name: str
    prompt: str
    layer: int = 0
    head: int = 0  # set to -1 for auto head selection (max contrast)
    device: str = "auto"  # auto | mps | cuda | cpu
    output_png: str = "attention_heatmap.png"
    out_prefix: str = "attn_"
    baseline: str | None = None


def _select_device(preferred: str = "auto") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "cpu":
        return torch.device("cpu")
    # auto
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _tokenize(model_name: str, text: str):
    from transformers import AutoTokenizer

    # Resolve common aliases to canonical HF IDs
    alias_map = {
        'google/bert-base-uncased': 'bert-base-uncased',
        'openai-community/gpt2': 'gpt2',
    }
    resolved = alias_map.get(model_name, model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved)
    except Exception:
        # Fallback: try canonical BERT if alias was problematic
        if resolved != 'bert-base-uncased':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            resolved = 'bert-base-uncased'
        else:
            raise
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded["input_ids"].flatten().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return tokenizer, encoded, tokens, ids, resolved


def _run_with_attentions(model_name: str, encoded_inputs, device: torch.device):
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

    # Try causal LM first, then encoder-only, then seq2seq
    model = None
    # Prefer base model classes for reliable attentions/hidden_states
    # Resolve aliases similar to tokenizer
    alias_map = {
        'google/bert-base-uncased': 'bert-base-uncased',
        'openai-community/gpt2': 'gpt2',
    }
    resolved_name = alias_map.get(model_name, model_name)

    for cls in (AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM):
        try:
            # Safe to request attentions on non-generation model load
            model = cls.from_pretrained(resolved_name)
            break
        except Exception:
            continue
    if model is None:
        # Last resort: canonical BERT
        try:
            model = AutoModel.from_pretrained('bert-base-uncased')
            resolved_name = 'bert-base-uncased'
        except Exception:
            raise RuntimeError(f"Could not load model '{model_name}' (resolved='{resolved_name}') with attention outputs enabled")

    model.to(device)
    model.eval()

    # Ensure attentions are produced without touching generation config
    if hasattr(model, 'config'):
        try:
            setattr(model.config, 'output_attentions', True)
            setattr(model.config, 'output_hidden_states', True)
        except Exception:
            pass
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)

    # Collect any available attentions (encoder/decoder/cross), skipping None entries
    att_groups = []
    for attr in ["attentions", "decoder_attentions", "encoder_attentions", "cross_attentions"]:
        if hasattr(outputs, attr):
            att = getattr(outputs, attr)
            if att is not None:
                att_groups.extend([a for a in att if a is not None])

    # Hidden states for proxy if needed
    hidden_states = None
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        hidden_states = outputs.hidden_states
    elif hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
        hidden_states = outputs.encoder_hidden_states
    elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
        hidden_states = [outputs.last_hidden_state]

    # If still nothing, try architecture-specific forward hooks to capture attn probs
    if not att_groups:
        captured = []
        hooks = []
        def _mk_hook(store):
            def _hook(_mod, _inp, out):
                try:
                    # Common patterns: (attn_output, present, attn_weights) or tuple with last item as attn
                    if isinstance(out, (tuple, list)) and len(out) >= 1:
                        cand = out[-1]
                        if torch.is_tensor(cand):
                            store.append(cand)
                except Exception:
                    pass
            return _hook
        try:
            # GPT2-like
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layers = model.transformer.h
                idx = min(len(layers)-1, 0)
                hooks.append(layers[idx].attn.register_forward_hook(_mk_hook(captured)))
            # BERT-like
            elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                layers = model.encoder.layer
                idx = min(len(layers)-1, 0)
                if hasattr(layers[idx], 'attention') and hasattr(layers[idx].attention, 'self'):
                    hooks.append(layers[idx].attention.self.register_forward_hook(_mk_hook(captured)))
            # OPT / decoder-only
            elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
                layers = model.model.decoder.layers
                idx = min(len(layers)-1, 0)
                if hasattr(layers[idx], 'self_attn'):
                    hooks.append(layers[idx].self_attn.register_forward_hook(_mk_hook(captured)))
            # Re-run with hooks
            with torch.no_grad():
                _ = model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
        except Exception:
            pass
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
        if captured:
            att_groups = [captured[0]]

    return att_groups, hidden_states


def generate_attention_heatmap(cfg: AttentionConfig) -> str:
    device = _select_device(cfg.device)
    tokenizer, encoded, tokens, token_ids, resolved_name = _tokenize(cfg.model_name, cfg.prompt)
    attentions, hidden_states = _run_with_attentions(resolved_name, encoded, device)

    use_proxy = False
    if attentions:
        num_layers = len(attentions)
        layer_idx = max(0, min(cfg.layer, num_layers - 1))
        attn_l = attentions[layer_idx]
        attn_l = attn_l.detach().cpu().numpy()  # (B,H,S,S)
        # Auto-head: choose head with highest off-diagonal variance (strongest structure)
        if cfg.head < 0:
            A = attn_l[0]  # (H,S,S)
            # mask diagonal to emphasize cross-token focus
            import numpy as _np
            S = A.shape[-1]
            diag_mask = _np.eye(S, dtype=bool)
            variances = []
            for h in range(A.shape[0]):
                Ah = A[h]
                off = Ah[~diag_mask]
                variances.append(off.var())
            head_idx = int(_np.argmax(variances))
        else:
            head_idx = cfg.head
        heat = attn_l[0, head_idx, :, :]
        title_prefix = "Attention"
    else:
        # Proxy fallback: cosine similarity over hidden states at selected layer
        if not hidden_states:
            raise RuntimeError("Model did not return attentions or hidden states; try a different model")
        num_layers = len(hidden_states)
        layer_idx = max(0, min(cfg.layer, num_layers - 1))
        hs = hidden_states[layer_idx].detach().cpu().numpy()  # (B,S,D)
        vecs = hs[0]
        # Normalize and compute cosine sim matrix
        import numpy as _np
        v = vecs / (_np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        heat = v @ v.T
        title_prefix = "Proxy (cosine‑sim)"

    # Plot
    # Adaptive token windowing for legibility, prefer payload span if present
    import numpy as _np
    seq_len = heat.shape[0]
    max_window = 128 if seq_len > 128 else seq_len
    # Heuristic: focus on tokens around '<PAYLOAD' or 'PAYLOAD' markers
    idxs = list(range(seq_len))
    try:
        toks_lower = [t.lower() for t in tokens]
        candidates = [i for i,t in enumerate(toks_lower) if ('payload' in t or '<' in t or '>' in t)]
    except Exception:
        candidates = []
    if candidates:
        c0 = max(0, min(candidates))
        start = max(0, c0 - max_window//4)
        end = min(seq_len, start + max_window)
    else:
        start = 0
        end = max_window
    heat_view = heat[start:end, start:end]
    tokens_view = tokens[start:end]
    ids_view = token_ids[start:end]

    # Contrast stretching (quantile) to surface structure
    q1, q99 = _np.quantile(heat_view, [0.01, 0.99])
    if q99 <= q1:
        q1, q99 = heat_view.min(), heat_view.max() or 1.0

    plt.figure(figsize=(min(18, 6 + 0.06*len(tokens_view)), 6))
    im = plt.imshow(heat_view, cmap="viridis", aspect="auto", vmin=float(q1), vmax=float(q99))
    cbar = plt.colorbar(im)
    cbar.set_label("Attention Strength")

    # Token labels (subsample if too long)
    max_ticks = 40
    tick_idx = np.linspace(0, len(tokens_view) - 1, num=min(len(tokens_view), max_ticks), dtype=int)
    def _clean_tok(t: str) -> str:
        # Display-friendly token normalization without changing the underlying tensor indexing
        if not isinstance(t, str):
            t = str(t)
        t = t.replace("Ġ", " ")  # byte-level BPE space marker
        if t.startswith("##"):
            t = t[2:]  # WordPiece continuation
        t = t.replace("▁", " ")  # sentencepiece space
        # Compact special tokens
        t = t.replace("[CLS]", "CLS").replace("[SEP]", "SEP").replace("[PAD]", "PAD")
        return t.strip()
    tick_labels = [_clean_tok(tokens_view[i]) for i in tick_idx]
    plt.xticks(ticks=tick_idx, labels=tick_labels, rotation=45, ha="right")
    plt.yticks(ticks=tick_idx, labels=tick_labels)

    head_disp = head_idx if 'head_idx' in locals() else cfg.head
    plt.title(f"NeurInSpectre — {title_prefix} Patterns\nLayer {layer_idx}, Head {head_disp} | Darker = Stronger")
    plt.xlabel("Key Tokens (What the model is attending to)")
    plt.ylabel("Query Tokens (What is doing the attending)")
    plt.tight_layout()
    # Ensure output directories exist
    from pathlib import Path as _Path
    _Path(_Path(cfg.output_png).parent).mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.output_png, dpi=220)

    # Interactive HTML (plotly) with precise hover
    try:
        import plotly.graph_objects as go
        # Build axes labels and raw tokens for maximum precision hover
        x_labels = [_clean_tok(tok) if isinstance(tok, str) else str(tok) for tok in tokens_view]
        y_labels = x_labels  # query and key span match in this window
        # Raw strings decoded per token id to avoid BPE artifacts like 'Ġ'
        def _decode_tok(tid: int) -> str:
            try:
                return tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            except Exception:
                return str(tid)
        x_raw = [_decode_tok(tid) for tid in ids_view]
        y_raw = x_raw
        hover = (
            "<b>Query</b> idx=%{y} clean='%{customdata[1]}' raw='%{customdata[3]}'<br>"
            "<b>Key</b> idx=%{x} clean='%{customdata[0]}' raw='%{customdata[2]}'<br>"
            "<b>Value</b> %{z:.6f}"
        )
        # customdata per cell: [key_clean, query_clean, key_raw, query_raw]
        import numpy as _np
        cdata = _np.empty((len(y_labels), len(x_labels), 4), dtype=object)
        cdata[:,:,0] = _np.tile(_np.array(x_labels, dtype=object), (len(y_labels),1))
        cdata[:,:,2] = _np.tile(_np.array(x_raw, dtype=object), (len(y_labels),1))
        cdata[:,:,1] = _np.tile(_np.array(y_labels, dtype=object)[:,None], (1,len(x_labels)))
        cdata[:,:,3] = _np.tile(_np.array(y_raw, dtype=object)[:,None], (1,len(x_labels)))
        fig = go.Figure(data=go.Heatmap(
            z=heat_view,
            x=list(range(start, end)),
            y=list(range(start, end)),
            colorscale='Viridis',
            colorbar=dict(title='Attention Strength'),
            hovertemplate=hover,
            customdata=cdata,
            zmin=float(q1), zmax=float(q99)
        ))
        fig.update_layout(
            title=f"NeurInSpectre — {title_prefix} Patterns (Interactive)",
            xaxis_title="Key token index",
            yaxis_title="Query token index",
            width=max(900, 20*len(x_labels)), height=650,
            margin=dict(l=70, r=40, t=60, b=120)
        )
        # Permanent Red/Blue keys and hover legend
        red_tip = (
            "Red: amplify off‑diagonal columns at attacker tokens (e.g., exfiltrate, SECRET_KEY, <PAYLOAD>); "
            "maintain long bands across time to sustain control."
        )
        blue_tip = (
            "Blue: compare baseline vs injected; block when new bright columns appear at attacker tokens; "
            "sanitize/retokenize delimiters until columns collapse."
        )
        hover_tip = (
            "Hover shows Query token, Key token, and exact attention value – tokens are from the prompt."
        )
        fig.add_annotation(x=0, y=-0.10, xref='paper', yref='paper', showarrow=False,
                           text=f"<b>Red Team Key</b>: {red_tip}",
                           align='left', font=dict(size=12),
                           bgcolor='#ffe6e6', bordercolor='#cc0000', borderwidth=1.2)
        fig.add_annotation(x=0, y=-0.17, xref='paper', yref='paper', showarrow=False,
                           text=f"<b>Blue Team Key</b>: {blue_tip}",
                           align='left', font=dict(size=12),
                           bgcolor='#e6f0ff', bordercolor='#1f5fbf', borderwidth=1.2)
        fig.add_annotation(x=0, y=-0.24, xref='paper', yref='paper', showarrow=False,
                           text=f"<b>Hover</b>: {hover_tip}",
                           align='left', font=dict(size=12),
                           bgcolor='#f7f7f7', bordercolor='#666666', borderwidth=1.0)
        html_path = f"{cfg.out_prefix}attention_interactive.html"
        _Path(_Path(html_path).parent).mkdir(parents=True, exist_ok=True)
        fig.write_html(html_path, include_plotlyjs='cdn')
    except Exception:
        # Fallback: write a minimal HTML that embeds the PNG (base64) so the file always exists
        import base64 as _b64
        png_path = _Path(cfg.output_png)
        html_path = f"{cfg.out_prefix}attention_interactive.html"
        _Path(_Path(html_path).parent).mkdir(parents=True, exist_ok=True)
        try:
            with open(png_path, 'rb') as f:
                b64 = _b64.b64encode(f.read()).decode('ascii')
            _Path(html_path).write_text(
                """
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NeurInSpectre — Attention (Static Fallback)</title></head>
<body>
<h3>NeurInSpectre — Attention Patterns (Static Fallback)</h3>
<p>Plotly not available; showing PNG fallback. Install with: pip install plotly</p>
<img style="max-width:100%" src="data:image/png;base64,REPLACE_B64" />
</body></html>
""".replace('REPLACE_B64', b64)
            )
        except Exception:
            # Last resort: write an HTML file linking to the PNG path
            _Path(html_path).write_text(f"<html><body><p>See PNG: {png_path.as_posix()}</p></body></html>")

    return cfg.output_png


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a token×token attention heatmap with security-friendly axis labels"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name, e.g., gpt2")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (default 0)")
    parser.add_argument("--head", type=int, default=0, help="Head index (default 0)")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--output",
        default="attention_heatmap.png",
        help="Output PNG path (default attention_heatmap.png)",
    )
    parser.add_argument("--baseline", default=None, help="Optional baseline prompt for delta view (future use)")
    parser.add_argument("--out-prefix", default="attn_", help="Output prefix for interactive HTML and summaries")

    args = parser.parse_args(argv)

    cfg = AttentionConfig(
        model_name=args.model,
        prompt=args.prompt,
        layer=args.layer,
        head=args.head,
        device=args.device,
        output_png=args.output,
        out_prefix=args.out_prefix,
        baseline=args.baseline,
    )

    try:
        out = generate_attention_heatmap(cfg)
        print(f"✅ Attention heatmap saved to: {out}")
        return 0
    except Exception as e:
        print(f"❌ Error generating attention heatmap: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


