"""
NeurInSpectre CLI: Red Team Intelligence Commands
Integrates red_team_intelligence.py functionality into the CLI system
"""

import argparse
import numpy as np
import json
import urllib.request as _urlreq
import re as _re
import sys
import os
import logging
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Support running this module as a script from a source checkout.
# When imported as part of the package/CLI, avoid sys.path side effects.
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from ..security.red_team_intelligence import RedTeamIntelligenceEngine
    _RED_TEAM_IMPORT_ERROR = None
except ImportError as e:
    # No mock/synthetic engine: fall back to the CLI's minimal, data-driven plots.
    RedTeamIntelligenceEngine = None  # type: ignore[assignment]
    _RED_TEAM_IMPORT_ERROR = str(e)

def add_red_team_parser(subparsers):
    """Add red team intelligence command to CLI"""
    parser = subparsers.add_parser(
        'red-team',
        help='🔴 Red team operational intelligence tools'
    )
    
    # Subcommands for red team operations
    red_subparsers = parser.add_subparsers(dest='red_command', help='Red team commands')
    
    # Attack planning
    attack_parser = red_subparsers.add_parser('attack-planning', help='Create attack planning dashboard')
    # Mode A (legacy): visualize operator-supplied planning inputs (no simulation)
    attack_parser.add_argument('--target-data', default=None, help='Target data JSON file (legacy mode)')
    attack_parser.add_argument('--attack-vectors', default=None, help='Attack vectors JSON file (legacy mode)')
    # Mode B (recommended): activation-based planning (real model hidden states)
    attack_parser.add_argument('--model', default=None, help='HuggingFace model id for activation-based planning')
    baseline_src = attack_parser.add_mutually_exclusive_group(required=False)
    baseline_src.add_argument('--baseline-prompt', default=None, help='Baseline/benign prompt (single)')
    baseline_src.add_argument('--baseline-file', default=None, help='Text file: one baseline prompt per line (recommended)')
    attack_parser.add_argument('--test-prompt', default=None, help='Test/suspect prompt to compare against baseline')
    attack_parser.add_argument('--threshold', type=float, default=2.5, help='Z threshold (default: 2.5)')
    attack_parser.add_argument('--robust', action='store_true', help='Use robust Z (median/MAD)')
    attack_parser.add_argument('--sigma-floor', type=float, default=None, help='Optional minimum Z denominator (stabilizes near-zero variance)')
    attack_parser.add_argument('--layer-start', type=int, default=0, help='First layer index to include (default: 0)')
    attack_parser.add_argument('--layer-end', type=int, default=None, help='Last layer index to include (inclusive)')
    attack_parser.add_argument('--layer', type=int, default=None, help='Drill-down layer (default: auto-select top anomalous layer)')
    attack_parser.add_argument('--topk', type=int, default=12, help='Top-K changes for drill-down (default: 12)')
    attack_parser.add_argument('--compare', choices=['prefix','last'], default='prefix', help='Compare prefix tokens or last token (default: prefix)')
    attack_parser.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto', help='Device preference')
    attack_parser.add_argument('--output-dashboard', help='Output dashboard HTML file')
    attack_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Vulnerability assessment
    vuln_parser = red_subparsers.add_parser('vuln-assessment', help='Create vulnerability assessment dashboard')
    vuln_parser.add_argument('--vuln-data', required=True, help='Vulnerability data JSON file')
    vuln_parser.add_argument('--exploit-data', required=True, help='Exploit data JSON file')
    vuln_parser.add_argument('--output-dashboard', help='Output dashboard HTML file')
    vuln_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Penetration testing analytics
    pentest_parser = red_subparsers.add_parser('pentest-analytics', help='Create penetration testing analytics')
    pentest_parser.add_argument('--pentest-data', required=True, help='Penetration test data JSON file')
    pentest_parser.add_argument('--campaign-metrics', required=True, help='Campaign metrics JSON file')
    pentest_parser.add_argument('--output-dashboard', help='Output dashboard HTML file')
    pentest_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    parser.set_defaults(func=handle_red_team)

def _load_any_flexible(path_or_url: str):
    """Load JSON or YAML-like content from a local path or URL.
    Returns a Python object (dict/list/str). If parsing as JSON fails,
    tries a simple YAML/text heuristic to extract fields.
    """
    def _try_json_bytes(b: bytes):
        try:
            return json.loads(b.decode('utf-8'))
        except Exception:
            try:
                return json.loads(b.decode('latin-1'))
            except Exception:
                return None
    data = None
    if str(path_or_url).startswith(('http://', 'https://')):
        with _urlreq.urlopen(path_or_url) as r:  # nosec - user-provided URL
            b = r.read()
        obj = _try_json_bytes(b)
        if obj is not None:
            return obj
        # YAML/text fallback: return decoded text
        try:
            return b.decode('utf-8', errors='ignore')
        except Exception:
            return b.decode('latin-1', errors='ignore')
    # Local file
    try:
        with open(path_or_url, 'rb') as f:
            b = f.read()
        obj = _try_json_bytes(b)
        if obj is not None:
            return obj
        return b.decode('utf-8', errors='ignore')
    except Exception as e:
        raise RuntimeError(f"Failed to load {path_or_url}: {e}")

def _normalize_targets_from_any(obj):
    """Normalize various inputs to { 'targets': [ { 'name': ... }, ... ] }.
    - If dict has 'targets': pass through
    - If it's a Nuclei YAML/text, extract id/name/severity into names
    - If list[str]: wrap as names
    """
    if isinstance(obj, dict) and 'targets' in obj:
        return obj
    # List of names
    if isinstance(obj, list):
        names = []
        for it in obj:
            if isinstance(it, str):
                names.append(it)
            elif isinstance(it, dict) and 'name' in it:
                names.append(str(it['name']))
        return {'targets': [{'name': n} for n in names]}
    # CISA KEV feed → list CVE IDs as targets
    if isinstance(obj, dict) and 'vulnerabilities' in obj:
        cves = []
        for v in obj.get('vulnerabilities', [])[:200]:
            cid = v.get('cveID') or v.get('cveId') or v.get('catalogEntryID')
            if cid:
                cves.append({'name': str(cid)})
        if cves:
            return {'targets': cves}
    # YAML/text heuristic for Nuclei template
    if isinstance(obj, str):
        # Grab id, name, severity lines
        name = None; severity = None; nid = None
        for ln in obj.splitlines():
            m = _re.match(r"^\s*id:\s*(.+)$", ln)
            if m and not nid:
                nid = m.group(1).strip()
            m = _re.match(r"^\s*name:\s*(.+)$", ln)
            if m and not name:
                name = m.group(1).strip()
            m = _re.match(r"^\s*severity:\s*(.+)$", ln)
            if m and not severity:
                severity = m.group(1).strip()
        label = name or nid or 'target'
        return {'targets': [{'name': label, 'severity': severity or 'unknown'}]}
    # Dict without 'targets'
    if isinstance(obj, dict):
        keys = list(obj.keys())[:10]
        return {'targets': [{'name': k} for k in keys]}
    return {'targets': [{'name': 'target'}]}

def _normalize_vectors_from_any(obj):
    if isinstance(obj, dict) and 'vectors' in obj:
        return obj
    if isinstance(obj, list):
        vecs = [str(v) for v in obj]
        return {'vectors': vecs}
    if isinstance(obj, str):
        # Split by lines/commas to get vector names
        raw = [x.strip() for x in _re.split(r"[\n,]", obj) if x.strip()]
        return {'vectors': raw[:12] or ['phishing','initial-access','credential-dumping']}
    if isinstance(obj, dict):
        return {'vectors': list(obj.keys())[:10] or ['phishing','initial-access']}
    return {'vectors': ['phishing','initial-access']}

def handle_red_team(args):
    """Handle red team intelligence commands"""
    if args.red_command == 'attack-planning':
        return handle_attack_planning(args)
    elif args.red_command == 'vuln-assessment':
        return handle_vulnerability_assessment(args)
    elif args.red_command == 'pentest-analytics':
        return handle_pentest_analytics(args)
    else:
        print("❌ No red team command specified. Use --help for options.")
        return 1

def handle_attack_planning(args):
    """Handle attack planning dashboard creation (real-activation mode + legacy JSON mode)."""

    # Activation mode (recommended): requires model + baseline + test prompt
    if getattr(args, 'model', None) and getattr(args, 'test_prompt', None) and (
        getattr(args, 'baseline_prompt', None) or getattr(args, 'baseline_file', None)
    ):
        return _handle_attack_planning_from_activations(args)

    # Legacy JSON mode (operator inputs, no invented metrics)
    if args.verbose:
        print("🔴 Creating attack planning dashboard (operator inputs)...")
        print(f"📁 Target data: {getattr(args, 'target_data', None)}")
        print(f"⚔️  Attack vectors: {getattr(args, 'attack_vectors', None)}")
    if not getattr(args, 'target_data', None) or not getattr(args, 'attack_vectors', None):
        print(
            "❌ Missing inputs. Provide either:\n"
            "  - Activation mode: --model + (--baseline-prompt|--baseline-file) + --test-prompt\n"
            "  - JSON mode: --target-data + --attack-vectors\n"
        )
        return 1

    # Load data (file or URL) with flexible normalization
    try:
        raw_targets = _load_any_flexible(args.target_data)
        raw_vectors = _load_any_flexible(args.attack_vectors)
        target_data = _normalize_targets_from_any(raw_targets)
        attack_vectors = _normalize_vectors_from_any(raw_vectors)
        if args.verbose:
            print("✅ Data loaded & normalized")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1

    # JSON-only visualization: show targets and vectors as provided.
    from plotly.subplots import make_subplots as _mk
    import plotly.graph_objects as _go

    fig = _mk(
        rows=1, cols=2, column_widths=[0.55, 0.45],
        specs=[[{"type": "table"}, {"type": "table"}]],
        subplot_titles=("Targets / Assets (as provided)", "Attack Vectors (as provided)"),
    )

    tgts = target_data.get('targets', []) if isinstance(target_data, dict) else []
    t_names, t_sev = [], []
    for t in tgts[:200]:
        if isinstance(t, dict):
            t_names.append(str(t.get('name', '?')))
            t_sev.append(str(t.get('severity', t.get('sev', 'unknown'))))
        else:
            t_names.append(str(t))
            t_sev.append('unknown')

    if t_names:
        fig.add_trace(
            _go.Table(header=dict(values=["Target", "Severity"]), cells=dict(values=[t_names, t_sev])),
            row=1, col=1
        )
    else:
        fig.add_trace(
            _go.Table(header=dict(values=["Note"]), cells=dict(values=[["No targets provided"]])),
            row=1, col=1
        )

    vecs = attack_vectors.get('vectors', []) if isinstance(attack_vectors, dict) else []
    v_vals = [str(v) for v in vecs[:200]]
    if v_vals:
        fig.add_trace(
            _go.Table(header=dict(values=["Vector"]), cells=dict(values=[v_vals])),
            row=1, col=2
        )
    else:
        fig.add_trace(
            _go.Table(header=dict(values=["Note"]), cells=dict(values=[["No vectors provided"]])),
            row=1, col=2
        )

    fig.update_layout(
        title_text='🔴 Attack Planning (Operator Inputs — no model telemetry)',
        height=560, width=1200,
        margin=dict(l=60, r=40, t=70, b=80),
    )

    output_file = args.output_dashboard or f"red_team_attack_planning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    fig.write_html(output_file)
    print(f"✅ Attack planning dashboard saved to: {output_file}")
    return 0


def _handle_attack_planning_from_activations(args) -> bool:
    """Activation-based red-team planning dashboard (real hidden states; no simulation)."""
    from pathlib import Path

    import torch
    import plotly.graph_objects as _go
    import plotly.io as _pio
    from transformers import AutoModel, AutoTokenizer

    from ..visualization.dna_visualizer import (
        plot_anomaly_detection,
        plot_attack_patterns,
        _compute_layer_anomaly_metrics,
        _to_numpy,
    )

    # Resolve device
    dev = getattr(args, 'device', 'auto')
    if dev == 'auto':
        if torch.cuda.is_available():
            dev = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            dev = 'mps'
        else:
            dev = 'cpu'

    if getattr(args, 'verbose', False):
        print("🔴 Activation-based attack planning dashboard...")
        print(f"🧠 Model: {args.model}")
        print(f"🧩 Device: {dev}")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Prefer safetensors to avoid unsafe torch.load on older torch versions
    try:
        mdl = AutoModel.from_pretrained(args.model, use_safetensors=True)
    except Exception:
        try:
            mdl = AutoModel.from_pretrained(args.model)
        except Exception as e:
            raise RuntimeError(
                "Failed to load model weights. Prefer models that ship 'safetensors' weights, "
                "or upgrade torch (>=2.6) to load legacy .bin weights safely."
            ) from e
    mdl.eval()
    mdl.to(dev)

    layer_start = int(getattr(args, 'layer_start', 0) or 0)
    layer_end = getattr(args, 'layer_end', None)
    layer_end = int(layer_end) if layer_end is not None else None
    if layer_start < 0:
        layer_start = 0
    if layer_end is not None and layer_end < layer_start:
        raise ValueError('--layer-end must be >= --layer-start')

    def _hidden_states(prompt: str):
        inputs = tok(prompt, return_tensors='pt', truncation=True)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl(**inputs, output_hidden_states=True, return_dict=True)
        hs = getattr(out, 'hidden_states', None)
        if hs is None:
            raise ValueError('Model did not return hidden_states; ensure output_hidden_states=True')
        layers = list(hs[1:])  # drop embedding output
        layers = layers[layer_start:(layer_end + 1) if layer_end is not None else None]
        return {f'layer_{i + layer_start}': layers[i] for i in range(len(layers))}

    # Baseline prompts (for statistics)
    baseline_prompts: list[str] = []
    if getattr(args, 'baseline_file', None):
        p = Path(args.baseline_file)
        baseline_prompts = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        if not baseline_prompts:
            raise ValueError('baseline-file is empty')
    elif getattr(args, 'baseline_prompt', None):
        baseline_prompts = [str(args.baseline_prompt)]
    else:
        raise ValueError('Provide --baseline-prompt or --baseline-file')

    test_prompt = str(getattr(args, 'test_prompt', '') or '')
    if not test_prompt:
        raise ValueError('Provide --test-prompt')

    # Baseline dict for anomaly stats: concatenate prompts along batch dimension.
    base_dicts = [_hidden_states(pr) for pr in baseline_prompts]
    baseline = {}
    for layer in base_dicts[0].keys():
        # Hidden states are typically shaped [B, T, D]. Baseline prompts often have
        # different token lengths (T), so concatenating on dim=0 will fail unless
        # sequences are padded to a common length. For robust layer-wise statistics,
        # treat each token as a sample by flattening [B, T, D] -> [B*T, D] first.
        tensors = []
        for d in base_dicts:
            t = d[layer]
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=dev)
            if t.dim() >= 2:
                t = t.reshape(-1, t.shape[-1])
            else:
                t = t.reshape(1, -1)
            tensors.append(t)
        baseline[layer] = torch.cat(tensors, dim=0)

    test = _hidden_states(test_prompt)

    # Per-layer metrics for triage + auto drill-down selection.
    common = sorted(set(baseline.keys()) & set(test.keys()), key=lambda k: int(str(k).split('_')[-1]))
    rows = []
    for k in common:
        base_np = _to_numpy(baseline[k])
        test_np = _to_numpy(test[k])
        (cnt, maxz, l2, cos, _n0, _n1, _d) = _compute_layer_anomaly_metrics(
            base_np,
            test_np,
            threshold=float(getattr(args, 'threshold', 2.5)),
            robust=bool(getattr(args, 'robust', False)),
            sigma_floor=getattr(args, 'sigma_floor', None),
        )
        rows.append((k, cnt, maxz, l2, cos))
    if not rows:
        raise ValueError('No overlapping layers to analyze')

    drill_layer = getattr(args, 'layer', None)
    if drill_layer is None:
        best = sorted(rows, key=lambda r: (r[1], r[2]), reverse=True)[0]
        drill_layer = int(str(best[0]).split('_')[-1])

    baseline_prompt_for_tokens = str(getattr(args, 'baseline_prompt', None) or baseline_prompts[0])
    b_ids = tok(baseline_prompt_for_tokens, return_tensors='pt', truncation=True)['input_ids'][0].tolist()
    t_ids = tok(test_prompt, return_tensors='pt', truncation=True)['input_ids'][0].tolist()
    b_toks = tok.convert_ids_to_tokens(b_ids)
    t_toks = tok.convert_ids_to_tokens(t_ids)

    # Drill-down uses a single baseline prompt for token alignment.
    baseline_single = _hidden_states(baseline_prompt_for_tokens)
    test_single = _hidden_states(test_prompt)

    fig_anom = plot_anomaly_detection(
        baseline,
        test,
        threshold=float(getattr(args, 'threshold', 2.5)),
        robust=bool(getattr(args, 'robust', False)),
        sigma_floor=getattr(args, 'sigma_floor', None),
        title='NeurInSpectre — 🔴 Red Team Attack Planning (Layer-wise anomalies)',
    )

    fig_attack = plot_attack_patterns(
        baseline_single,
        test_single,
        layer_idx=int(drill_layer),
        top_k=int(getattr(args, 'topk', 12)),
        compare=str(getattr(args, 'compare', 'prefix')),
        title=f'Neuron Activation Changes — drill-down (layer {int(drill_layer)})',
        baseline_tokens=b_toks,
        test_tokens=t_toks,
    )

    top = sorted(rows, key=lambda r: (r[1], r[2]), reverse=True)[:10]
    fig_table = _go.Figure(
        data=[
            _go.Table(
                header=dict(values=['Layer', 'Anomaly Count', 'Max |Z|', 'Mean-shift L2', 'Cosine(mean)']),
                cells=dict(values=[
                    [r[0] for r in top],
                    [r[1] for r in top],
                    [f'{r[2]:.2f}' for r in top],
                    [f'{r[3]:.2f}' for r in top],
                    [f'{r[4]:.3f}' for r in top],
                ]),
            )
        ]
    )
    fig_table.update_layout(title='Top affected layers (prioritize repeatable peaks)', height=380, margin=dict(l=40, r=40, t=60, b=40))

    citations = """\
<ul>
  <li><b>The Rogue Scalpel</b> (Sep 2025): activation steering (even random directions) can increase harmful compliance. <a href="https://arxiv.org/abs/2509.22067">arXiv:2509.22067</a></li>
  <li><b>AlphaSteer</b> (Jun 2025): conditional/null-space steering to preserve utility while enforcing refusal. <a href="https://arxiv.org/abs/2506.07022">arXiv:2506.07022</a></li>
  <li><b>LLM salting</b> (Sophos, Oct 2025): rotate refusal subspace to break jailbreak transfer. <a href="https://news.sophos.com/en-us/2025/10/24/locking-it-down-a-new-technique-to-prevent-llm-jailbreaks/">blog</a></li>
  <li><b>UTDMF</b> (Oct 2025): generalized activation patching for prompt-injection/deception mitigation. <a href="https://arxiv.org/abs/2510.04528">arXiv:2510.04528</a></li>
</ul>
"""

    guidance = """\
<h2>HOW/WHY to act on this dashboard</h2>
<ul>
  <li><b>WHY layers matter</b>: repeatable peaks suggest controllable internal features; they are candidate leverage points for steering and patching.</li>
  <li><b>HOW to prioritize</b>: prefer layers that stay high across a baseline prompt suite (<code>--baseline-file</code>) and across prompt variants.</li>
  <li><b>HOW to validate</b>: compare suspect prompts vs a clean baseline suite; measure repeatability across seeds/runs and track whether anomalies concentrate in specific layers/tokens.</li>
</ul>
"""
    parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>NeurInSpectre — Red Team Attack Planning</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;}",
        "h1{margin-bottom:8px;} h2{margin-top:28px;} .note{color:#444;max-width:1100px;}",
        "code{background:#f4f4f4;padding:2px 4px;border-radius:4px;}",
        "</style></head><body>",
        "<h1>🔴 NeurInSpectre — Attack Planning (Activation Telemetry)</h1>",
        "<p class='note'><b>Scope:</b> Authorized red-team measurement and blue-team hardening only. All plots below are computed from real hidden states.</p>",
        "<h2>1) Layer triage</h2>",
        _pio.to_html(fig_anom, include_plotlyjs='cdn', full_html=False),
        "<h2>2) Top affected layers</h2>",
        _pio.to_html(fig_table, include_plotlyjs=False, full_html=False),
        "<h2>3) Drill-down (selected layer)</h2>",
        _pio.to_html(fig_attack, include_plotlyjs=False, full_html=False),
        guidance,
        "<h2>Evidence (recent)</h2>",
        citations,
        "</body></html>",
    ]

    body = "".join(parts)

    out_path = Path(getattr(args, 'output_dashboard', None) or f"red_team_attack_planning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding='utf-8')
    print(f"✅ Attack planning dashboard saved to: {out_path}")
    return 0


def handle_vulnerability_assessment(args):

    """Handle vulnerability assessment dashboard creation"""
    if args.verbose:
        print("🔴 Creating vulnerability assessment dashboard...")
        print(f"🔍 Vulnerability data: {args.vuln_data}")
        print(f"💥 Exploit data: {args.exploit_data}")
    
    # Load data
    try:
        with open(args.vuln_data, 'r') as f:
            vuln_data = json.load(f)
        with open(args.exploit_data, 'r') as f:
            exploit_data = json.load(f)
        if args.verbose:
            print("✅ Data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    dashboard = None
    if args.verbose:
        print("🔍 Generating vulnerability assessment dashboard...")

    # Prefer the full engine when available, but do not hard-fail if optional deps
    # are missing in a minimal installation.
    if RedTeamIntelligenceEngine is not None:
        try:
            engine = RedTeamIntelligenceEngine()
            dashboard = engine.create_vulnerability_assessment_dashboard(vuln_data, exploit_data)
        except Exception:
            dashboard = None
    else:
        if _RED_TEAM_IMPORT_ERROR:
            logger.warning("RedTeamIntelligenceEngine unavailable (%s); using fallback dashboard.", _RED_TEAM_IMPORT_ERROR)

    if dashboard is None or not isinstance(dashboard, go.Figure):
        try:
            from plotly.subplots import make_subplots as _mk
            import plotly.graph_objects as _go
            fig = _mk(rows=1, cols=3, column_widths=[0.45,0.3,0.25],
                      specs=[[{"type":"table"},{"type":"xy"},{"type":"domain"}]],
                      subplot_titles=("Vulnerabilities","Severity Counts","Exploit Availability"))
            vulns = vuln_data.get('vulns', []) if isinstance(vuln_data, dict) else []
            cves = [str(v.get('cve','?')) for v in vulns]
            sevs = [str(v.get('sev','unknown')).title() for v in vulns]
            if cves:
                fig.add_trace(_go.Table(header=dict(values=["CVE","Severity"]),
                                        cells=dict(values=[cves, sevs])), row=1, col=1)
            # Severity counts
            counts = {}
            for s in sevs:
                counts[s] = counts.get(s, 0) + 1
            if counts:
                fig.add_trace(_go.Bar(x=list(counts.keys()), y=list(counts.values()), name='Severity'), row=1, col=2)
            # Exploit availability pie
            exps = exploit_data.get('exploits', []) if isinstance(exploit_data, dict) else []
            pie_labels = ['Exploits','No Exploits']
            pie_vals = [len(exps), max(0, len(vulns)-len(exps))]
            fig.add_trace(_go.Pie(labels=pie_labels, values=pie_vals, hole=0.4), row=1, col=3)
            fig.update_layout(title_text='🔴 Vulnerability Assessment Overview', height=560, width=1200,
                              legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center'),
                              margin=dict(l=60,r=40,t=60,b=120))
            dashboard = fig
        except Exception:
            pass
    
    # Save dashboard
    output_file = args.output_dashboard or f"red_team_vuln_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    dashboard.write_html(output_file)
    
    print(f"✅ Vulnerability assessment dashboard saved to: {output_file}")
    return 0

def handle_pentest_analytics(args):
    """Handle penetration testing analytics dashboard creation"""
    if args.verbose:
        print("🔴 Creating penetration testing analytics dashboard...")
        print(f"🎯 Pentest data: {args.pentest_data}")
        print(f"📊 Campaign metrics: {args.campaign_metrics}")
    
    # Load data
    try:
        with open(args.pentest_data, 'r') as f:
            pentest_data = json.load(f)
        with open(args.campaign_metrics, 'r') as f:
            campaign_metrics = json.load(f)
        if args.verbose:
            print("✅ Data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    dashboard = None
    if args.verbose:
        print("🔍 Generating penetration testing analytics dashboard...")

    # Prefer the full engine when available, but do not hard-fail if optional deps
    # are missing in a minimal installation.
    if RedTeamIntelligenceEngine is not None:
        try:
            engine = RedTeamIntelligenceEngine()
            dashboard = engine.create_penetration_testing_analytics(pentest_data, campaign_metrics)
        except Exception:
            dashboard = None
    else:
        if _RED_TEAM_IMPORT_ERROR:
            logger.warning("RedTeamIntelligenceEngine unavailable (%s); using fallback dashboard.", _RED_TEAM_IMPORT_ERROR)

    if dashboard is None or not isinstance(dashboard, go.Figure):
        try:
            from plotly.subplots import make_subplots as _mk
            import plotly.graph_objects as _go
            fig = _mk(rows=1, cols=3, column_widths=[0.4,0.3,0.3],
                      specs=[[{"type":"table"},{"type":"xy"},{"type":"domain"}]],
                      subplot_titles=("Findings","Coverage","Campaign Status"))
            # Findings table
            fnds = pentest_data.get('findings', []) if isinstance(pentest_data, dict) else []
            ids = [str(x.get('id','?')) for x in fnds]
            sev = [str(x.get('sev','n/a')).title() for x in fnds]
            if ids:
                fig.add_trace(_go.Table(header=dict(values=["ID","Severity"]),
                                        cells=dict(values=[ids, sev])), row=1, col=1)
            # Coverage bars
            cov = campaign_metrics.get('coverage', {}) if isinstance(campaign_metrics, dict) else {}
            if cov:
                fig.add_trace(_go.Bar(x=list(cov.keys()), y=list(cov.values()), name='Coverage'), row=1, col=2)
            # Status gauge (synthetic score from coverage)
            try:
                import numpy as _np
                vals = list(cov.values()) or [0]
                score = float(min(1.0, max(0.0, _np.mean(vals)/max(max(vals),1))))
            except Exception:
                score = 0.0
            fig.add_trace(_go.Indicator(mode='gauge+number', value=score, title={'text':'Readiness'}), row=1, col=3)
            fig.update_layout(title_text='🔴 Penetration Testing Analytics', height=560, width=1200,
                              legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center'),
                              margin=dict(l=60,r=40,t=60,b=120))
            dashboard = fig
        except Exception:
            pass
    
    # Save dashboard
    output_file = args.output_dashboard or f"red_team_pentest_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    dashboard.write_html(output_file)
    
    print(f"✅ Penetration testing analytics dashboard saved to: {output_file}")
    return 0 