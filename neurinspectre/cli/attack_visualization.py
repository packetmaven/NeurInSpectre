#!/usr/bin/env python3
"""
NeurInSpectre ATLAS Attack Graph Visualization
Part of the Time Travel Debugger suite
"""

import logging
import json
from pathlib import Path
import math

try:
    import networkx as nx
    import plotly.graph_objects as go
except Exception:
    nx = None
    go = None

logger = logging.getLogger(__name__)

# MITRE ATLAS tactic color mapping (approximate palette)
ATLAS_PHASE_COLORS = {
    'AI Model Access': '#2563eb',
    'AI Attack Staging': '#dc2626',
    'Reconnaissance': '#1f77b4',
    'Resource Development': '#0891b2',
    'Initial Access': '#2ca02c',
    'Execution': '#f97316',
    'Persistence': '#9467bd',
    'Defense Evasion': '#8c564b',
    'Infrastructure': '#7c3aed',
    'Collection': '#17becf',
    'Exfiltration': '#ff7f0e',
    'Impact': '#ff9896',
    'Credential Access': '#e377c2',
    'Discovery': '#7f7f7f',
    'Privilege Escalation': '#d62728',
    'Lateral Movement': '#bcbd22',
    # Back-compat (not an ATLAS tactic; kept if older inputs contain it)
    'Command and Control': '#aec7e8',
}

def _short_label(label: str, max_len: int = 22) -> str:
    """Return a display-friendly label (underscores→spaces, trimmed with ellipsis).
    This keeps hover text full-length while improving on-canvas readability.
    """
    if not isinstance(label, str):
        label = str(label)
    disp = label.replace('_', ' ')
    if len(disp) > max_len:
        return disp[: max_len - 1] + '…'
    return disp

def run_attack_visualization(args):
    """Run ATLAS attack graph visualization"""
    open_after = bool(getattr(args, "open", False))
    logger.info("🗺️ Running ATLAS attack graph visualization...")
    logger.info(f"📊 Input path: {args.input_path}")
    logger.info(f"📸 Output path: {args.output_path}")
    logger.info(f"📋 Title: {args.title}")

    def _maybe_open_html(p: Path) -> None:
        if not open_after:
            return
        try:
            import webbrowser
            url = p.resolve().as_uri()
            ok = webbrowser.open(url)
            if ok:
                logger.info(f"🌐 Opened in browser: {url}")
            else:
                logger.info(f"🌐 Open in browser: {p}")
        except Exception as e:
            logger.warning(f"⚠️ Auto-open failed ({e}). Open manually: {p}")
    
    # Check if input file exists (no synthetic/demo fallback)
    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error(f"❌ Input file not found: {args.input_path}")
        logger.error("   No sample data will be generated. Provide a real JSON produced by `neurinspectre attack-graph prepare`.")
        return 1
    
    try:
        # Load attack data
        with open(input_path, 'r') as f:
            attack_data = json.load(f)
        # Remove dataset-level timestamp usage; we will prefer per-node timestamps in hover if present

        # If provided a combined AI/ML attack metadata file, convert to ATLAS-style nodes/edges
        if 'nodes' not in attack_data and 'datasets' in attack_data:
            ds = attack_data['datasets']
            nodes = []
            edges = []
            # Create a minimal set of phase nodes based on presence
            used_phases = set()
            # Map tactic names to official AML.TA IDs (vendored STIX)
            try:
                from ..mitre_atlas.registry import list_atlas_tactics
                tactic_name_to_id = {t.name: t.tactic_id for t in list_atlas_tactics()}
            except Exception:
                tactic_name_to_id = {}

            # Map common attack keys to official AML.T technique IDs
            technique_id_map = {
                'prompt_injection': 'AML.T0051',
                'jailbreak': 'AML.T0054',
                'model_extraction': 'AML.T0024.002',
                'data_poisoning': 'AML.T0020',
                'fine_tuning_poisoning': 'AML.T0020',
                'backdoor': 'AML.T0043.004',
                'watermark_removal': 'AML.T0031',
                'tool_abuse': 'AML.T0053',
                'ts_inverse': 'AML.T0024.001',
                'concretizer': 'AML.T0024.001',
                'ednn': 'AML.T0070',
                'demarking': 'AML.T0068',
                'transport_evasion': 'AML.T0015',
                'rlhf_bypass': 'AML.T0054',
            }
            def add_phase(phase_name: str):
                if phase_name not in used_phases:
                    nodes.append({
                        'id': phase_name,
                        'label': phase_name,
                        'atlas_phase': phase_name,
                        'atlas_id': tactic_name_to_id.get(phase_name, f'UNMAPPED-{phase_name.replace(" ", "-")}')
                    })
                    used_phases.add(phase_name)
            # Map dataset keys to ATLAS phases
            phase_map = {
                # Existing keys
                'ts_inverse': 'Exfiltration',
                'concretizer': 'Exfiltration',
                'ednn': 'Collection',
                'demarking': 'Defense Evasion',
                'transport_evasion': 'Defense Evasion',
                # AI attack families from academic literature and ATLAS case studies
                'prompt_injection': 'Execution',
                'jailbreak': 'Defense Evasion',
                'model_extraction': 'Exfiltration',
                'data_poisoning': 'Persistence',
                'fine_tuning_poisoning': 'Persistence',
                'backdoor': 'Persistence',
                'watermark_removal': 'Impact',
                'tool_abuse': 'Execution',
                'exfiltration': 'Exfiltration',
                'supply_chain': 'Initial Access',
                'rlhf_bypass': 'Defense Evasion',
            }
            for key, meta in ds.items():
                phase = phase_map.get(key, 'Discovery')
                add_phase(phase)
                label = meta.get('attack_method', key)
                # Build technique node with metrics inline in label
                nodes.append({
                    'id': key,
                    'label': label,
                    'atlas_phase': phase,
                    'atlas_id': technique_id_map.get(key, f'UNMAPPED-{key}'),
                    'metrics': meta,
                    # Optional per-node timestamp if present in meta, e.g., meta['timestamp']
                    'timestamp': meta.get('timestamp')
                })
                edges.append({'source': phase, 'target': key})
            # Simple flow: if both Exfiltration and Defense Evasion present, connect Evasion -> Exfiltration
            if 'Defense Evasion' in used_phases and 'Exfiltration' in used_phases:
                edges.append({'source': 'Defense Evasion', 'target': 'Exfiltration'})
        else:
            nodes = attack_data.get('nodes', [])
            edges = attack_data.get('edges', [])
        
        logger.info(f"✅ Loaded attack data: {len(nodes)} nodes, {len(edges)} edges")
        
        # Build interactive graph if dependencies are available
        logger.info("🎨 Generating attack graph visualization...")
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if go is None:
            # Fallback: minimal HTML with data listing
            with open(output_path, 'w') as f:
                f.write(f"<h2>{args.title}</h2>\n<p>Nodes: {len(nodes)} | Edges: {len(edges)}</p>\n")
                f.write("<ul>" + "".join([f"<li>{json.dumps(n)}</li>" for n in nodes]) + "</ul>")
            logger.info("ℹ️ plotly not available; wrote minimal HTML.")
            _maybe_open_html(output_path)
            return 0

        # Build positions and geometry
        node_ids = [n.get('id') or n.get('name') for n in nodes if (n.get('id') or n.get('name')) is not None]
        if not node_ids:
            with open(output_path, 'w') as f:
                f.write(f"<h2>{args.title}</h2><p>No nodes/edges to render.</p>")
            _maybe_open_html(output_path)
            return 0

        if nx is not None:
            G = nx.DiGraph()
            for n in nodes:
                nid = n.get('id') or n.get('name')
                if nid is None:
                    continue
                G.add_node(nid, **n)
            for e in edges:
                s = e.get('source')
                t = e.get('target')
                if s is None or t is None:
                    continue
                G.add_edge(s, t, **e)
            # Centrality for sizing/criticality
            try:
                centrality = nx.pagerank(G)
            except Exception:
                centrality = {n: 1.0 for n in G.nodes()}
            pos = nx.spring_layout(G, seed=42, k=0.7 / math.sqrt(max(1, G.number_of_nodes())))
            degrees = {n: G.degree[n] for n in G.nodes()}
            edge_list = list(G.edges())
        else:
            # Circular layout without networkx
            n = len(node_ids)
            pos = {}
            for i, nid in enumerate(node_ids):
                ang = 2 * math.pi * i / max(1, n)
                pos[nid] = (math.cos(ang), math.sin(ang))
            # Approximate degree and centrality
            degrees = {nid: 0 for nid in node_ids}
            centrality = {nid: 1.0 for nid in node_ids}
            edge_list = []
            for e in edges:
                s = e.get('source')
                t = e.get('target')
                if s in pos and t in pos:
                    edge_list.append((s, t))
                    degrees[s] = degrees.get(s, 0) + 1
                    degrees[t] = degrees.get(t, 0) + 1

        # Edge traces
        edge_x, edge_y = [], []
        for u, v in edge_list:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=1, color='#888'),
                                hoverinfo='none',
                                mode='lines',
                                name='Edges')

        # Node traces
        node_x, node_y, texts, colors, sizes = [], [], [], [], []
        max_deg = max(1, max(degrees.values()) if degrees else 1)
        for n in node_ids:
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            # Pull attributes
            data = next((d for d in nodes if (d.get('id') or d.get('name')) == n), {})
            label = data.get('label', n)
            # ATLAS fields
            atlas_phase = data.get('atlas_phase')
            atlas_id = data.get('atlas_id', '')
            ntype = data.get('type', 'node')
            # Hover text prioritizes ATLAS
            if atlas_phase:
                # Append key metrics and per-node timestamp if present
                metrics = data.get('metrics')
                ts = data.get('timestamp')
                extras = []
                if isinstance(metrics, dict) and metrics:
                    items = list(metrics.items())[:3]
                    extras.append(", ".join([f"{k}={v}" for k,v in items]))
                if ts is not None:
                    extras.append(f"ts={ts}")
                extra_txt = ("<br>" + "; ".join(extras)) if extras else ""
                texts.append(f"{label} — {atlas_phase} ({atlas_id}){extra_txt}")
            else:
                texts.append(f"{label} ({ntype})")
            # color by ATLAS phase, fallback to type
            if atlas_phase and atlas_phase in ATLAS_PHASE_COLORS:
                color = ATLAS_PHASE_COLORS[atlas_phase]
            else:
                color = {'phase': '#1f77b4', 'technique': '#2ca02c', 'node': '#9467bd'}.get(ntype, '#9467bd')
            colors.append(color)
            sz = 16 + 8 * (degrees.get(n, 0) / max_deg) + 12 * float(centrality.get(n, 1.0))
            sizes.append(sz)
        # Compute readable display labels and text sizes that scale with node size
        display_labels = []
        for n in node_ids:
            data = next((d for d in nodes if (d.get('id') or d.get('name')) == n), {})
            lbl = data.get('label', n)
            display_labels.append(_short_label(lbl))
        # Scale text size between 10 and 16 based on marker size
        if sizes:
            smin, smax = min(sizes), max(sizes)
            span = (smax - smin) or 1.0
            text_sizes = [int(10 + 6 * ((s - smin) / span)) for s in sizes]
        else:
            text_sizes = 12
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=display_labels,
            textposition='top center',
            textfont=dict(size=text_sizes, color='black'),
            hovertext=texts,
            hoverinfo='text',
            marker=dict(color=colors, size=sizes, line=dict(width=1, color='white')),
            name='Nodes',
            cliponaxis=False,
        )
        # Add a compact legend for centrality quartiles to guide interpretability
        try:
            import numpy as _np
            cent_vals = _np.array([float(centrality.get(n, 1.0)) for n in node_ids])
            q = _np.quantile(cent_vals, [0.25, 0.5, 0.75])
            cent_legend = f"Centrality quartiles: Q1={q[0]:.2f}, Q2={q[1]:.2f}, Q3={q[2]:.2f}"
        except Exception:
            cent_legend = None

        # Arrow annotations for directionality
        arrow_ann = []
        for u, v in edge_list:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            arrow_ann.append(dict(x=x1, y=y1, ax=x0, ay=y0, xref='x', yref='y', axref='x', ayref='y',
                                   showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1, arrowcolor='#888'))

        # Red/Blue guidance
        red_tip = "Red: Pivot via high-centrality ATLAS phases/techniques; chain along dense transitions."
        blue_tip = "Blue: Monitor central phases; add controls on transitions with highest edge density."

        fig = go.Figure(data=[edge_trace, node_trace])
        # Compose a legend for present ATLAS phases
        present_phases = []
        for d in nodes:
            ph = d.get('atlas_phase')
            if ph and ph not in present_phases:
                present_phases.append(ph)
        legend_html = " &nbsp; ".join([f"<span style='display:inline-block;width:10px;height:10px;background:{ATLAS_PHASE_COLORS.get(ph,'#999')};margin-right:4px'></span>{ph}" for ph in present_phases])

        # Centrality quartiles explanation for key box
        cent_explain = (
            "Centrality quartiles: Q1 (low) → peripheral nodes; Q2→mid; "
            "Q3→high influence; Q4 (top)→critical hubs/bridges. "
            "Red: leverage Q3/Q4 to escalate and pivot. Blue: prioritize monitoring, auth, and rate‑limits on Q3/Q4; "
            "treat Q1/Q2 as lower priority unless they spike in degree."
        )

        fig.update_layout(
            title=dict(text=args.title, x=0.5),
            showlegend=False,
            height=900,
            width=1800,
            margin=dict(l=60, r=60, t=80, b=360),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            uniformtext=dict(minsize=10, mode='hide'),
            annotations=arrow_ann + [
                dict(x=0.5, y=-0.18, xref='paper', yref='paper', showarrow=False,
                     text=f"Nodes: {len(node_ids)} | Edges: {len(edge_list)} — ATLAS phases: {legend_html}",
                     font=dict(size=11)),
                dict(x=0.5, y=-0.24, xref='paper', yref='paper', showarrow=False,
                     text=f"Mapping aligns nodes to MITRE ATLAS phases; colors denote phase. {red_tip} {blue_tip}",
                     font=dict(size=12)),
                # Permanent Red/Blue key boxes
                dict(x=0.01, y=-0.36, xref='paper', yref='paper', xanchor='left', showarrow=False,
                     text=f"Red Team Key: {red_tip} Target Q3/Q4 central nodes and connect across phases to maintain momentum.",
                     align='left', font=dict(size=12), bgcolor='#ffe6e6', bordercolor='#cc0000', borderwidth=1.5),
                dict(x=0.01, y=-0.48, xref='paper', yref='paper', xanchor='left', showarrow=False,
                     text=f"Blue Team Key: {blue_tip} Focus telemetry and controls on Q3/Q4 nodes; gate high‑degree transitions.",
                     align='left', font=dict(size=12), bgcolor='#e6f0ff', bordercolor='#1f5fbf', borderwidth=1.5),
                # Centrality quartiles explanation box
                dict(x=0.01, y=-0.60, xref='paper', yref='paper', xanchor='left', showarrow=False,
                     text=cent_explain,
                     align='left', font=dict(size=12), bgcolor='#f7f7f7', bordercolor='#666666', borderwidth=1.2)
            ]
        )
        if cent_legend:
            fig.add_annotation(x=0.5, y=-0.30, xref='paper', yref='paper', showarrow=False,
                               text=cent_legend, font=dict(size=11))

        # Save HTML
        fig.write_html(output_path, include_plotlyjs='cdn')
        logger.info("✅ ATLAS attack visualization complete")
        logger.info(f"📄 HTML saved: {output_path}")
        _maybe_open_html(output_path)
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in input file: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        return 1
    
    return 0 