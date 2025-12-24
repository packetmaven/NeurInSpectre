"""
Layer-Level Causal Impact Visualization for Neural Network Security Analysis

This module implements layer-level causal attribution analysis based on recent
research in adversarial AI security, including:

1. "SoK: Comprehensive Causality Analysis Framework for LLM Security" (Dec 2025)
   - Localized safety mechanisms in early-to-middle layers
   - Empirical findings are benchmark-dependent; use this tool to visualize layer sensitivity
   - Layer-level interventions for analysis and mitigation workflows

2. "Backdoor Attribution: Elucidating and Controlling Backdoor in LLMs" (Sep 2025)
   - Layer-specific backdoor feature processing (paper context)

3. "HAct: Activation Clustering for Attack Detection" (2024)
   - Activation histogram analysis for OOD / anomaly detection (paper context; benchmark-dependent)

Security Applications:
- Monitor layer-level divergence/shift for anomaly detection
- Identify “hot” layers (e.g., top-percentile impact) for focused inspection/mitigation
- Support incident response by summarizing which layers change most under suspected attack conditions
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, List, Tuple, Any
import torch
import logging

logger = logging.getLogger(__name__)


def compute_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute KL divergence KL(P||Q) with numerical stability.
    
    Args:
        p: Reference distribution (baseline)
        q: Comparison distribution (test/adversarial)
        epsilon: Smoothing factor to prevent log(0)
    
    Returns:
        KL divergence value
    """
    p = np.asarray(p).flatten() + epsilon
    q = np.asarray(q).flatten() + epsilon
    
    # Normalize to valid probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(p * np.log(p / q))


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric, bounded).
    
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    """
    m = 0.5 * (p + q)
    return 0.5 * compute_kl_divergence(p, m) + 0.5 * compute_kl_divergence(q, m)


def extract_layer_activations(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str = 'cpu'
) -> Dict[int, np.ndarray]:
    """
    Extract activation statistics from each layer of a transformer model.
    
    Returns:
        Dictionary mapping layer_idx -> activation statistics
    """
    model.eval()
    model.to(device)
    
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    layer_activations = {}
    hooks = []
    
    def create_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # Extract hidden states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # Compute statistics: mean, std, L2 norm per neuron
            act = hidden.detach().cpu().numpy()
            layer_activations[layer_idx] = act.reshape(-1)  # Flatten
        return hook_fn
    
    # Register hooks on transformer layers  
    if hasattr(model, 'h'):  # GPT-2 style (GPT2Model)
        layers = model.h
    elif hasattr(model, 'bert'):  # BERT style
        layers = model.bert.encoder.layer
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):  # BERT-like
        layers = model.encoder.layer
    elif hasattr(model, 'transformer'):  # DistilBERT, GPT-2 older
        if hasattr(model.transformer, 'layer'):  # DistilBERT
            layers = model.transformer.layer
        elif hasattr(model.transformer, 'h'):  # GPT-2 older style
            layers = model.transformer.h
        else:
            raise ValueError(f"Unsupported transformer architecture: {type(model)}")
    elif hasattr(model, 'distilbert'):  # DistilBERT (alt naming)
        layers = model.distilbert.transformer.layer
    elif hasattr(model, 'model'):  # Some other architectures
        if hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model.model, 'encoder'):
            layers = model.model.encoder.layer
        else:
            raise ValueError(f"Unsupported model architecture: {type(model)}")
    else:
        raise ValueError(f"Unsupported model architecture: {type(model)}, available attributes: {[a for a in dir(model) if not a.startswith('_')][:20]}")
    
    for idx, layer in enumerate(layers):
        hook = layer.register_forward_hook(create_hook(idx))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return layer_activations


def compute_layer_causal_impact(
    baseline_activations: Dict[int, np.ndarray],
    test_activations: Dict[int, np.ndarray],
    method: str = 'kl'
) -> Dict[int, float]:
    """
    Compute causal impact score for each layer.
    
    Args:
        baseline_activations: Activations from benign prompt
        test_activations: Activations from test/adversarial prompt
        method: 'kl' (KL divergence), 'js' (Jensen-Shannon), 'l2' (Euclidean)
    
    Returns:
        Dictionary mapping layer_idx -> impact score
    """
    impact_scores = {}
    
    for layer_idx in baseline_activations.keys():
        if layer_idx not in test_activations:
            continue
        
        baseline = baseline_activations[layer_idx]
        test = test_activations[layer_idx]
        
        # Ensure same dimensionality
        min_len = min(len(baseline), len(test))
        baseline = baseline[:min_len]
        test = test[:min_len]
        
        if method == 'kl':
            # Convert to histograms for KL divergence
            baseline = baseline[np.isfinite(baseline)]
            test = test[np.isfinite(test)]
            baseline_hist, _ = np.histogram(baseline, bins=50)
            test_hist, _ = np.histogram(test, bins=50)
            score = compute_kl_divergence(baseline_hist, test_hist)
        
        elif method == 'js':
            baseline = baseline[np.isfinite(baseline)]
            test = test[np.isfinite(test)]
            baseline_hist, _ = np.histogram(baseline, bins=50)
            test_hist, _ = np.histogram(test, bins=50)
            score = compute_js_divergence(baseline_hist, test_hist)
        
        elif method == 'l2':
            score = np.linalg.norm(baseline - test) / np.sqrt(len(baseline))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        impact_scores[layer_idx] = score
    
    return impact_scores


def identify_hot_layers(
    impact_scores: Dict[int, float],
    percentile: float = 95.0
) -> List[int]:
    """
    Identify "hot" layers that exceed the specified percentile threshold.
    
    Based on research showing localized mechanisms (1-2% of neurons).
    
    Args:
        impact_scores: Layer impact scores
        percentile: Threshold percentile (default 95th)
    
    Returns:
        List of layer indices exceeding threshold
    """
    scores = list(impact_scores.values())
    threshold = np.percentile(scores, percentile)
    
    hot_layers = [
        idx for idx, score in impact_scores.items()
        if score >= threshold
    ]
    
    return hot_layers


def create_causal_impact_visualization(
    impact_scores: Dict[int, float],
    hot_layers: List[int],
    title: str = "Layer-Level Causal Impact (hot layers in red)",
    percentile: float = 95.0,
    method: str = 'kl',
    interactive: bool = False
) -> Tuple[go.Figure, str]:
    """
    Create bar chart visualization of layer-level causal impact.
    
    Args:
        impact_scores: Dictionary of layer_idx -> impact score
        hot_layers: List of hot layer indices
        title: Plot title
        percentile: Percentile threshold used
        method: Divergence method used
        interactive: Whether to create interactive HTML
    
    Returns:
        Plotly figure and HTML string (if interactive=True)
    """
    layer_indices = sorted(impact_scores.keys())
    scores = [impact_scores[idx] for idx in layer_indices]
    
    # Determine colors: red for hot layers, blue for normal
    colors = [
        'rgba(231, 107, 102, 0.8)' if idx in hot_layers
        else 'rgba(100, 149, 237, 0.8)'
        for idx in layer_indices
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=layer_indices,
        y=scores,
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        hovertemplate=(
            '<b>Layer %{x}</b><br>' +
            'Impact: %{y:.2e}<br>' +
            '<extra></extra>'
        ),
        showlegend=False
    ))
    
    # Add threshold line
    if len(scores) > 0:
        threshold = np.percentile(scores, percentile)
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="rgba(0,0,0,0.5)",
            annotation_text=f"{percentile}th pct.",
            annotation_position="right"
        )
    
    # Update layout
    method_label = {
        'kl': 'ΔKL',
        'js': 'JS divergence',
        'l2': 'L2 distance'
    }.get(method, method)
    
    # Prepend NeurInSpectre to title if not already there
    if not title.startswith('NeurInSpectre'):
        title = f'NeurInSpectre — {title}'
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Layer index',
            tickmode='linear',
            tick0=0,
            dtick=1,
            gridcolor='rgba(200,200,200,0.3)'
        ),
        yaxis=dict(
            title=f'Mean {method_label}',
            gridcolor='rgba(200,200,200,0.3)'
        ),
        plot_bgcolor='rgba(240,240,245,0.8)',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=60, r=60, t=80, b=180),
        height=500,
        hovermode='x unified'
    )
    
    # Add comprehensive annotations with red/blue team guidance
    hot_layer_str = ', '.join([str(layer_idx) for layer_idx in hot_layers]) if hot_layers else 'None'
    hot_pct = (len(hot_layers) / len(impact_scores) * 100) if impact_scores else 0
    
    # Build actionable guidance based on findings
    if len(hot_layers) == 0:
        status = "✅ No Anomalies"
        action = "Continue monitoring"
    elif len(hot_layers) == 1:
        status = "⚠️ Single Hot Layer"
        layer_pos = "early" if hot_layers[0] < len(impact_scores) // 3 else ("middle" if hot_layers[0] < 2 * len(impact_scores) // 3 else "late")
        action = f"Investigate {layer_pos}-layer activity"
    else:
        status = "🚨 Multiple Hot Layers"
        action = "Immediate analysis required"
    
    red_blue_text = (
        f"<b>Status:</b> {status} | <b>Hot Layers:</b> [{hot_layer_str}] ({hot_pct:.1f}% of total)<br>"
        f"<b>🔴 Red Team:</b> Target 85-94th percentile layers for stealth; exploit hot layers for max impact<br>"
        f"<b>🔵 Blue Team:</b> {action} | Apply activation clipping/pruning to hot layers<br>"
        f"<b>Threshold:</b> ≥{percentile}th percentile (research: 1-2% of neurons causally relevant)"
    )
    
    fig.add_annotation(
        text=red_blue_text,
        xref='paper',
        yref='paper',
        x=0.5,
        y=-0.22,
        xanchor='center',
        yanchor='top',
        showarrow=False,
        font=dict(size=10),
        align='left',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='rgba(0,0,0,0.2)',
        borderwidth=1,
        borderpad=8
    )
    
    html_str = ""
    if interactive:
        html_str = fig.to_html(include_plotlyjs='cdn', full_html=True)
    
    return fig, html_str


def analyze_layer_causal_impact(
    model: Any,
    tokenizer: Any,
    baseline_prompt: str,
    test_prompt: str,
    device: str = 'cpu',
    method: str = 'kl',
    percentile: float = 95.0,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None
) -> Tuple[Dict[int, float], List[int]]:
    """
    Full pipeline: Extract activations and compute causal impact.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer
        baseline_prompt: Benign/reference prompt
        test_prompt: Test/adversarial prompt
        device: 'cpu', 'cuda', or 'mps'
        method: Divergence method ('kl', 'js', 'l2')
        percentile: Threshold for hot layers
        layer_start: Optional starting layer index
        layer_end: Optional ending layer index
    
    Returns:
        (impact_scores, hot_layers)
    """
    logger.info("📊 Analyzing layer-level causal impact...")
    logger.info(f"   Baseline: {baseline_prompt[:50]}...")
    logger.info(f"   Test: {test_prompt[:50]}...")
    
    # Extract activations
    baseline_act = extract_layer_activations(model, tokenizer, baseline_prompt, device)
    test_act = extract_layer_activations(model, tokenizer, test_prompt, device)
    
    # Filter layers if specified
    if layer_start is not None or layer_end is not None:
        start = layer_start if layer_start is not None else 0
        end = layer_end if layer_end is not None else max(baseline_act.keys())
        
        baseline_act = {k: v for k, v in baseline_act.items() if start <= k <= end}
        test_act = {k: v for k, v in test_act.items() if start <= k <= end}
    
    # Compute impact scores
    impact_scores = compute_layer_causal_impact(baseline_act, test_act, method=method)
    
    # Identify hot layers
    hot_layers = identify_hot_layers(impact_scores, percentile=percentile)
    
    logger.info(f"✅ Analysis complete: {len(impact_scores)} layers analyzed")
    logger.info(f"🔥 Hot layers (≥{percentile}th percentile): {hot_layers}")
    
    return impact_scores, hot_layers
