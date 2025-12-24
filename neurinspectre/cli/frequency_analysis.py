#!/usr/bin/env python3
"""
NeurInSpectre Frequency Adversarial Analysis
Advanced spectral analysis for gradient leakage detection
"""

import logging
import json
from pathlib import Path
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)
"""
Note: This module previously contained two definitions of run_frequency_adversarial.
The implementation has been consolidated below; duplicate header/import blocks removed.
"""

def _get_matplotlib():
    """Lazy import matplotlib"""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for frequency analysis. Install with: pip install matplotlib"
        ) from e

def _get_scipy():
    """Lazy import scipy"""
    try:
        from scipy import signal, fft
        return signal, fft
    except ImportError as e:
        raise ImportError(
            "scipy is required for frequency analysis. Install with: pip install scipy"
        ) from e

def _strip_emoji(text: str) -> str:
    """Remove emoji/symbols that some Matplotlib fonts render as blank boxes.
    Falls back to returning the original text on failure.
    """
    try:
        return text.encode('ascii', 'ignore').decode('ascii')
    except Exception:
        return text

def analyze_gradient_spectrum(gradient_data, threshold=0.75):
    """Analyze gradient spectrum for adversarial patterns"""
    signal_mod, fft_mod = _get_scipy()
    
    # Flatten gradient data if multidimensional
    if len(gradient_data.shape) > 1:
        grad_flat = gradient_data.reshape(gradient_data.shape[0], -1)
    else:
        grad_flat = gradient_data.reshape(-1, 1)
    
    results = {
        'adversarial_indicators': [],
        'spectral_anomalies': [],
        'frequency_peaks': [],
        'vulnerability_score': 0.0,
        'red_team_insights': [],
        'blue_team_recommendations': []
    }
    
    # Analyze each gradient dimension (limit for performance)
    for i in range(min(grad_flat.shape[1], 20)):
        grad_series = grad_flat[:, i]
        
        # Compute FFT
        fft_vals = fft_mod.fft(grad_series)
        freqs = fft_mod.fftfreq(len(grad_series))
        power_spectrum = np.abs(fft_vals)**2
        
        # Detect high-frequency anomalies
        high_freq_power = np.sum(power_spectrum[len(power_spectrum)//4:])
        total_power = np.sum(power_spectrum)
        high_freq_ratio = high_freq_power / total_power if total_power > 0 else 0
        # Add spectral flatness and crest factor
        ps = power_spectrum + 1e-12
        flatness = float(np.exp(np.mean(np.log(ps))) / np.mean(ps))
        crest = float(np.max(np.abs(grad_series)) / (np.sqrt(np.mean(grad_series**2)) + 1e-12))
        
        if high_freq_ratio > threshold or flatness < 0.3 or crest > 5.0:
            results['adversarial_indicators'].append({
                'dimension': i,
                'high_freq_ratio': float(high_freq_ratio),
                'spectral_flatness': flatness,
                'crest_factor': crest,
                'severity': 'CRITICAL' if high_freq_ratio > 0.9 else 'HIGH'
            })
        
        # Find dominant frequencies
        peak_indices, _ = signal_mod.find_peaks(power_spectrum, height=np.max(power_spectrum)*0.1)
        if len(peak_indices) > 0:
            dominant_freqs = freqs[peak_indices]
            peak_powers = power_spectrum[peak_indices]
            
            for freq, power in zip(dominant_freqs, peak_powers):
                if freq > 0.1:  # High frequency peaks are suspicious
                    results['frequency_peaks'].append({
                        'frequency': float(freq),
                        'power': float(power),
                        'dimension': i,
                        'risk_level': 'HIGH' if freq > 0.3 else 'MEDIUM'
                    })
    
    # Calculate composite vulnerability score
    num_anomalies = len(results['adversarial_indicators'])
    num_peaks = len(results['frequency_peaks'])
    max_ratio = max([x['high_freq_ratio'] for x in results['adversarial_indicators']], default=0)
    min_flat = min([x.get('spectral_flatness',1.0) for x in results['adversarial_indicators']], default=1.0)
    max_crest = max([x.get('crest_factor',0.0) for x in results['adversarial_indicators']], default=0.0)
    
    results['vulnerability_score'] = min(1.0,
        0.25 * (num_anomalies / max(1, min(grad_flat.shape[1], 20))) +
        0.25 * max_ratio +
        0.25 * (1.0 - min(1.0, max(0.0, min_flat))) +
        0.25 * min(1.0, max_crest / 10.0)
    )
    
    # Generate operator guidance (heuristics; validate on your environment)
    if results['vulnerability_score'] > 0.8:
        results['red_team_insights'].append(
            f"⚠️ HIGH: Vulnerability={results['vulnerability_score']:.2f} → Elevated high-frequency energy / low flatness / high crest factor"
        )
        results['red_team_insights'].append(
            f"🔎 Signal: {num_anomalies} anomalous dimensions flagged (inspect these channels first)"
        )
        results['red_team_insights'].append(
            "🧪 Validation: compare against a clean baseline and run your approved leakage benchmark before concluding exploitability"
        )
        results['blue_team_recommendations'].append(
            f"🚨 URGENT: {num_anomalies} critical dimensions → Apply gradient clipping (max_norm=1.0)"
        )
        results['blue_team_recommendations'].append(
            "🛡️ DEFENSE: Inject DP noise σ≥2.0 | Implement spectral filtering on high-freq bands"
        )
        results['blue_team_recommendations'].append(
            f"⚠️ MONITOR: Deploy real-time spectral anomaly detection | Alert threshold={threshold}"
        )
    elif results['vulnerability_score'] > 0.5:
        results['red_team_insights'].append(
            f"🟡 MODERATE: Vulnerability={results['vulnerability_score']:.2f} → Some spectral anomalies present"
        )
        results['red_team_insights'].append(
            f"📊 Observation: {num_peaks} notable frequency peaks (may correlate with structured artifacts; validate in context)"
        )
        results['blue_team_recommendations'].append(
            f"⚠️ ENHANCE: Add gradient noise σ≥1.0 | Consider gradient masking on {num_anomalies} dims"
        )
        results['blue_team_recommendations'].append(
            "🛡️ PRIVACY: Implement differential privacy ε≤1.0 | Gradient quantization (8-bit)"
        )
    else:
        results['red_team_insights'].append(
            f"⏸️ LOW: Vulnerability={results['vulnerability_score']:.2f} → Spectral profile normal"
        )
        results['blue_team_recommendations'].append(
            "✅ SECURE: Spectral profile normal | Maintain baseline DP ε=1.0"
        )
    
    return results

def create_frequency_visualization(gradient_data, results, output_file):
    """Create INTERACTIVE Plotly frequency analysis visualization with hover guidance"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("plotly required. Install with: pip install plotly")
        return
    
    signal_mod, fft_mod = _get_scipy()
    
    # Flatten gradient data
    if len(gradient_data.shape) > 1:
        grad_flat = gradient_data.reshape(gradient_data.shape[0], -1)
    else:
        grad_flat = gradient_data.reshape(-1, 1)
    
    # Compute FFT for power spectrum
    grad_sample = grad_flat[:, 0] if grad_flat.shape[1] > 0 else grad_flat.flatten()
    fft_vals = fft_mod.fft(grad_sample)
    freqs = fft_mod.fftfreq(len(grad_sample))
    power_spectrum = np.abs(fft_vals)**2
    pos_freqs = freqs[:len(freqs)//2]
    pos_power = power_spectrum[:len(power_spectrum)//2]
    
    # Create 2x2 subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '📊 Gradient Time Series (Sample)',
            '🎯 Power Spectrum Analysis',
            '⚠️ Dimensional Vulnerability Metrics',
            '📋 Security Summary'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'bar'}, {'type': 'table'}]],  # Panel 4 is now table type
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )
    
    # Panel 1: Time Series with hover
    sample_dim = min(3, grad_flat.shape[1])
    for i in range(sample_dim):
        threat_category = "HIGH" if np.max(np.abs(grad_flat[:100, i])) > 1.0 else "NORMAL"
        variance = np.var(grad_flat[:100, i])
        
        fig.add_trace(go.Scatter(
            x=list(range(100)),
            y=grad_flat[:100, i],
            mode='lines',
            name=f'Dim {i+1}',
            line=dict(width=2),
            hovertemplate=(
                f'<b>Dimension {i+1}</b><br>' +
                'Step: %{x}<br>' +
                'Value: %{y:.4f}<br>' +
                f'Variance: {variance:.4f}<br>' +
                f'Threat: {threat_category}<br>' +
                '<b>🔴 Red Team:</b> High variance = MI target<br>' +
                '<b>🔵 Blue Team:</b> Clip spikes & add noise<br>' +
                '<extra></extra>'
            )
        ), row=1, col=1)
    
    # Panel 2: Power Spectrum with peaks
    peaks_pos = [p for p in results['frequency_peaks'] if p['frequency'] > 0]
    peaks_sorted = sorted(peaks_pos, key=lambda x: x.get('power', 0.0), reverse=True)[:3]
    
    fig.add_trace(go.Scatter(
        x=pos_freqs,
        y=pos_power,
        mode='lines',
        name='Power Spectrum',
        line=dict(color='#3498DB', width=2),
        fill='tonexty',
        fillcolor='rgba(52, 152, 219, 0.2)',
        hovertemplate=(
            '<b>Power Spectrum</b><br>' +
            'Frequency: %{x:.4f}<br>' +
            'Power: %{y:.2e}<br>' +
            '<b>🔴 Red:</b> High peaks = obfuscation signatures<br>' +
            '<b>🔵 Blue:</b> Filter high-freq bands<br>' +
            '<extra></extra>'
        )
    ), row=1, col=2)
    
    # Add suspect band and peak markers
    if peaks_pos:
        fmin, fmax = min(p['frequency'] for p in peaks_pos), max(p['frequency'] for p in peaks_pos)
        fig.add_vrect(x0=fmin, x1=fmax, fillcolor="red", opacity=0.15, layer="below", line_width=0, row=1, col=2)
        
        for pk in peaks_sorted:
            fig.add_trace(go.Scatter(
                x=[pk['frequency']], y=[pk['power']],
                mode='markers', marker=dict(size=10, color='red', symbol='x'),
                showlegend=False,
                hovertemplate=(
                    f"<b>Peak</b><br>f={pk['frequency']:.4f}<br>Power={pk['power']:.2e}<br>" +
                    f"Risk: {pk['risk_level']}<br>" +
                    '<b>🔴 Red:</b> Exploit peak<br><b>🔵 Blue:</b> Filter here<br><extra></extra>'
                )
            ), row=1, col=2)
    
    # Panel 3: Vulnerability Bars (ALL dimensions analyzed, scrollable via zoom)
    if results['adversarial_indicators']:
        # Sort by dimension for better visualization
        sorted_indicators = sorted(results['adversarial_indicators'], key=lambda x: x['dimension'])
        
        dims = [x['dimension'] for x in sorted_indicators]
        ratios = [x['high_freq_ratio'] for x in sorted_indicators]
        flats = [1.0 - x.get('spectral_flatness', 1.0) for x in sorted_indicators]
        crests = [min(1.0, x.get('crest_factor', 0.0)/10.0) for x in sorted_indicators]
        
        # Create grouped bars for ALL dimensions
        fig.add_trace(go.Bar(
            x=dims, y=ratios, name='High-Freq Ratio', 
            marker_color='#ff9f80',
            hovertemplate=(
                'Dimension: %{x}<br>High-Freq Ratio: %{y:.3f}<br>' +
                '<b>🔴 Red Team:</b> High ratio enables data extraction<br>' +
                '<b>🔵 Blue Team:</b> Apply low-pass filtering<br>' +
                '<extra></extra>'
            )
        ), row=2, col=1)
        
        fig.add_trace(go.Bar(
            x=dims, y=flats, name='(1-Flatness)', 
            marker_color='#80bfff',
            hovertemplate=(
                'Dimension: %{x}<br>(1-Flatness): %{y:.3f}<br>' +
                '<b>🔴 Red Team:</b> Non-flat = structured attack signal<br>' +
                '<b>🔵 Blue Team:</b> Add spectral noise<br>' +
                '<extra></extra>'
            )
        ), row=2, col=1)
        
        fig.add_trace(go.Bar(
            x=dims, y=crests, name='Crest/10', 
            marker_color='#b380ff',
            hovertemplate=(
                'Dimension: %{x}<br>Crest Factor/10: %{y:.3f}<br>' +
                '<b>🔴 Red Team:</b> High crest = membership inference opportunity<br>' +
                '<b>🔵 Blue Team:</b> Gradient clipping required<br>' +
                '<extra></extra>'
            )
        ), row=2, col=1)
        
        # Add critical threshold line
        fig.add_hline(y=0.75, line_dash="dash", line_color="red", line_width=2, 
                     annotation_text="Critical Threshold (0.75)", annotation_position="right",
                     row=2, col=1)
        
        logger.info(f"📊 Displaying vulnerability metrics for {len(dims)} dimensions (scrollable)")
    else:
        # If no anomalies, show message
        fig.add_annotation(
            text="✅ No Significant Anomalies Detected",
            xref="x3", yref="y3",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color='white')
        )
    
    # Panel 4: Security Summary with elegant table format
    summary_data = {
        'Metric': ['Vulnerability Score', 'Anomalies Detected', 'Frequency Peaks', 'Analysis Time'],
        'Value': [
            f"{results['vulnerability_score']:.2f}",
            str(len(results['adversarial_indicators'])),
            str(len(results['frequency_peaks'])),
            datetime.now().strftime('%H:%M:%S')
        ]
    }
    
    fig.add_trace(go.Table(
        header=dict(values=['<b>Metric</b>', '<b>Value</b>'],
                   fill_color='rgba(128,128,128,0.5)',
                   align='left',
                   font=dict(color='white', size=12)),
        cells=dict(values=[summary_data['Metric'], summary_data['Value']],
                  fill_color='rgba(50,50,50,0.5)',
                  align='left',
                  font=dict(color='white', size=11),
                  height=30)
    ), row=2, col=2)
    
    # Add Red/Blue annotations BELOW the Security Summary table (non-overlapping)
    red_text = "<br>".join(results['red_team_insights'][:3])
    blue_text = "<br>".join(results['blue_team_recommendations'][:3])
    
    fig.add_annotation(
        text=f"<b>🔴 RED TEAM GUIDANCE</b><br>{red_text}", 
        xref="paper", yref="paper",
        x=0.75, y=-0.18,  # Positioned below bottom-right panel
        showarrow=False, 
        font=dict(size=11, color='white'), 
        align="left",
        bgcolor='rgba(204,0,0,0.85)', 
        bordercolor='#cc0000', 
        borderwidth=2, 
        borderpad=12,
        xanchor='center',
        width=800
    )
    
    fig.add_annotation(
        text=f"<b>🔵 BLUE TEAM DEFENSE</b><br>{blue_text}", 
        xref="paper", yref="paper",
        x=0.25, y=-0.18,  # Positioned below bottom-left panel
        showarrow=False, 
        font=dict(size=11, color='white'), 
        align="left",
        bgcolor='rgba(31,95,191,0.85)', 
        bordercolor='#1f5fbf', 
        borderwidth=2, 
        borderpad=12,
        xanchor='center',
        width=800
    )
    
    # Layout with increased bottom margin for guidance boxes
    fig.update_layout(
        title=dict(text='⚡ NeurInSpectre Frequency Domain Security Analysis', x=0.5, xanchor='center',
                  font=dict(size=20, color='white')),
        height=1250, width=1800, template='plotly_dark', hovermode='closest', uirevision='constant',
        showlegend=True, legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.7)', font=dict(color='white')),
        plot_bgcolor='rgba(20,20,20,0.95)', paper_bgcolor='rgba(10,10,10,1)',
        margin=dict(l=80, r=80, t=100, b=280)  # Increased bottom margin for guidance boxes
    )
    
    # Update axes
    fig.update_xaxes(title_text="Training Step", row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Gradient Value", row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_xaxes(title_text="Frequency", row=1, col=2, gridcolor='rgba(128,128,128,0.2)', type='log')
    fig.update_yaxes(title_text="Power (log scale)", row=1, col=2, gridcolor='rgba(128,128,128,0.2)', type='log')
    fig.update_xaxes(title_text="Gradient Dimension", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Scaled Metric (0-1)", row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
    
    # Save interactive HTML
    html_file = output_file.replace('.png', '_interactive.html')
    fig.write_html(html_file)
    logger.info(f"📊 Interactive HTML saved: {html_file}")
    logger.info("🔍 Features: Zoom, Pan, Hover tooltips with Red/Blue guidance")
    
    # Save static PNG (simplified matplotlib version)
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        with mpl.rc_context({'font.size': 10}):
            fig_mpl, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
            fig_mpl.suptitle('NeurInSpectre — Frequency Domain Security Analysis', fontsize=14, fontweight='bold')
            
            # Simplified static plots
            for i in range(sample_dim):
                axes[0, 0].plot(grad_flat[:100, i], label=f'Dim {i+1}', alpha=0.7)
            axes[0, 0].set_title('Gradient Time Series'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].semilogy(pos_freqs, pos_power, 'b-', linewidth=2)
            axes[0, 1].set_title('Power Spectrum'); axes[0, 1].grid(True, alpha=0.3)
            
            if results['adversarial_indicators']:
                dims = [x['dimension'] for x in results['adversarial_indicators']]
                ratios = [x['high_freq_ratio'] for x in results['adversarial_indicators']]
                axes[1, 0].bar(dims, ratios, color='#ff9f80')
                axes[1, 0].axhline(0.75, color='red', ls='--')
                axes[1, 0].set_title('Vulnerability Metrics')
            
            axes[1, 1].axis('off')
            axes[1, 1].text(0.5, 0.5, f"Vulnerability: {results['vulnerability_score']:.2f}\n" +
                           f"Anomalies: {len(results['adversarial_indicators'])}", 
                           ha='center', va='center', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        logger.info(f"📊 Static PNG saved: {output_file}")
    except Exception as e:
        logger.warning(f"Static PNG generation failed: {e}")

def run_frequency_adversarial(args):
    """Run enhanced frequency adversarial analysis"""
    logger.info("🎯 Starting NeurInSpectre Frequency Domain Analysis...")
    logger.info(f"📊 Input spectrum: {args.input_spectrum}")
    logger.info(f"📈 Detection threshold: {args.threshold}")
    
    # Check if input file exists
    input_path = Path(args.input_spectrum)
    if not input_path.exists():
        logger.error(f"❌ Input file not found: {args.input_spectrum}")
        return 1
    
    file_size_mb = input_path.stat().st_size / (1024*1024)
    logger.info(f"✅ Loading data: {input_path.name} ({file_size_mb:.1f}MB)")
    
    try:
        # Load gradient data
        gradient_data = np.load(input_path)
        logger.info(f"📊 Data shape: {gradient_data.shape}")
        
        # Run spectral analysis
        logger.info("🔍 Running spectral analysis...")
        results = analyze_gradient_spectrum(gradient_data, args.threshold)
        
        # Log key findings
        logger.info(f"🎯 Vulnerability Score: {results['vulnerability_score']:.2f}")
        logger.info(f"⚠️ Anomalies Detected: {len(results['adversarial_indicators'])}")
        logger.info(f"📊 Frequency Peaks: {len(results['frequency_peaks'])}")
        
        # Create visualization (BOTH interactive HTML and static PNG)
        if args.output_plot:
            logger.info("🎨 Generating interactive visualization...")
            create_frequency_visualization(gradient_data, results, args.output_plot)
            logger.info(f"📊 View interactive: open {args.output_plot.replace('.png', '_interactive.html')}")
        
        # Save metrics
        if args.save_metrics:
            logger.info(f"💾 Saving metrics to: {args.save_metrics}")
            with open(args.save_metrics, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Print security summary
        if results['vulnerability_score'] > 0.8:
            logger.warning("🚨 CRITICAL: High vulnerability detected!")
        elif results['vulnerability_score'] > 0.5:
            logger.warning("⚠️ MODERATE: Some vulnerabilities detected")
        else:
            logger.info("✅ LOW: Spectral profile appears normal")
        
        logger.info("✅ Frequency adversarial analysis complete")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return 1 