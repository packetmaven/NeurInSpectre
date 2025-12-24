#!/usr/bin/env python3
"""
Obfuscated Gradient Visualizer for NeurInSpectre
Comprehensive visualization tools for gradient obfuscation analysis
Integrates with advanced_mathematical_foundations.py and dashboard systems
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager, nullcontext
from typing import Dict, Tuple, Any

class ObfuscatedGradientVisualizer:
    """
    Comprehensive visualization system for obfuscated gradient analysis
    Integrates with NeurInSpectre mathematical foundations
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        self.figsize = figsize

        self.colors = {
            'clean': '#2A9D8F',       # teal
            'obfuscated': '#9B59B6',  # violet (amethyst)
            'suspicious': '#F4A261',  # sand
            'background': '#0f172a',  # dark slate
            'text': '#E2E8F0',        # light slate
            'accent': '#1F5FBF'       # cobalt
        }

        # Per-instance plotting style. Applied via context managers at render time
        # to avoid process-global matplotlib/seaborn state mutation.
        self._mpl_style = 'seaborn-v0_8-darkgrid'
        self._sns_palette = [
            "#2A9D8F",  # teal
            "#9B59B6",  # violet
            "#F4A261",  # sand
            "#264653",  # deep blue-green
            "#1F5FBF",  # cobalt
        ]
        self._mpl_rc = {
            'figure.facecolor': self.colors['background'],
            'axes.facecolor': self.colors['background'],
            'text.color': self.colors['text'],
            'axes.labelcolor': self.colors['text'],
            'xtick.color': self.colors['text'],
            'ytick.color': self.colors['text'],
            'legend.facecolor': self.colors['background'],
            'legend.edgecolor': self.colors['text'],
            'axes.titleweight': 'bold',
        }

    @contextmanager
    def _plot_context(self):
        """Context manager to apply style/palette without mutating global state."""
        style_ctx = nullcontext()
        try:
            style_ctx = plt.style.context(self._mpl_style)
        except Exception:
            pass

        palette_ctx = nullcontext()
        try:
            palette_ctx = sns.color_palette(self._sns_palette)
        except Exception:
            pass

        with style_ctx, mpl.rc_context(rc=self._mpl_rc), palette_ctx:
            yield
    
    def load_sample_gradients(self) -> Dict[str, np.ndarray]:
        """Generate sample gradient data for visualization demos (in-memory only).

        Note: This does NOT read/write any on-disk sample files. CLI workflows that
        analyze real data should pass explicit input files.
        """
        return {
            "clean": self._generate_clean_gradients(),
            "obfuscated": self._generate_obfuscated_gradients(),
        }
    
    def _generate_clean_gradients(self, n_samples: int = 512) -> np.ndarray:
        """Generate clean gradient data"""
        return np.random.normal(0, 0.1, n_samples)
    
    def _generate_obfuscated_gradients(self, n_samples: int = 512) -> np.ndarray:
        """Generate obfuscated gradient data with attack patterns"""
        base_gradients = self._generate_clean_gradients(n_samples)
        
        # Add obfuscation patterns
        obfuscated = base_gradients.copy()
        
        # 1. Periodic obfuscation (common in adversarial attacks)
        obfuscated += 0.05 * np.sin(np.arange(n_samples) * 0.1)
        
        # 2. Random spikes (gradient masking)
        spike_indices = np.random.choice(n_samples, size=n_samples//10, replace=False)
        obfuscated[spike_indices] += 0.3 * np.random.randn(len(spike_indices))
        
        # 3. Systematic bias (model poisoning)
        obfuscated += 0.02 * np.linspace(-1, 1, n_samples)
        
        # 4. High-frequency noise (evasion technique)
        obfuscated += 0.01 * np.random.randn(n_samples) * np.sin(np.arange(n_samples) * 0.5)
        
        return obfuscated
    
    def create_comprehensive_visualization(self, gradients: Dict[str, np.ndarray], 
                                         save_path: str = "gradient_analysis_dashboard.png") -> None:
        """Create comprehensive gradient analysis visualization - STATIC AND INTERACTIVE"""
        
        # Create STATIC matplotlib version (original)
        with self._plot_context():
            fig = plt.figure(figsize=self.figsize)
            fig.suptitle('GRADIENT OBFUSCATION ANALYSIS DASHBOARD', 
                         fontsize=20, fontweight='bold', y=0.98)
            
            # Create 2x3 grid layout
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Panel 1: Gradient Comparison (top-left)
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_gradient_comparison(ax1, gradients)
            
            # Panel 2: Spectral Analysis (top-middle)
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_spectral_analysis(ax2, gradients)
            
            # Panel 3: Statistical Distribution (top-right)
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_statistical_distribution(ax3, gradients)
            
            # Panel 4: Obfuscation Patterns (bottom-left)
            ax4 = fig.add_subplot(gs[1, 0])
            self._plot_obfuscation_patterns(ax4, gradients)
            
            # Panel 5: Threat Assessment (bottom-middle)
            ax5 = fig.add_subplot(gs[1, 1])
            self._plot_threat_assessment(ax5, gradients)
            
            # Panel 6: Real-time Monitoring (bottom-right)
            ax6 = fig.add_subplot(gs[1, 2])
            self._plot_realtime_monitoring(ax6, gradients)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.colors['background'])
        
        print(f"🎯 Static PNG saved to: {save_path}")
        
        # Create INTERACTIVE Plotly version for real-time debugging
        html_path = save_path.replace('.png', '_interactive.html')
        self._create_interactive_plotly_dashboard(gradients, html_path)
        print(f"🚀 Interactive HTML saved to: {html_path}")
    
    def _plot_gradient_comparison(self, ax, gradients: Dict[str, np.ndarray]):
        """Plot gradient comparison"""
        ax.set_title('GRADIENT COMPARISON', fontsize=14, fontweight='bold')
        
        # Define a small color cycle to avoid single-color multi-series errors
        series_palette = [self.colors['clean'], self.colors['obfuscated'], self.colors['accent'], '#F1C40F']
        
        for series_index, (grad_type, grad_data) in enumerate(gradients.items()):
            # Normalize input to 1D for plotting
            if isinstance(grad_data, np.ndarray) and grad_data.ndim == 2:
                # Reduce 2D [T, D] to a single representative 1D series (mean across features)
                grad_1d = np.mean(grad_data, axis=1)
            else:
                grad_1d = np.asarray(grad_data).reshape(-1)
            
            color = series_palette[series_index % len(series_palette)]
            
            # Plot first 200 points for clarity
            x_values = np.arange(min(200, len(grad_1d)))
            ax.plot(x_values, grad_1d[:200], 
                   color=color, alpha=0.8, linewidth=2,
                   label=f'{grad_type.title()}')
        
        ax.set_xlabel('Gradient Index')
        ax.set_ylabel('Gradient Magnitude')
        # Only create legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_spectral_analysis(self, ax, gradients: Dict[str, np.ndarray]):
        """Plot spectral analysis of gradients"""
        ax.set_title('SPECTRAL ANALYSIS', fontsize=14, fontweight='bold')
        
        series_palette = [self.colors['clean'], self.colors['obfuscated'], self.colors['accent'], '#F1C40F']
        
        for series_index, (grad_type, grad_data) in enumerate(gradients.items()):
            # Normalize input to 1D for FFT
            if isinstance(grad_data, np.ndarray) and grad_data.ndim == 2:
                grad_1d = np.mean(grad_data, axis=1)
            else:
                grad_1d = np.asarray(grad_data).reshape(-1)
            # Compute FFT
            fft_data = np.fft.fft(grad_1d)
            freqs = np.fft.fftfreq(len(grad_1d))
            magnitude = np.abs(fft_data)
            
            # Plot power spectrum
            color = series_palette[series_index % len(series_palette)]
            ax.semilogy(freqs[:len(freqs)//2], magnitude[:len(magnitude)//2], 
                       color=color, alpha=0.8, linewidth=2,
                       label=f'{grad_type.title()} Spectrum')
        
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power (log scale)')
        # Only create legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_statistical_distribution(self, ax, gradients: Dict[str, np.ndarray]):
        """Plot statistical distribution analysis"""
        ax.set_title('DISTRIBUTION ANALYSIS', fontsize=14, fontweight='bold')
        
        series_palette = [self.colors['clean'], self.colors['obfuscated'], self.colors['accent'], '#F1C40F']
        
        for series_index, (grad_type, grad_data) in enumerate(gradients.items()):
            if isinstance(grad_data, np.ndarray) and grad_data.ndim == 2:
                grad_1d = np.mean(grad_data, axis=1)
            else:
                grad_1d = np.asarray(grad_data).reshape(-1)
            color = series_palette[series_index % len(series_palette)]
            
            # Plot histogram
            ax.hist(grad_1d, bins=50, alpha=0.6, color=color, 
                   label=f'{grad_type.title()} Distribution', density=True)
        
        ax.set_xlabel('Gradient Value')
        ax.set_ylabel('Probability Density')
        # Only create legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_obfuscation_patterns(self, ax, gradients: Dict[str, np.ndarray]):
        """Plot obfuscation pattern detection"""
        ax.set_title('OBFUSCATION PATTERNS', fontsize=14, fontweight='bold')
        # Choose a series to analyze (prefer an explicit obfuscated/observed series).
        key = None
        for k in ("obfuscated", "observed", "loaded", "reference", "clean"):
            if k in gradients:
                key = k
                break
        if key is None:
            ax.set_xlabel('Position')
            ax.set_ylabel('Statistical Measure')
            ax.grid(True, alpha=0.3)
            return

        grad_data = gradients[key]
        # Normalize to 1D for rolling stats
        if isinstance(grad_data, np.ndarray) and grad_data.ndim == 2:
            grad_1d = np.mean(grad_data, axis=1)
        else:
            grad_1d = np.asarray(grad_data).reshape(-1)

        # Detect patterns using rolling statistics
        window_size = 50
        if grad_1d.size >= window_size:
            rolling_mean = np.convolve(grad_1d, np.ones(window_size)/window_size, mode='valid')
            rolling_std = np.array([np.std(grad_1d[i:i+window_size]) 
                                  for i in range(len(grad_1d)-window_size+1)])
            
            # Plot patterns
            x_values = np.arange(len(rolling_mean))
            ax.plot(x_values, rolling_mean, color=self.colors['obfuscated'], 
                   linewidth=2, label=f'Rolling Mean ({key})')
            ax.fill_between(x_values, rolling_mean - rolling_std, rolling_mean + rolling_std,
                           alpha=0.3, color=self.colors['suspicious'], label='±1 Std Dev')
            
            # Highlight anomalies
            anomaly_threshold = np.mean(rolling_std) + 2 * np.std(rolling_std)
            anomalies = np.where(rolling_std > anomaly_threshold)[0]
            
            if len(anomalies) > 0:
                ax.scatter(anomalies, rolling_mean[anomalies], 
                          color='red', s=50, zorder=5, label='Anomalies')
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Statistical Measure')
        # Only create legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_threat_assessment(self, ax, gradients: Dict[str, np.ndarray]):
        """Plot threat assessment metrics"""
        ax.set_title('THREAT ASSESSMENT', fontsize=14, fontweight='bold')
        
        # Calculate threat metrics for each gradient type
        threat_metrics = {}
        
        for grad_type, grad_data in gradients.items():
            # Entropy (higher = more suspicious)
            counts, _ = np.histogram(grad_data, bins=50)
            p = counts.astype(np.float64)
            p_sum = float(np.sum(p))
            if p_sum <= 0.0 or len(p) < 2:
                entropy_norm = 0.0
            else:
                p = p / (p_sum + 1e-12)
                ent_bits = -float(np.sum(p * np.log2(p + 1e-12)))
                entropy_norm = float(ent_bits / np.log2(len(p)))
                entropy_norm = float(np.clip(entropy_norm, 0.0, 1.0))
            
            # Variance (higher = more suspicious)
            variance = np.var(grad_data)
            
            # Skewness (deviation from normal)
            mean_val = np.mean(grad_data)
            std_val = float(np.std(grad_data))
            if std_val <= 0.0 or not np.isfinite(std_val):
                skewness = 0.0
            else:
                skewness = float(np.mean(((grad_data - mean_val) / std_val) ** 3))
            
            # Kurtosis (tail heaviness)
            if std_val <= 0.0 or not np.isfinite(std_val):
                kurtosis = 0.0
            else:
                kurtosis = float(np.mean(((grad_data - mean_val) / std_val) ** 4) - 3.0)
            
            threat_metrics[grad_type] = {
                'entropy': entropy_norm,
                'variance': variance,
                'skewness': abs(skewness),
                'kurtosis': abs(kurtosis)
            }
        
        # Create radar chart
        metrics = ['entropy', 'variance', 'skewness', 'kurtosis']
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        
        for grad_type, metrics_dict in threat_metrics.items():
            values = [metrics_dict[metric] for metric in metrics]
            # Normalize values for visualization
            max_vals = [max(threat_metrics[gt][m] for gt in threat_metrics.keys()) for m in metrics]
            normalized_values = [v/max_v if max_v > 0 else 0 for v, max_v in zip(values, max_vals)]
            
            color = self.colors['clean'] if grad_type == 'clean' else self.colors['obfuscated']
            ax.plot(angles, normalized_values, 'o-', linewidth=2, label=grad_type.title(), color=color)
            ax.fill(angles, normalized_values, alpha=0.25, color=color)
        
        ax.set_xticks(angles)
        ax.set_xticklabels([m.title() for m in metrics])
        ax.set_ylim(0, 1)
        # Only create legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        ax.grid(True)
    
    def _plot_realtime_monitoring(self, ax, gradients: Dict[str, np.ndarray]):
        """Plot a simple, data-driven threat timeline from the provided gradients.

        Notes:
        - This is a *relative* timeline (window index). If you have real timestamps, map them
          externally and pass pre-binned series to this visualizer.
        - We intentionally do not generate synthetic threat curves.
        """
        ax.set_title('REAL-TIME MONITORING', fontsize=14, fontweight='bold')

        # Choose an analysis series (prefer obfuscated/observed when present).
        key = None
        for k in ("obfuscated", "observed", "loaded", "reference", "clean"):
            if k in gradients:
                key = k
                break

        series = None
        if key is not None:
            g = gradients[key]
            if isinstance(g, np.ndarray) and g.ndim == 2:
                series = np.mean(g, axis=1)
            else:
                series = np.asarray(g).reshape(-1)

        if series is None or series.size < 4:
            ax.text(0.5, 0.5, "No input series for monitoring.", ha="center", va="center")
            ax.set_axis_off()
            return

        series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
        window = int(max(5, min(50, series.size // 20 if series.size >= 20 else series.size)))
        window = int(max(2, min(window, series.size)))

        rolling_mean = np.convolve(series, np.ones(window) / float(window), mode='valid')
        if rolling_mean.size == 0:
            ax.text(0.5, 0.5, "Series too short for monitoring window.", ha="center", va="center")
            ax.set_axis_off()
            return

        rolling_std = np.array(
            [np.std(series[max(0, i - window) : min(series.size, i + window)]) for i in range(rolling_mean.size)]
        )

        anomaly_threshold = 3.0 * float(np.std(series)) if series.size >= 2 else 0.0
        mean_anomalies = np.abs(series[: rolling_mean.size] - rolling_mean) > anomaly_threshold
        std_thresh = 2.0 * float(np.mean(rolling_std)) if rolling_std.size else float("inf")
        std_anomalies = rolling_std > std_thresh

        threat_levels = []
        for i in range(rolling_mean.size):
            local_var = float(rolling_std[i] ** 2)
            local_mag = float(abs(rolling_mean[i]))

            variance_component = min(0.4, local_var * 20.0)
            magnitude_component = min(0.3, local_mag * 3.0)
            base_threat = variance_component + magnitude_component

            if bool(mean_anomalies[i]) or bool(std_anomalies[i]):
                base_threat = min(1.0, base_threat + 0.4)

            threat_levels.append(float(np.clip(base_threat, 0.0, 1.0)))

        time_series = np.arange(len(threat_levels))

        # Plot threat level over time (relative window index)
        ax.plot(time_series, threat_levels, color=self.colors['obfuscated'], linewidth=3, label='Threat Level')
        
        # Add threshold lines
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Normal Threshold')
        
        # Fill areas
        ax.fill_between(time_series, 0, threat_levels, 
                       where=np.array(threat_levels) >= 0.7, 
                       color='red', alpha=0.3, label='Critical Zone')
        ax.fill_between(time_series, 0, threat_levels, 
                       where=(np.array(threat_levels) >= 0.5) & (np.array(threat_levels) < 0.7), 
                       color='orange', alpha=0.3, label='Warning Zone')
        
        ax.set_xlabel('Window index (relative)')
        ax.set_ylabel('Threat Level')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _create_interactive_plotly_dashboard(self, gradients: Dict[str, np.ndarray], 
                                            save_path: str = "gradient_analysis_interactive.html") -> None:
        """Create INTERACTIVE Plotly dashboard for real-time debugging"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            print("🔧 Creating interactive Plotly dashboard...")
            
            # Create subplot layout matching the static version - FIXED SPACING
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    '📊 GRADIENT COMPARISON', 
                    '🌊 SPECTRAL ANALYSIS', 
                    '📈 DISTRIBUTION ANALYSIS',
                    '🔍 OBFUSCATION PATTERNS', 
                    '⚠️ THREAT ASSESSMENT', 
                    '⚡ REAL-TIME MONITORING'
                ),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'histogram'}],
                       [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'scatter'}]],
                horizontal_spacing=0.12,  # Increased spacing to prevent overlap
                vertical_spacing=0.15,    # Better vertical spacing
                row_heights=[0.5, 0.5],
                column_widths=[0.33, 0.33, 0.33]
            )
            
            # Panel 1: Gradient Comparison with DETAILED HOVER
            for grad_type, grad_data in gradients.items():
                grad_1d = grad_data.flatten() if grad_data.ndim > 1 else grad_data
                
                # Calculate threat info for each point
                threat_levels = ['🔴 CRITICAL' if abs(g) > 0.3 else '🟠 HIGH' if abs(g) > 0.2 else '🟡 MEDIUM' if abs(g) > 0.1 else '🟢 LOW' for g in grad_1d]
                red_actions = ['EXPLOIT' if abs(g) > 0.3 else 'TARGET' if abs(g) > 0.2 else 'PROBE' if abs(g) > 0.1 else 'SKIP' for g in grad_1d]
                blue_actions = ['URGENT: Clip & mask' if abs(g) > 0.3 else 'Monitor closely' if abs(g) > 0.2 else 'Track feature' if abs(g) > 0.1 else 'Baseline' for g in grad_1d]
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(grad_1d))),
                    y=grad_1d,
                    name=f'{grad_type.title()} Gradients',
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=4),
                    customdata=list(zip(threat_levels, red_actions, blue_actions)),
                    hovertemplate=
                        f'<b>{grad_type.upper()} GRADIENT</b><br>' +
                        '━━━━━━━━━━━━━━━━━━━━━━<br>' +
                        '<b>Index:</b> %{x}<br>' +
                        '<b>Value:</b> %{y:.6f}<br>' +
                        '<b>Threat:</b> %{customdata[0]}<br>' +
                        '<b>🔴 Red Team:</b> %{customdata[1]}<br>' +
                        '<b>🔵 Blue Team:</b> %{customdata[2]}<br>' +
                        '<extra></extra>'
                ), row=1, col=1)
            
            # Panel 2: Spectral Analysis with hover
            for grad_type, grad_data in gradients.items():
                grad_1d = grad_data.flatten() if grad_data.ndim > 1 else grad_data
                fft_result = np.fft.fft(grad_1d)
                frequencies = np.fft.fftfreq(len(grad_1d))
                power = np.abs(fft_result)**2
                
                # Only positive frequencies
                pos_idx = frequencies >= 0
                
                fig.add_trace(go.Scatter(
                    x=frequencies[pos_idx],
                    y=power[pos_idx],
                    name=f'{grad_type.title()} Spectrum',
                    mode='lines',
                    line=dict(width=2),
                    hovertemplate=
                        f'<b>{grad_type.upper()} SPECTRUM</b><br>' +
                        '<b>Frequency:</b> %{x:.4f}<br>' +
                        '<b>Power:</b> %{y:.2e}<br>' +
                        '<extra></extra>'
                ), row=1, col=2)
            
            # Panel 3: Distribution with detailed stats
            for grad_type, grad_data in gradients.items():
                grad_1d = grad_data.flatten() if grad_data.ndim > 1 else grad_data
                
                fig.add_trace(go.Histogram(
                    x=grad_1d,
                    name=f'{grad_type.title()} Distribution',
                    opacity=0.7,
                    nbinsx=50,
                    hovertemplate=
                        f'<b>{grad_type.upper()}</b><br>' +
                        '<b>Value Range:</b> %{x}<br>' +
                        '<b>Count:</b> %{y}<br>' +
                        '<extra></extra>'
                ), row=1, col=3)
            
            # Update subplot titles for clarity
            for i in fig['layout']['annotations']:
                i['font'] = dict(size=13, color='cyan')
            
            # Panel 4: Obfuscation Patterns
            for grad_type, grad_data in gradients.items():
                grad_1d = grad_data.flatten() if grad_data.ndim > 1 else grad_data
                
                # Calculate rolling statistics for obfuscation pattern detection
                window = 20
                rolling_mean = np.convolve(grad_1d, np.ones(window)/window, mode='valid')
                rolling_std = np.array([np.std(grad_1d[max(0,i-window):i+window]) for i in range(len(rolling_mean))])
                
                # Detect anomalies using both mean deviation AND high std (research-based)
                anomaly_threshold = 3 * np.std(grad_1d)
                
                # Anomaly detection: High deviation from mean OR high local std deviation
                mean_anomalies = np.abs(grad_1d[:len(rolling_mean)] - rolling_mean) > anomaly_threshold
                std_anomalies = rolling_std > (2 * np.mean(rolling_std))  # High variance regions
                anomalies_idx = np.where(mean_anomalies | std_anomalies)[0]
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(rolling_mean))),
                    y=rolling_mean,
                    name=f'{grad_type.title()} Rolling Mean',
                    mode='lines',
                    line=dict(width=2),
                    fill='tonexty' if grad_type != 'clean' else None,
                    hovertemplate=
                        f'<b>{grad_type.upper()} PATTERN</b><br>' +
                        '<b>Position:</b> %{x}<br>' +
                        '<b>Mean:</b> %{y:.6f}<br>' +
                        '<extra></extra>'
                ), row=2, col=1)
                
                # Add anomaly markers
                if len(anomalies_idx) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomalies_idx,
                        y=rolling_mean[anomalies_idx],
                        mode='markers',
                        name=f'{grad_type} Anomalies',
                        marker=dict(size=10, color='red', symbol='x'),
                        hovertemplate=
                            '<b>🚨 ANOMALY DETECTED</b><br>' +
                            '<b>Position:</b> %{x}<br>' +
                            '<b>Value:</b> %{y:.6f}<br>' +
                            '<b>🔴 Red Team:</b> High-value target<br>' +
                            '<b>🔵 Blue Team:</b> Investigate immediately<br>' +
                            '<extra></extra>',
                        showlegend=False
                    ), row=2, col=1)
            
            # Panel 5: Threat Assessment (Bar chart)
            threat_levels = {
                'Clean Baseline': 0.2,
                'Noise Injection': 0.5,
                'Periodic Pattern': 0.7,
                'Gradient Masking': 0.9,
                'Combined Attack': 0.95
            }
            
            mitre_techniques = {
                'Clean Baseline': 'Baseline',
                # Obfuscation patterns are best aligned to ATLAS evasion behavior.
                # Official ATLAS technique for evasion: AML.T0015 (Evade AI Model).
                'Noise Injection': 'AML.T0015 (Evade AI Model)',
                'Periodic Pattern': 'AML.T0015 (Evade AI Model)',
                'Gradient Masking': 'AML.T0015 (Evade AI Model)',
                # Combined obfuscation + reconstruction threat surface (multi-technique).
                'Combined Attack': 'AML.T0015 (Evade AI Model) + AML.T0024.001 (Invert AI Model)',
            }
            
            red_team_guidance = {
                'Clean Baseline': 'SKIP',
                'Noise Injection': 'PROBE',
                'Periodic Pattern': 'TARGET',
                'Gradient Masking': 'EXPLOIT',
                'Combined Attack': 'CRITICAL EXPLOIT'
            }
            
            blue_team_guidance = {
                'Clean Baseline': 'Baseline monitoring',
                'Noise Injection': 'Monitor & track',
                'Periodic Pattern': 'Increase DP noise',
                'Gradient Masking': 'Apply clipping NOW',
                'Combined Attack': 'URGENT: Full mitigation'
            }
            
            threats = list(threat_levels.keys())
            levels = list(threat_levels.values())
            colors_bar = ['green' if lvl < 0.3 else 'orange' if lvl < 0.7 else 'red' for lvl in levels]
            
            fig.add_trace(go.Bar(
                x=threats,
                y=levels,
                name='Threat Level',
                marker_color=colors_bar,
                customdata=[[mitre_techniques[t], red_team_guidance[t], blue_team_guidance[t]] for t in threats],
                hovertemplate=
                    '<b>%{x}</b><br>' +
                    '━━━━━━━━━━━━━━━━━━━━━━<br>' +
                    '<b>Threat Level:</b> %{y:.0%}<br>' +
                    '<b>🛡️ MITRE ATLAS:</b> %{customdata[0]}<br>' +
                    '<b>🔴 Red Team:</b> %{customdata[1]}<br>' +
                    '<b>🔵 Blue Team:</b> %{customdata[2]}<br>' +
                    '<extra></extra>'
            ), row=2, col=2)
            
            # Panel 6: Real-Time Monitoring - CONSISTENT WITH ANOMALY DETECTION
            # Use SAME gradient data and anomaly detection as Panel 4 for consistency
            
            # Re-use the obfuscated gradient data that was analyzed in Panel 4
            analysis_grad = gradients.get('obfuscated', gradients.get('loaded', list(gradients.values())[0]))
            grad_flat = analysis_grad.flatten() if analysis_grad.ndim > 1 else analysis_grad
            
            # Calculate rolling statistics (SAME as Panel 4)
            window = 20
            rolling_mean = np.convolve(grad_flat, np.ones(window)/window, mode='valid')
            rolling_std = np.array([np.std(grad_flat[max(0,i-window):i+window]) for i in range(len(rolling_mean))])
            
            # Detect anomalies (SAME logic as Panel 4)
            anomaly_threshold = 3 * np.std(grad_flat)
            mean_anomalies = np.abs(grad_flat[:len(rolling_mean)] - rolling_mean) > anomaly_threshold
            std_anomalies = rolling_std > (2 * np.mean(rolling_std))
            
            # Create timeline matching the sample positions
            time_points = len(rolling_mean)
            time_series = np.linspace(0, 10, time_points)
            
            # Build threat timeline based on ACTUAL anomaly detection
            threat_timeline = []
            
            for i in range(time_points):
                # Base threat from gradient statistics
                local_var = rolling_std[i]**2  # Variance at this position
                local_mag = abs(rolling_mean[i])  # Magnitude at this position
                
                # Calculate threat (matches Panel 4 anomaly logic)
                variance_component = min(0.4, local_var * 20)     # Variance contribution
                magnitude_component = min(0.3, local_mag * 3)     # Magnitude contribution
                
                base_threat = variance_component + magnitude_component
                
                # CRITICAL: Add spike if anomaly detected (CONSISTENCY with Panel 4)
                if i < len(mean_anomalies) and (mean_anomalies[i] or std_anomalies[i]):
                    # Anomaly detected - spike threat level
                    base_threat = min(1.0, base_threat + 0.4)  # Add 0.4 for anomalies
                
                threat_timeline.append(np.clip(base_threat, 0.0, 1.0))
            
            # Add enhanced hover data showing anomaly correlation
            anomaly_status = ['🚨 ANOMALY' if (mean_anomalies[i] or std_anomalies[i]) else 'Normal' for i in range(time_points)]
            threat_colors = ['red' if t > 0.7 else 'orange' if t > 0.5 else 'yellow' if t > 0.3 else 'green' for t in threat_timeline]
            
            fig.add_trace(go.Scatter(
                x=time_series,
                y=threat_timeline,
                name='Threat Level',
                mode='lines+markers',
                line=dict(color='purple', width=3),
                marker=dict(size=4, color=threat_colors),  # Color-coded by threat level
                fill='tozeroy',
                fillcolor='rgba(128,0,128,0.3)',
                customdata=anomaly_status,
                hovertemplate=
                    '<b>⚡ REAL-TIME MONITORING</b><br>' +
                    '━━━━━━━━━━━━━━━━━━━━━━<br>' +
                    '<b>Time:</b> %{x:.2f}s<br>' +
                    '<b>Threat Level:</b> %{y:.2%}<br>' +
                    '<b>Anomaly Status:</b> %{customdata}<br>' +
                    '<b>Classification:</b> ' +
                    '<span style="color:red">CRITICAL</span> (%{y} > 0.7) | ' +
                    '<span style="color:orange">HIGH</span> (0.5-0.7) | ' +
                    '<span style="color:yellow">MEDIUM</span> (0.3-0.5) | ' +
                    '<span style="color:green">LOW</span> (< 0.3)<br>' +
                    '<b>🔴 Red Team:</b> Correlate with Panel 4 anomalies<br>' +
                    '<b>🔵 Blue Team:</b> Investigate spikes immediately<br>' +
                    '<extra></extra>'
            ), row=2, col=3)
            
            # Add threshold lines to monitoring panel
            fig.add_hline(y=0.7, line_dash="dash", line_color="red", row=2, col=3, 
                         annotation_text="Critical", annotation_position="right")
            fig.add_hline(y=0.5, line_dash="dash", line_color="orange", row=2, col=3,
                         annotation_text="High", annotation_position="right")
            fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=2, col=3,
                         annotation_text="Medium", annotation_position="right")
            
            # Add clear, elegant axis labels for all panels
            fig.update_xaxes(title_text="Gradient Index", row=1, col=1)
            fig.update_yaxes(title_text="Gradient Magnitude", row=1, col=1)
            
            fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
            fig.update_yaxes(title_text="Power Spectral Density", row=1, col=2)
            
            fig.update_xaxes(title_text="Gradient Value", row=1, col=3)
            fig.update_yaxes(title_text="Probability Density", row=1, col=3)
            
            fig.update_xaxes(title_text="Sample Position", row=2, col=1)
            fig.update_yaxes(title_text="Rolling Mean ± σ", row=2, col=1)
            
            fig.update_xaxes(title_text="Attack Type", tickangle=45, row=2, col=2)
            fig.update_yaxes(title_text="Threat Level (0-1)", row=2, col=2)
            
            fig.update_xaxes(title_text="Time (seconds)", row=2, col=3)
            fig.update_yaxes(title_text="Threat Level", row=2, col=3)
            
            # Update layout with permanent title and zoom/scroll capabilities
            fig.update_layout(
                title=dict(
                    text='NeurInSpectre - Obfuscated Gradient Analysis',
                    font=dict(size=20),
                    y=0.98,
                    x=0.5,
                    xanchor='center'
                ),
                # Enable professional zoom/pan/scroll tools
                dragmode='zoom',  # Default to zoom mode
                hovermode='closest',  # Show nearest data point
                selectdirection='h',  # Horizontal selection for time series
                showlegend=True,
                legend=dict(
                    orientation="h",  # Horizontal legend
                    yanchor="top",
                    y=-0.12,  # Much further down to avoid rotated x-axis labels
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(20,20,20,0.9)",
                    bordercolor="gray",
                    borderwidth=1,
                    font=dict(size=9)
                ),
                template='plotly_dark',
                height=1300,  # Even taller to prevent x-axis label overlap with legend
                width=1600,   # Wider for better spacing
                margin=dict(l=80, r=80, t=140, b=250),  # Much larger bottom margin for rotated labels
                hoverlabel=dict(
                    bgcolor="rgba(0,0,0,0.95)",
                    font_size=12,
                    font_family="monospace",
                    bordercolor="cyan",
                    font_color="white"
                ),
                # Footer info - VERY FAR BELOW to prevent overlap with rotated labels
                annotations=[
                    dict(
                        text='📊 Dataset: 512 samples each (clean vs obfuscated) | Patterns: Periodic + Spikes + Bias + HF noise | ' +
                             '💡 Hover over any chart for red🔴/blue🔵 team guidance',
                        xref="paper", yref="paper",
                        x=0.5, y=-0.18,  # Very far below legend and rotated labels
                        xanchor='center', yanchor='top',
                        showarrow=False,
                        font=dict(size=9, color='darkgray')
                    )
                ]
            )
            
            # Ensure we're saving to the correct output directory
            from pathlib import Path
            output_path = Path(save_path)
            
            # Save interactive HTML with professional configuration - PRESERVE HOVER ON ZOOM
            config = {
                'displayModeBar': True,  # Always show toolbar
                'displaylogo': False,    # Remove Plotly logo
                'modeBarButtonsToAdd': ['select2d', 'lasso2d'],  # Add selection tools
                'modeBarButtonsToRemove': [],  # Keep all zoom/pan tools
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'neurinspectre_gradient_analysis',
                    'height': 1300,
                    'width': 1600,
                    'scale': 2  # Higher resolution export
                },
                'scrollZoom': True,  # Enable scroll to zoom - CRITICAL for large datasets
                'editable': False,   # Prevent accidental edits
                'responsive': True   # Responsive resizing
            }
            
            # Ensure hover data persists through zoom (uirevision)
            fig.update_layout(uirevision='constant')  # CRITICAL: Preserves hover on zoom
            
            fig.write_html(str(output_path), config=config)
            print(f"🚀 Interactive Plotly dashboard saved to: {output_path}")
            print("   ✅ Zoom: Click and drag | Pan: Shift+drag | Reset: Double-click")
            print("   ✅ Scroll zoom enabled for precise navigation")
            
            # Also save static PNG (backward compatibility) in same location
            png_path = str(output_path).replace('_interactive.html', '.png')
            self._create_static_matplotlib_dashboard(gradients, png_path)
            
        except Exception as e:
            print(f"⚠️ Could not create interactive dashboard: {e}")
            print("📋 Falling back to static PNG only")
    
    def _create_static_matplotlib_dashboard(self, gradients: Dict[str, np.ndarray], png_path: str):
        """Create static matplotlib PNG dashboard"""
        fig = plt.figure(figsize=self.figsize)
        fig.suptitle('GRADIENT OBFUSCATION ANALYSIS DASHBOARD', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_gradient_comparison(ax1, gradients)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_spectral_analysis(ax2, gradients)
        
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_statistical_distribution(ax3, gradients)
        
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_obfuscation_patterns(ax4, gradients)
        
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_threat_assessment(ax5, gradients)
        
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_realtime_monitoring(ax6, gradients)
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight', 
                   facecolor=self.colors['background'])
        
        print(f"🎯 Static PNG saved to: {png_path}")
    
    def create_attack_timeline_visualization(self, gradients: Dict[str, np.ndarray],
                                           save_path: str = "attack_timeline.png") -> None:
        """Create a timeline visualization derived from the provided gradients.

        Important:
        - The x-axis is *relative time* unless the caller supplies real timestamps externally.
        - This function does not fabricate detection confidence. The "Detection Confidence" curve
          is a normalized threat timeline derived from windowed magnitude/variance + anomaly flags.
        """
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('GRADIENT ATTACK TIMELINE ANALYSIS', fontsize=16, fontweight='bold')
        
        # Timeline 1: Attack Detection Over Time
        ax1.set_title('Attack Detection Timeline', fontsize=12, fontweight='bold')
        
        # Pick a series to derive a threat timeline.
        key = "obfuscated" if "obfuscated" in gradients else next(iter(gradients.keys()), None)
        if key is None:
            raise ValueError("No gradients provided for timeline visualization.")

        grad_series = gradients[key]
        if isinstance(grad_series, np.ndarray) and grad_series.ndim == 2:
            series = np.mean(grad_series, axis=1).reshape(-1)
        else:
            series = np.asarray(grad_series).reshape(-1)
        series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
        if series.size < 4:
            raise ValueError("Gradient series too short for timeline visualization.")

        window = int(max(5, min(50, series.size // 20 if series.size >= 20 else series.size)))
        window = int(max(2, min(window, series.size)))
        rolling_mean = np.convolve(series, np.ones(window) / float(window), mode='valid')
        if rolling_mean.size == 0:
            raise ValueError("Gradient series too short for selected window size.")
        rolling_std = np.array(
            [np.std(series[max(0, i - window) : min(series.size, i + window)]) for i in range(rolling_mean.size)]
        )

        anomaly_threshold = 3.0 * float(np.std(series)) if series.size >= 2 else 0.0
        mean_anomalies = np.abs(series[: rolling_mean.size] - rolling_mean) > anomaly_threshold
        std_thresh = 2.0 * float(np.mean(rolling_std)) if rolling_std.size else float("inf")
        std_anomalies = rolling_std > std_thresh
        anomaly_mask = mean_anomalies | std_anomalies

        threat_timeline = []
        for i in range(rolling_mean.size):
            local_var = float(rolling_std[i] ** 2)
            local_mag = float(abs(rolling_mean[i]))
            variance_component = min(0.4, local_var * 20.0)
            magnitude_component = min(0.3, local_mag * 3.0)
            base_threat = variance_component + magnitude_component
            if bool(anomaly_mask[i]):
                base_threat = min(1.0, base_threat + 0.4)
            threat_timeline.append(float(np.clip(base_threat, 0.0, 1.0)))

        time_points = np.linspace(0.0, 24.0, len(threat_timeline))  # relative hours
        detection_confidence = np.asarray(threat_timeline, dtype=float)

        ax1.plot(time_points, detection_confidence, color=self.colors['obfuscated'], linewidth=2)
        ax1.fill_between(time_points, 0, detection_confidence, alpha=0.3, color=self.colors['obfuscated'])

        # Mark anomaly events (limit markers to avoid clutter).
        anomaly_idx = np.where(anomaly_mask)[0]
        if anomaly_idx.size > 0:
            if anomaly_idx.size > 10:
                pick = np.linspace(0, anomaly_idx.size - 1, 10).astype(int)
                anomaly_idx = anomaly_idx[pick]
            for idx in anomaly_idx:
                t = float(time_points[int(idx)])
                ax1.axvline(x=t, color='red', linestyle='--', alpha=0.8)
                ax1.text(t, 0.95, '!', fontsize=12, ha='center')
        
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Detection Confidence')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Timeline 2: Gradient Evolution
        ax2.set_title('Gradient Evolution Pattern', fontsize=12, fontweight='bold')
        
        if key is not None:
            grad_data = series
            # Show evolution over time windows
            window_size = max(1, int(len(grad_data) // 20))
            evolution_points = []
            time_windows = []
            
            for i in range(0, len(grad_data), window_size):
                window_data = grad_data[i:i+window_size]
                if len(window_data) > 0:
                    evolution_points.append(np.mean(np.abs(window_data)))
                    time_windows.append(i / len(grad_data) * 24)  # Convert to hours
            
            ax2.plot(time_windows, evolution_points, 'o-', color=self.colors['accent'], 
                    linewidth=2, markersize=6)
            ax2.fill_between(time_windows, 0, evolution_points, alpha=0.3, color=self.colors['accent'])
        
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Gradient Magnitude')
        ax2.grid(True, alpha=0.3)
        
        # Timeline 3: Recommended Response Actions (derived from anomaly windows)
        ax3.set_title('Response Actions Timeline', fontsize=12, fontweight='bold')

        response_actions = []
        if anomaly_idx.size > 0:
            for idx in anomaly_idx:
                lvl = float(detection_confidence[int(idx)])
                if lvl >= 0.7:
                    action, color = "Block", "red"
                elif lvl >= 0.5:
                    action, color = "Alert", "orange"
                else:
                    action, color = "Monitor", "blue"
                response_actions.append((float(time_points[int(idx)]), action, color))
        else:
            response_actions.append((0.0, "Monitor", "blue"))

        for i, (time_val, action, color) in enumerate(response_actions):
            ax3.barh(i, 1, left=time_val, height=0.6, color=color, alpha=0.7)
            ax3.text(time_val + 0.1, i, action, va='center', fontweight='bold')
        
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Response Actions')
        ax3.set_ylim(-0.5, len(response_actions)-0.5)
        ax3.set_xlim(0, 24)
        ax3.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.colors['background'])
        print(f"⏰ Attack timeline visualization saved to: {save_path}")
    
    def integrate_with_mathematical_foundations(self) -> Dict[str, Any]:
        """Integrate with the mathematical foundations module"""
        
        try:
            # Prefer relative import for in-repo execution / type checkers.
            from ...mathematical import GPUAcceleratedMathEngine
            
            print("🔗 Integrating with NeurInSpectre Mathematical Foundations...")
            
            # Initialize math engine
            math_engine = GPUAcceleratedMathEngine(precision='float32', device_preference='auto')
            
            # Load gradients (demo-only)
            gradients = self.load_sample_gradients()
            
            results = {}
            
            for grad_type, grad_data in gradients.items():
                print(f"   🔬 Analyzing {grad_type} gradients...")
                
                # Perform advanced spectral decomposition
                analysis_results = math_engine.advanced_spectral_decomposition(
                    grad_data, decomposition_levels=3
                )
                
                results[grad_type] = {
                    'spectral_analysis': analysis_results,
                    'gradient_data': grad_data,
                    'summary_stats': {
                        'mean': float(np.mean(grad_data)),
                        'std': float(np.std(grad_data)),
                        'min': float(np.min(grad_data)),
                        'max': float(np.max(grad_data))
                    }
                }
            
            print("✅ Mathematical foundations integration complete!")
            return results
            
        except ImportError as e:
            print(f"⚠️ Mathematical foundations not available: {e}")
            # Fallback to basic analysis
            gradients = self.load_sample_gradients()
            return {grad_type: {'gradient_data': grad_data} for grad_type, grad_data in gradients.items()}
        except Exception as e:
            print(f"❌ Integration error: {e}")
            return {}

def main():
    """Main function for standalone execution"""
    print("🎨 NeurInSpectre Obfuscated Gradient Visualizer")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = ObfuscatedGradientVisualizer()
    
    # Load or generate gradient data (demo-only)
    print("📊 Loading sample gradient data (demo)...")
    gradients = visualizer.load_sample_gradients()
    
    # Create comprehensive visualization
    print("🎯 Creating comprehensive gradient analysis dashboard...")
    visualizer.create_comprehensive_visualization(gradients)
    
    # Create attack timeline
    print("⏰ Creating attack timeline visualization...")
    visualizer.create_attack_timeline_visualization(gradients)
    
    # Integrate with mathematical foundations
    print("🔗 Testing mathematical foundations integration...")
    analysis_results = visualizer.integrate_with_mathematical_foundations()
    
    print("\n✅ Visualization suite complete!")
    print("Generated files:")
    print("   • gradient_analysis_dashboard.png")
    print("   • attack_timeline.png")
    
    if analysis_results:
        print("\n📊 Analysis Summary:")
        for grad_type, results in analysis_results.items():
            if 'summary_stats' in results:
                stats = results['summary_stats']
                print(f"   {grad_type.title()}: μ={stats['mean']:.4f}, σ={stats['std']:.4f}")

if __name__ == "__main__":
    main() 