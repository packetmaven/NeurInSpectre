"""
Enhanced Security Visualizer for NeurInSpectre
Advanced visualization utilities for NeurInSpectre security reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import logging
from contextlib import contextmanager, nullcontext

logger = logging.getLogger(__name__)

class Enhanced2025SecurityVisualizer:
    """
    Enhanced security visualizer for generating charts and dashboards from analysis outputs.
    """
    
    def __init__(self, output_dir: str = "./security_reports"):
        """Initialize the enhanced security visualizer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Per-instance plotting style (applied via context managers at render time).
        # Avoid mutating global matplotlib/seaborn state.
        self._mpl_style = 'seaborn-v0_8'
        self._sns_palette = "husl"

        # Do not configure global logging handlers here; leave that to the application/CLI.

    @contextmanager
    def _plot_context(self):
        """Apply plotting style/palette without mutating global state."""
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

        with style_ctx, palette_ctx:
            yield
        
    def create_threat_landscape_visualization(self, 
                                            threat_data: Dict[str, Any],
                                            save_path: Optional[str] = None) -> str:
        """Create comprehensive threat landscape visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Threat Distribution", "Attack Timeline", 
                          "Confidence Scores", "Detection Methods"),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "radar"}]]
        )
        
        # Threat distribution pie chart
        if 'threat_distribution' in threat_data:
            labels = list(threat_data['threat_distribution'].keys())
            values = list(threat_data['threat_distribution'].values())
            
            fig.add_trace(
                go.Pie(labels=labels, values=values, name="Threats"),
                row=1, col=1
            )
        
        # Attack timeline
        if 'attack_timeline' in threat_data:
            timeline = threat_data['attack_timeline']
            fig.add_trace(
                go.Scatter(
                    x=timeline.get('timestamps', []),
                    y=timeline.get('threat_levels', []),
                    mode='lines+markers',
                    name='Threat Level Over Time'
                ),
                row=1, col=2
            )
        
        # Confidence scores bar chart
        if 'confidence_scores' in threat_data:
            methods = list(threat_data['confidence_scores'].keys())
            scores = list(threat_data['confidence_scores'].values())
            
            fig.add_trace(
                go.Bar(x=methods, y=scores, name="Confidence"),
                row=2, col=1
            )
        
        # Detection methods radar chart
        if 'detection_methods' in threat_data:
            methods = threat_data['detection_methods']
            
            fig.add_trace(
                go.Scatterpolar(
                    r=list(methods.values()),
                    theta=list(methods.keys()),
                    fill='toself',
                    name='Detection Coverage'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="NeurInSpectre Security Threat Landscape Analysis",
            showlegend=True,
            height=800
        )
        
        # Save visualization
        if save_path is None:
            save_path = self.output_dir / "threat_landscape.html"
        
        fig.write_html(save_path)
        logger.info(f"Threat landscape visualization saved to: {save_path}")
        
        return str(save_path)
    
    def create_attack_detection_heatmap(self, 
                                       detection_results: Dict[str, Any],
                                       save_path: Optional[str] = None) -> str:
        """Create attack detection heatmap visualization"""
        
        # Create detection matrix
        if 'detection_matrix' in detection_results:
            matrix = np.array(detection_results['detection_matrix'])

            with self._plot_context():
                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    matrix,
                    annot=True,
                    cmap='RdYlBu_r',
                    cbar_kws={'label': 'Detection Score'},
                    xticklabels=detection_results.get('attack_types', []),
                    yticklabels=detection_results.get('detection_methods', [])
                )
                
                plt.title('Attack Detection Effectiveness Heatmap')
                plt.xlabel('Attack Types')
                plt.ylabel('Detection Methods')
                plt.tight_layout()
                
                # Save heatmap
                if save_path is None:
                    save_path = self.output_dir / "detection_heatmap.png"
                
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Detection heatmap saved to: {save_path}")
            return str(save_path)
        
        return None
    
    def create_neural_transport_dynamics_plot(self, 
                                             transport_data: Dict[str, Any],
                                             save_path: Optional[str] = None) -> str:
        """Create neural transport dynamics visualization"""

        with self._plot_context():
            fig = plt.figure(figsize=(15, 10))
        
            # Create 3D subplot for transport dynamics
            ax1 = fig.add_subplot(221, projection='3d')
        
        if 'transport_trajectories' in transport_data:
            trajectories = transport_data['transport_trajectories']
            
            # Plot normal vs anomalous trajectories
            normal_traj = trajectories.get('normal', None)
            anomalous_traj = trajectories.get('anomalous', None)

            # No synthetic/demo fallback: only plot what was provided.
            plotted_any = False
            if normal_traj is not None:
                normal_traj = np.asarray(normal_traj)
                if normal_traj.ndim == 2 and normal_traj.shape[1] >= 3 and normal_traj.shape[0] > 0:
                    ax1.scatter(normal_traj[:, 0], normal_traj[:, 1], normal_traj[:, 2],
                               c='blue', alpha=0.6, label='Normal')
                    plotted_any = True
            if anomalous_traj is not None:
                anomalous_traj = np.asarray(anomalous_traj)
                if anomalous_traj.ndim == 2 and anomalous_traj.shape[1] >= 3 and anomalous_traj.shape[0] > 0:
                    ax1.scatter(anomalous_traj[:, 0], anomalous_traj[:, 1], anomalous_traj[:, 2],
                               c='red', alpha=0.8, label='Anomalous')
                    plotted_any = True
            
            ax1.set_title('Neural Transport Dynamics')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            if plotted_any:
                ax1.legend()
        
            # Evasion score distribution
            ax2 = fig.add_subplot(222)
            if 'evasion_scores' in transport_data:
                scores = transport_data['evasion_scores']
                ax2.hist(scores, bins=50, alpha=0.7, color='orange')
                ax2.set_title('Evasion Score Distribution')
                ax2.set_xlabel('Evasion Score')
                ax2.set_ylabel('Frequency')
        
            # Transport velocity over time
            ax3 = fig.add_subplot(223)
            if 'transport_velocity' in transport_data:
                velocity = transport_data['transport_velocity']
                time_steps = np.arange(len(velocity))
                ax3.plot(time_steps, velocity, 'g-', linewidth=2)
                ax3.set_title('Transport Velocity Over Time')
                ax3.set_xlabel('Time Steps')
                ax3.set_ylabel('Velocity')
        
            # Detection confidence over time
            ax4 = fig.add_subplot(224)
            if 'detection_confidence' in transport_data:
                confidence = transport_data['detection_confidence']
                time_steps = np.arange(len(confidence))
                ax4.plot(time_steps, confidence, 'purple', linewidth=2)
                ax4.axhline(y=0.7, color='red', linestyle='--', label='Threshold')
                ax4.set_title('Detection Confidence Over Time')
                ax4.set_xlabel('Time Steps')
                ax4.set_ylabel('Confidence')
                ax4.legend()
        
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = self.output_dir / "transport_dynamics.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Transport dynamics plot saved to: {save_path}")
        return str(save_path)
    
    def create_adversarial_attack_analysis(self, 
                                         attack_data: Dict[str, Any],
                                         save_path: Optional[str] = None) -> str:
        """Create comprehensive adversarial attack analysis visualization"""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=("TS-Inverse Detection", "ConcreTizer Analysis", 
                          "EDNN Attack Patterns", "AttentionGuard Results",
                          "Attack Effectiveness", "Gradient Inversion"),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}, {"type": "surface"}]]
        )
        
        # TS-Inverse detection results
        if 'ts_inverse_results' in attack_data:
            ts_data = attack_data['ts_inverse_results']
            fig.add_trace(
                go.Scatter(
                    x=ts_data.get('feature_indices', []),
                    y=ts_data.get('inversion_scores', []),
                    mode='markers',
                    name='TS-Inverse'
                ),
                row=1, col=1
            )
        
        # ConcreTizer heatmap
        if 'concretizer_results' in attack_data:
            concretizer_data = attack_data['concretizer_results']
            if 'reconstruction_matrix' in concretizer_data:
                matrix = np.array(concretizer_data['reconstruction_matrix'])
                fig.add_trace(
                    go.Heatmap(
                        z=matrix,
                        colorscale='Viridis',
                        name='ConcreTizer'
                    ),
                    row=1, col=2
                )
        
        # EDNN attack patterns
        if 'ednn_results' in attack_data:
            ednn_data = attack_data['ednn_results']
            fig.add_trace(
                go.Bar(
                    x=ednn_data.get('neighbor_indices', []),
                    y=ednn_data.get('attack_scores', []),
                    name='EDNN Scores'
                ),
                row=1, col=3
            )
        
        # AttentionGuard results
        if 'attention_guard_results' in attack_data:
            ag_data = attack_data['attention_guard_results']
            fig.add_trace(
                go.Scatter(
                    x=ag_data.get('attention_weights', []),
                    y=ag_data.get('misbehavior_scores', []),
                    mode='markers',
                    name='AttentionGuard'
                ),
                row=2, col=1
            )
        
        # Attack effectiveness pie chart
        if 'attack_effectiveness' in attack_data:
            effectiveness = attack_data['attack_effectiveness']
            fig.add_trace(
                go.Pie(
                    labels=list(effectiveness.keys()),
                    values=list(effectiveness.values()),
                    name="Effectiveness"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Adversarial Attack Analysis Dashboard",
            showlegend=True,
            height=1000
        )
        
        # Save visualization
        if save_path is None:
            save_path = self.output_dir / "adversarial_analysis.html"
        
        fig.write_html(save_path)
        logger.info(f"Adversarial attack analysis saved to: {save_path}")
        
        return str(save_path)
    
    def create_security_timeline_visualization(self, 
                                             timeline_data: Dict[str, Any],
                                             save_path: Optional[str] = None) -> str:
        """Create security timeline visualization"""
        
        fig = go.Figure()
        
        if 'events' in timeline_data:
            events = timeline_data['events']
            
            # Create timeline traces for different event types
            event_types = {}
            for event in events:
                event_type = event.get('type', 'unknown')
                if event_type not in event_types:
                    event_types[event_type] = {
                        'timestamps': [],
                        'threat_levels': [],
                        'descriptions': []
                    }
                
                event_types[event_type]['timestamps'].append(event.get('timestamp', 0))
                event_types[event_type]['threat_levels'].append(event.get('threat_level', 0))
                event_types[event_type]['descriptions'].append(event.get('description', ''))
            
            # Add traces for each event type
            colors = ['red', 'orange', 'yellow', 'blue', 'green', 'purple']
            for i, (event_type, data) in enumerate(event_types.items()):
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamps'],
                        y=data['threat_levels'],
                        mode='lines+markers',
                        name=event_type.replace('_', ' ').title(),
                        line=dict(color=colors[i % len(colors)]),
                        text=data['descriptions'],
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Time: %{x}<br>' +
                                    'Threat Level: %{y}<br>' +
                                    '<extra></extra>'
                    )
                )
        
        fig.update_layout(
            title="Security Events Timeline",
            xaxis_title="Time",
            yaxis_title="Threat Level",
            hovermode='closest',
            height=600
        )
        
        # Save timeline
        if save_path is None:
            save_path = self.output_dir / "security_timeline.html"
        
        fig.write_html(save_path)
        logger.info(f"Security timeline saved to: {save_path}")
        
        return str(save_path)
    
    def create_comprehensive_security_dashboard(self, 
                                              security_data: Dict[str, Any],
                                              save_path: Optional[str] = None) -> str:
        """Create comprehensive security dashboard"""
        
        # Create main dashboard figure
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=("Threat Overview", "Attack Detection", "Neural Transport",
                          "Adversarial Patterns", "Evasion Analysis", "Security Metrics",
                          "Time Series Analysis", "Correlation Matrix", "Recommendations"),
            specs=[[{"type": "pie"}, {"type": "bar"}, {"type": "scatter3d"}],
                   [{"type": "heatmap"}, {"type": "scatter"}, {"type": "indicator"}],
                   [{"type": "scatter"}, {"type": "heatmap"}, {"type": "table"}]]
        )
        
        # Threat overview
        if 'threat_overview' in security_data:
            threat_data = security_data['threat_overview']
            fig.add_trace(
                go.Pie(
                    labels=list(threat_data.keys()),
                    values=list(threat_data.values()),
                    name="Threats"
                ),
                row=1, col=1
            )
        
        # Attack detection scores
        if 'detection_scores' in security_data:
            scores = security_data['detection_scores']
            fig.add_trace(
                go.Bar(
                    x=list(scores.keys()),
                    y=list(scores.values()),
                    name="Detection"
                ),
                row=1, col=2
            )
        
        # Security metrics indicator
        if 'security_metrics' in security_data:
            metrics = security_data['security_metrics']
            overall_score = metrics.get('overall_score', 0.5)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_score,
                    title={'text': "Security Score"},
                    gauge={'axis': {'range': [None, 1]},
                          'bar': {'color': "darkblue"},
                          'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                   {'range': [0.5, 1], 'color': "gray"}],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': 0.9}}
                ),
                row=2, col=3
            )
        
        fig.update_layout(
            title="NeurInSpectre Comprehensive Security Dashboard",
            showlegend=True,
            height=1200
        )
        
        # Save dashboard
        if save_path is None:
            save_path = self.output_dir / "comprehensive_dashboard.html"
        
        fig.write_html(save_path)
        logger.info(f"Comprehensive security dashboard saved to: {save_path}")
        
        return str(save_path)
    
    def generate_security_report(self, 
                               analysis_results: Dict[str, Any],
                               save_path: Optional[str] = None) -> str:
        """Generate comprehensive security report"""
        
        # Create report data
        report_data = {
            'timestamp': analysis_results.get('timestamp', 'Unknown'),
            'overall_threat_level': analysis_results.get('overall_threat_level', 'Unknown'),
            'confidence_score': analysis_results.get('confidence_score', 0.0),
            'detected_attacks': analysis_results.get('detected_attacks', []),
            'security_scores': analysis_results.get('security_scores', {}),
            'recommendations': analysis_results.get('recommendations', [])
        }
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NeurInSpectre Security Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .threat-high {{ border-left-color: #ff4444; }}
                .threat-medium {{ border-left-color: #ff8800; }}
                .threat-low {{ border-left-color: #44ff44; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                ul {{ padding-left: 20px; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🛡️ NeurInSpectre Security Analysis Report</h1>
                <p><strong>Generated:</strong> {report_data['timestamp']}</p>
                <p><strong>Overall Threat Level:</strong> <span style="color: red; font-weight: bold;">{report_data['overall_threat_level'].upper()}</span></p>
                <p><strong>Confidence Score:</strong> {report_data['confidence_score']:.3f}</p>
            </div>
            
            <div class="section">
                <h2>🚨 Detected Attacks</h2>
                <ul>
        """
        
        for attack in report_data['detected_attacks']:
            html_content += f"<li><strong>{attack.get('type', 'Unknown')}</strong> - Confidence: {attack.get('confidence', 0.0):.3f}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>📊 Security Component Scores</h2>
        """
        
        for component, score in report_data['security_scores'].items():
            html_content += f'<div class="metric"><strong>{component.replace("_", " ").title()}:</strong> {score:.3f}</div>'
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
                <ul>
        """
        
        for rec in report_data['recommendations']:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save report
        if save_path is None:
            save_path = self.output_dir / "security_report.html"
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Security report saved to: {save_path}")
        return str(save_path) 