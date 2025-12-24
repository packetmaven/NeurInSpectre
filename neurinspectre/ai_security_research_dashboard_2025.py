#!/usr/bin/env python3
"""
AI Security Research Dashboard 2025
Interactive dashboard for AI security monitoring and analysis.
"""

import logging
import importlib
import random
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Enhanced upload integration is optional and is loaded lazily in `__main__`
# to avoid import-time side effects when optional dependencies are missing.
ENHANCED_UPLOAD_AVAILABLE = False
integrate_enhanced_upload = None  # type: ignore

# Latest 2025 AI Security Research Database
RESEARCH_2025 = {
    "Membership Inference Attacks": {
        "Gradient-based MINT (Mancera et al.)": {
            "auc": 0.99, "threat": "CRITICAL",
            "description": "Gradient-based Membership Inference Test achieving 85-99% AUC on LLMs",
            "red_action": "Exploit gradient information during model inference to identify training data membership",
            "blue_action": "Implement differential privacy (ε≤1.0) and gradient noise injection (σ=0.1-0.5)"
        },
        "Adversarial Iterations MIA (Xue et al.)": {
            "auc": 0.94, "threat": "HIGH",
            "description": "Novel attack using adversarial sample generation iteration counts for inference",
            "red_action": "Analyze adversarial generation patterns and iteration requirements for membership classification",
            "blue_action": "Randomize adversarial training procedures and implement iteration count obfuscation"
        },
        "SOFT Defense Bypass (Zhang et al.)": {
            "auc": 0.87, "threat": "HIGH", 
            "description": "Attacks against selective data obfuscation defenses in LLM fine-tuning",
            "red_action": "Target influential data selection mechanisms and exploit obfuscation patterns",
            "blue_action": "Enhance obfuscation randomization and implement multi-layer protection strategies"
        }
    },
    "Model Extraction Attacks": {
        "Query-Efficient Extraction": {
            "success": 0.92, "threat": "CRITICAL",
            "description": "Optimized active learning strategies for model stealing with minimal API calls",
            "red_action": "Use uncertainty-based sampling and exploit model confidence patterns for extraction",
            "blue_action": "Implement query rate limiting (≤100/hour), anomaly detection, and watermarking"
        },
        "Federated Learning Extraction": {
            "success": 0.89, "threat": "HIGH",
            "description": "Model extraction attacks exploiting gradient sharing in federated environments",
            "red_action": "Exploit gradient aggregation patterns and inject malicious clients for reconstruction",
            "blue_action": "Implement secure aggregation protocols and Byzantine-robust aggregation methods"
        }
    },
    "Gradient Leakage Attacks": {
        "Enhanced Deep Gradient Inversion": {
            "fidelity": 0.96, "threat": "CRITICAL",
            "description": "High-fidelity training data reconstruction from gradient information",
            "red_action": "Use optimization-based inversion for pixel-level reconstruction from gradients",
            "blue_action": "Apply gradient compression, noise injection, and secure aggregation methods"
        },
        "Batch-level Reconstruction": {
            "fidelity": 0.91, "threat": "HIGH",
            "description": "Reconstruct entire training batches from shared gradient information",
            "red_action": "Target batch normalization statistics and exploit gradient correlations",
            "blue_action": "Implement batch shuffling, gradient masking, and size randomization"
        }
    },
    "Property Inference Attacks": {
        "Distribution Inference": {
            "accuracy": 0.93, "threat": "HIGH",
            "description": "Infer sensitive training data distribution properties from model behavior",
            "red_action": "Analyze output distributions and exploit statistical patterns in predictions",
            "blue_action": "Add output perturbation mechanisms and implement distribution masking"
        },
        "Attribute Inference": {
            "accuracy": 0.88, "threat": "MEDIUM",
            "description": "Infer sensitive demographic and behavioral attributes from model predictions",
            "red_action": "Correlate predictions with auxiliary information to infer sensitive attributes",
            "blue_action": "Implement fairness-aware training and output sanitization techniques"
        }
    }
}

# MITRE ATLAS Framework Mapping (selected high-signal techniques)
# Official source for names/IDs: `neurinspectre/mitre_atlas/stix-atlas.json`
ATLAS_TECHNIQUES = {
    "AML.T0024.002": {"name": "Extract AI Model", "severity": "HIGH", "detection": 0.55},
    "AML.T0024.000": {"name": "Infer Training Data Membership", "severity": "CRITICAL", "detection": 0.45},
    "AML.T0024.001": {"name": "Invert AI Model", "severity": "HIGH", "detection": 0.50},
    "AML.T0043": {"name": "Craft Adversarial Data", "severity": "HIGH", "detection": 0.69},
    "AML.T0020": {"name": "Poison Training Data", "severity": "CRITICAL", "detection": 0.41},
    "AML.T0070": {"name": "RAG Poisoning", "severity": "CRITICAL", "detection": 0.52},
    "AML.T0051": {"name": "LLM Prompt Injection", "severity": "CRITICAL", "detection": 0.61},
    "AML.T0054": {"name": "LLM Jailbreak", "severity": "HIGH", "detection": 0.58},
    "AML.T0057": {"name": "LLM Data Leakage", "severity": "CRITICAL", "detection": 0.45},
}

class AISecurityResearchDashboard:
    def __init__(self, *, allow_simulated: bool = False):
        """
        Args:
            allow_simulated: If True, run a demo mode that generates synthetic timeline
                metrics (randomized). If False (default), the dashboard will not
                fabricate telemetry; timelines will remain empty unless wired to
                a real data source.
        """
        self.app = dash.Dash(__name__)
        self.allow_simulated = bool(allow_simulated)
        
        # Real-time data streams
        self.mia_timeline = deque(maxlen=60)
        self.gradient_timeline = deque(maxlen=60)
        self.extraction_timeline = deque(maxlen=60)
        self.property_timeline = deque(maxlen=60)
        
        # Threat intelligence
        self.threat_metrics = {
            "active_campaigns": 0,
            "new_cves": 0,
            "attack_vectors": 0,
            "compromised_models": 0
        }
        
        if self.allow_simulated:
            self.start_data_generation()
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            # Header Section
            html.Div([
                html.H1("🔒 AI Security Research Dashboard 2025", 
                       style={'textAlign': 'center', 'color': '#00ff00', 'marginBottom': '10px'}),
                html.H3("Offensive AI/ML Security Research | Red/Blue Team Intelligence",
                       style={'textAlign': 'center', 'color': '#ffffff', 'marginBottom': '20px'}),
                html.P("Comprehensive analysis across: Membership Inference, Gradient Leakage, Model Extraction, Property Inference",
                      style={'textAlign': 'center', 'color': '#aaaaaa', 'fontSize': '14px'})
            ], style={'backgroundColor': '#1a1a1a', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),

            # Mode banner
            html.Div(
                [
                    html.B("Mode: "),
                    ("SIMULATED (demo)" if self.allow_simulated else "REAL-DATA ONLY (no synthetic telemetry)"),
                ],
                style={
                    "backgroundColor": "#111827",
                    "color": "#e5e7eb",
                    "padding": "8px 12px",
                    "borderRadius": "8px",
                    "marginBottom": "14px",
                    "fontFamily": "monospace",
                },
            ),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Update Interval:", style={'color': '#ffffff', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='interval-selector',
                        options=[
                            {'label': '1 Second', 'value': 1000},
                            {'label': '2 Seconds', 'value': 2000},
                            {'label': '5 Seconds', 'value': 5000}
                        ],
                        value=2000,
                        style={'width': '120px', 'display': 'inline-block'}
                    )
                ], style={'display': 'inline-block', 'marginRight': '20px'}),
                
                html.Div([
                    html.Label("Threat Filter:", style={'color': '#ffffff', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='threat-filter',
                        options=[
                            {'label': 'All Threats', 'value': 'ALL'},
                            {'label': 'Critical Only', 'value': 'CRITICAL'},
                            {'label': 'High & Critical', 'value': 'HIGH_CRITICAL'}
                        ],
                        value='ALL',
                        style={'width': '150px', 'display': 'inline-block'}
                    )
                ], style={'display': 'inline-block'})
            ], style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            # Main Dashboard Grid
            html.Div([
                # Row 1: Threat Intelligence Overview
                html.Div([
                    html.Div([dcc.Graph(id='threat-overview')], 
                            style={'width': '33%', 'display': 'inline-block'}),
                    html.Div([dcc.Graph(id='attack-success-matrix')], 
                            style={'width': '33%', 'display': 'inline-block'}),
                    html.Div([dcc.Graph(id='atlas-heatmap')], 
                            style={'width': '33%', 'display': 'inline-block'})
                ], style={'marginBottom': '20px'}),
                
                # Row 2: Attack Analysis
                html.Div([
                    html.Div([dcc.Graph(id='mia-analysis')], 
                            style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([dcc.Graph(id='gradient-analysis')], 
                            style={'width': '50%', 'display': 'inline-block'})
                ], style={'marginBottom': '20px'}),
                
                # Row 3: Advanced Attacks
                html.Div([
                    html.Div([dcc.Graph(id='extraction-analysis')], 
                            style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([dcc.Graph(id='property-analysis')], 
                            style={'width': '50%', 'display': 'inline-block'})
                ], style={'marginBottom': '20px'}),
                
                # Row 4: Red/Blue Team Intelligence
                html.Div([
                    html.Div(id='red-blue-intelligence', style={'width': '100%'})
                ], style={'marginBottom': '20px'}),
                
                # Row 5: Research Timeline
                html.Div([
                    html.Div([dcc.Graph(id='research-timeline')], 
                            style={'width': '100%'})
                ], style={'marginBottom': '20px'})
                
            ], style={'backgroundColor': '#000000', 'padding': '20px'}),
            
            # Auto-refresh component
            dcc.Interval(id='interval-component', interval=2000, n_intervals=0)
            
        ], style={'backgroundColor': '#000000', 'fontFamily': 'Arial, sans-serif', 'minHeight': '100vh'})

    def setup_callbacks(self):
        @self.app.callback(
            [Output('threat-overview', 'figure'),
             Output('attack-success-matrix', 'figure'),
             Output('atlas-heatmap', 'figure'),
             Output('mia-analysis', 'figure'),
             Output('gradient-analysis', 'figure'),
             Output('extraction-analysis', 'figure'),
             Output('property-analysis', 'figure'),
             Output('red-blue-intelligence', 'children'),
             Output('research-timeline', 'figure'),
             Output('interval-component', 'interval')],
            [Input('interval-component', 'n_intervals'),
             Input('interval-selector', 'value'),
             Input('threat-filter', 'value')]
        )
        def update_dashboard(n, interval, threat_filter):
            return (
                self.create_threat_overview(),
                self.create_attack_success_matrix(),
                self.create_atlas_heatmap(),
                self.create_mia_analysis(),
                self.create_gradient_analysis(),
                self.create_extraction_analysis(),
                self.create_property_analysis(),
                self.create_red_blue_intelligence(),
                self.create_research_timeline(),
                interval
            )

    def create_threat_overview(self):
        categories = ['Active Campaigns', 'New CVEs', 'Attack Vectors', 'Compromised Models']
        values = [
            self.threat_metrics['active_campaigns'],
            self.threat_metrics['new_cves'],
            self.threat_metrics['attack_vectors'],
            self.threat_metrics['compromised_models']
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=['#ff4444', '#ff8800', '#ffaa00', '#ff6600'],
            text=values,
            textposition='auto',
            textfont=dict(color='white', size=14)
        ))
        
        fig.update_layout(
            title="🚨 AI Threat Intelligence Overview",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font_color='#ffffff',
            title_font_color='#ff4444',
            height=350
        )
        
        return fig

    def create_attack_success_matrix(self):
        attacks = []
        success_rates = []
        
        for category, techniques in RESEARCH_2025.items():
            for technique, data in techniques.items():
                attacks.append(technique.split('(')[0][:15])
                if 'auc' in data:
                    success_rates.append(data['auc'] * 100)
                elif 'success' in data:
                    success_rates.append(data['success'] * 100)
                elif 'fidelity' in data:
                    success_rates.append(data['fidelity'] * 100)
                else:
                    success_rates.append(data['accuracy'] * 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=attacks,
            y=success_rates,
            mode='markers+lines',
            marker=dict(
                size=12,
                color=success_rates,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Success Rate %", titlefont=dict(color='white'))
            ),
            line=dict(color='#ff4444', width=2)
        ))
        
        fig.update_layout(
            title="📈 Attack Success Rates (2025 Research)",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font_color='#ffffff',
            title_font_color='#ff4444',
            height=350,
            xaxis_tickangle=-45
        )
        
        return fig

    def create_atlas_heatmap(self):
        techniques = list(ATLAS_TECHNIQUES.keys())
        names = [ATLAS_TECHNIQUES[t]['name'] for t in techniques]
        severities = [ATLAS_TECHNIQUES[t]['severity'] for t in techniques]
        detection_rates = [ATLAS_TECHNIQUES[t]['detection'] for t in techniques]
        
        severity_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        severity_numeric = [severity_map[s] for s in severities]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=detection_rates,
            y=severity_numeric,
            mode='markers+text',
            marker=dict(
                size=15,
                color=severity_numeric,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Severity", titlefont=dict(color='white'))
            ),
            text=techniques,
            hovertext=[f"{tech}: {name}" for tech, name in zip(techniques, names)],
            hoverinfo="text+x+y",
            textposition='middle center',
            textfont=dict(size=8, color='white')
        ))
        
        fig.update_layout(
            title="🎯 MITRE ATLAS Technique Matrix",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font_color='#ffffff',
            title_font_color='#ff4444',
            height=350,
            xaxis_title="Detection Rate",
            yaxis_title="Severity Level",
            yaxis=dict(tickvals=[1,2,3,4], ticktext=['LOW','MEDIUM','HIGH','CRITICAL'])
        )
        
        return fig

    def create_mia_analysis(self):
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Membership Inference Success Timeline', 'Defense Effectiveness'),
            vertical_spacing=0.3
        )
        
        if self.mia_timeline:
            times = list(range(len(self.mia_timeline)))
            fig.add_trace(go.Scatter(
                x=times,
                y=list(self.mia_timeline),
                mode='lines+markers',
                line=dict(color='#ff4444', width=2),
                marker=dict(size=6),
                name='MIA Success'
            ), row=1, col=1)
        
        defenses = ['Differential Privacy', 'Gradient Noise', 'Data Obfuscation', 'Regularization']
        effectiveness = [0.85, 0.72, 0.68, 0.59]
        
        fig.add_trace(go.Bar(
            x=defenses,
            y=effectiveness,
            marker_color='#00ff00',
            name='Defense Rate'
        ), row=2, col=1)
        
        fig.update_layout(
            title="🔍 Membership Inference Attack Analysis",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font_color='#ffffff',
            title_font_color='#ff4444',
            height=500,
            showlegend=False
        )
        
        return fig

    def create_gradient_analysis(self):
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Gradient Leakage Risk Timeline', 'Reconstruction Fidelity'),
            vertical_spacing=0.3
        )
        
        if self.gradient_timeline:
            times = list(range(len(self.gradient_timeline)))
            fig.add_trace(go.Scatter(
                x=times,
                y=list(self.gradient_timeline),
                mode='lines+markers',
                line=dict(color='#ff8800', width=2),
                marker=dict(size=6),
                fill='tonexty',
                name='Leakage Risk'
            ), row=1, col=1)
        
        attacks = ['Deep Gradient Inversion', 'Batch Reconstruction', 'FL Leakage']
        fidelity = [0.96, 0.91, 0.87]
        
        fig.add_trace(go.Bar(
            x=attacks,
            y=fidelity,
            marker_color='#ff8800',
            name='Fidelity'
        ), row=2, col=1)
        
        fig.update_layout(
            title="🔥 Gradient Leakage Attack Analysis",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font_color='#ffffff',
            title_font_color='#ff8800',
            height=500,
            showlegend=False
        )
        
        return fig

    def create_extraction_analysis(self):
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Model Extraction Success Timeline', 'Query Efficiency'),
            vertical_spacing=0.3
        )
        
        if self.extraction_timeline:
            times = list(range(len(self.extraction_timeline)))
            fig.add_trace(go.Scatter(
                x=times,
                y=list(self.extraction_timeline),
                mode='lines+markers',
                line=dict(color='#aa00ff', width=2),
                marker=dict(size=6),
                name='Extraction Rate'
            ), row=1, col=1)
        
        methods = ['Active Learning', 'Random Queries', 'Adversarial Queries', 'Hybrid']
        efficiency = [0.92, 0.45, 0.78, 0.89]
        
        fig.add_trace(go.Bar(
            x=methods,
            y=efficiency,
            marker_color='#aa00ff',
            name='Efficiency'
        ), row=2, col=1)
        
        fig.update_layout(
            title="🎯 Model Extraction Attack Analysis",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font_color='#ffffff',
            title_font_color='#aa00ff',
            height=500,
            showlegend=False
        )
        
        return fig

    def create_property_analysis(self):
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Property Inference Accuracy Timeline', 'Attribute Detection'),
            vertical_spacing=0.3
        )
        
        if self.property_timeline:
            times = list(range(len(self.property_timeline)))
            fig.add_trace(go.Scatter(
                x=times,
                y=list(self.property_timeline),
                mode='lines+markers',
                line=dict(color='#00aaff', width=2),
                marker=dict(size=6),
                name='Inference Accuracy'
            ), row=1, col=1)
        
        attributes = ['Demographics', 'Medical Info', 'Financial Data', 'Behavioral']
        detection = [0.88, 0.93, 0.76, 0.82]
        
        fig.add_trace(go.Bar(
            x=attributes,
            y=detection,
            marker_color='#00aaff',
            name='Detection Rate'
        ), row=2, col=1)
        
        fig.update_layout(
            title="🔎 Property Inference Attack Analysis",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font_color='#ffffff',
            title_font_color='#00aaff',
            height=500,
            showlegend=False
        )
        
        return fig

    def create_red_blue_intelligence(self):
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Select random research for current focus
        all_techniques = []
        for category, techniques in RESEARCH_2025.items():
            for technique, data in techniques.items():
                all_techniques.append((technique, data, category))
        
        selected = random.choice(all_techniques)
        technique_name, technique_data, category = selected
        
        return html.Div([
            html.H2("🎯 Red/Blue Team Actionable Intelligence", 
                   style={'color': '#ffffff', 'textAlign': 'center', 'marginBottom': '30px'}),
            
            html.Div([
                # Red Team Operations
                html.Div([
                    html.H3("🔴 Red Team Operations", style={'color': '#ff4444', 'marginBottom': '15px'}),
                    html.Div([
                        html.H4(f"Target: {technique_name.split('(')[0]}", 
                               style={'color': '#ffffff', 'fontSize': '16px', 'marginBottom': '10px'}),
                        html.P(f"Category: {category}", 
                              style={'color': '#ffaaaa', 'fontSize': '14px', 'marginBottom': '10px'}),
                        html.P(technique_data['description'], 
                              style={'color': '#ffffff', 'fontSize': '13px', 'marginBottom': '15px'}),
                        html.Div([
                            html.H5("🎯 Attack Vector:", style={'color': '#ff6666', 'marginBottom': '8px'}),
                            html.P(technique_data['red_action'], 
                                  style={'color': '#ffcccc', 'fontSize': '12px', 'fontStyle': 'italic'})
                        ], style={'backgroundColor': '#2a1a1a', 'padding': '12px', 'borderRadius': '5px', 'marginBottom': '10px'}),
                        html.P(f"Threat Level: {technique_data['threat']}", 
                              style={'color': '#ff4444', 'fontSize': '14px', 'fontWeight': 'bold'})
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                
                # Blue Team Defense
                html.Div([
                    html.H3("🔵 Blue Team Defense", style={'color': '#4444ff', 'marginBottom': '15px'}),
                    html.Div([
                        html.H4(f"Defend: {technique_name.split('(')[0]}", 
                               style={'color': '#ffffff', 'fontSize': '16px', 'marginBottom': '10px'}),
                        html.P(f"Priority: {technique_data['threat']}", 
                              style={'color': '#aaaaff', 'fontSize': '14px', 'marginBottom': '10px'}),
                        html.P(technique_data['description'], 
                              style={'color': '#ffffff', 'fontSize': '13px', 'marginBottom': '15px'}),
                        html.Div([
                            html.H5("🛡️ Mitigation Strategy:", style={'color': '#6666ff', 'marginBottom': '8px'}),
                            html.P(technique_data['blue_action'], 
                                  style={'color': '#ccccff', 'fontSize': '12px', 'fontStyle': 'italic'})
                        ], style={'backgroundColor': '#1a1a2a', 'padding': '12px', 'borderRadius': '5px', 'marginBottom': '10px'}),
                        html.P(f"Defense Priority: {technique_data['threat']}", 
                              style={'color': '#4444ff', 'fontSize': '14px', 'fontWeight': 'bold'})
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
            ], style={'marginBottom': '25px'}),
            
            # Research Reference
            html.Div([
                html.H4("📚 Research Reference & Metrics", style={'color': '#ffffff', 'textAlign': 'center', 'marginBottom': '15px'}),
                html.Div([
                    html.Div([
                        html.P("Success Rate:", style={'color': '#aaaaaa', 'fontSize': '12px', 'margin': '0'}),
                        success_val := next((f"{v*100:.1f}%" for k, v in technique_data.items() if k in ['auc', 'success', 'fidelity', 'accuracy']), "N/A"),
                        html.P(success_val, style={'color': '#ffffff', 'fontSize': '16px', 'fontWeight': 'bold', 'margin': '0'})
                    ], style={'display': 'inline-block', 'marginRight': '30px'}),
                    html.Div([
                        html.P("Threat Level:", style={'color': '#aaaaaa', 'fontSize': '12px', 'margin': '0'}),
                        html.P(technique_data['threat'], style={'color': '#ff4444', 'fontSize': '16px', 'fontWeight': 'bold', 'margin': '0'})
                    ], style={'display': 'inline-block', 'marginRight': '30px'}),
                    html.Div([
                        html.P("Last Updated:", style={'color': '#aaaaaa', 'fontSize': '12px', 'margin': '0'}),
                        html.P(current_time, style={'color': '#ffffff', 'fontSize': '16px', 'fontWeight': 'bold', 'margin': '0'})
                    ], style={'display': 'inline-block'})
                ], style={'textAlign': 'center'})
            ], style={'backgroundColor': '#1a1a1a', 'padding': '15px', 'borderRadius': '5px'})
        ], style={'backgroundColor': '#0a0a0a', 'padding': '25px', 'borderRadius': '10px'})

    def create_research_timeline(self):
        research_dates = [
            datetime.now() - timedelta(days=i*5) for i in range(8, 0, -1)
        ]
        
        research_papers = [
            "Gradient-based MINT",
            "Adversarial Iterations MIA", 
            "SOFT Defense Bypass",
            "Enhanced DLG Research",
            "FL Extraction Methods",
            "Distribution Inference",
            "Query-Efficient Extraction",
            "Batch Reconstruction"
        ]
        
        impact_scores = [99, 94, 87, 96, 89, 93, 92, 91]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=research_dates,
            y=impact_scores,
            mode='markers+lines+text',
            marker=dict(
                size=12,
                color=impact_scores,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Impact Score", titlefont=dict(color='white'))
            ),
            line=dict(color='#ffffff', width=2),
            text=research_papers,
            textposition='top center',
            textfont=dict(size=9, color='white')
        ))
        
        fig.update_layout(
            title="📚 AI Security Research Timeline (2025)",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font_color='#ffffff',
            title_font_color='#ffffff',
            height=400,
            xaxis_title="Publication Date",
            yaxis_title="Impact Score"
        )
        
        return fig

    def start_data_generation(self):
        def generate_data():
            while True:
                # Update real-time attack timelines
                self.mia_timeline.append(random.uniform(0.75, 0.99))
                self.gradient_timeline.append(random.uniform(0.65, 0.96))
                self.extraction_timeline.append(random.uniform(0.55, 0.92))
                self.property_timeline.append(random.uniform(0.65, 0.93))
                
                # Update threat intelligence metrics
                self.threat_metrics['active_campaigns'] = random.randint(18, 47)
                self.threat_metrics['new_cves'] = random.randint(6, 23)
                self.threat_metrics['attack_vectors'] = random.randint(9, 28)
                self.threat_metrics['compromised_models'] = random.randint(4, 18)
                
                time.sleep(2)
        
        thread = threading.Thread(target=generate_data, daemon=True)
        thread.start()

    def run(self, host='127.0.0.1', port=8117, debug: bool = False):
        print("🔒 Starting AI Security Research Dashboard 2025...")
        print(f"🌐 Dashboard URL: http://{host}:{port}")
        print("📊 Features: Latest 2025 research, Red/Blue team intelligence, Real-time monitoring")
        print("🎯 Coverage: MIA, Gradient Leakage, Model Extraction, Property Inference")
        print("🛡️ MITRE ATLAS integration with actionable defense strategies")
        if not self.allow_simulated:
            print("ℹ️ Running in REAL-DATA ONLY mode (no synthetic telemetry).")
        
        self.app.run(host=host, port=port, debug=bool(debug))

if __name__ == '__main__':
    # When run as a standalone script, enable simulated demo mode by default so the UI is populated.
    dashboard = AISecurityResearchDashboard(allow_simulated=True)

    # Lazy-load enhanced upload integration (optional dependency) at runtime.
    try:
        _eui = importlib.import_module("enhanced_upload_integration")
        integrate_enhanced_upload = getattr(_eui, "integrate_enhanced_upload", None)
        ENHANCED_UPLOAD_AVAILABLE = integrate_enhanced_upload is not None
    except Exception as e:
        ENHANCED_UPLOAD_AVAILABLE = False
        integrate_enhanced_upload = None  # type: ignore
        logger.debug("Enhanced upload system not available (%s).", e)

    # Optional: register upload callbacks if available; layout integration is still manual.
    if ENHANCED_UPLOAD_AVAILABLE and integrate_enhanced_upload is not None:
        try:
            dashboard.upload_section = integrate_enhanced_upload(dashboard.app)
            logger.info("Enhanced upload callbacks registered (layout integration still manual).")
        except Exception as e:
            logger.warning("Enhanced upload integration failed (%s).", e)

    dashboard.run()
