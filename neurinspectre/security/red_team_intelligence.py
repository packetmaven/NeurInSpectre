"""
NeurInSpectre: Red Team Operational Intelligence Tools
Enhanced with Advanced Offensive Security Analytics (2025)

This module implements comprehensive red team operational intelligence tools for
gradient obfuscation attack planning, vulnerability assessment, and penetration testing
analytics, incorporating latest research in offensive security operations.

Research Integration:
- Advanced attack vector analysis and planning
- Vulnerability exploitation effectiveness metrics
- Penetration testing campaign management
- Offensive security operations intelligence
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


# Avoid process-global warning suppression at import time. If you want to silence warnings,
# do it in your application entry point.

class RedTeamIntelligenceEngine:
    """
    Advanced red team operational intelligence engine for offensive cybersecurity operations.
    Implements attack planning, vulnerability assessment, and penetration testing analytics.
    """
    
    def __init__(self):
        """Initialize red team intelligence engine."""
        self.attack_database = []
        self.vulnerability_database = []
        self.campaign_history = []
        self.effectiveness_metrics = {}
        
        # Red team color scheme
        self.red_colors = {
            'primary': '#d62728',
            'secondary': '#ff7f7f',
            'success': '#ff4500',
            'warning': '#ffa500',
            'danger': '#8b0000',
            'info': '#dc143c'
        }
        
        # Attack complexity levels
        self.complexity_levels = {
            'trivial': 1,
            'simple': 2,
            'moderate': 3,
            'complex': 4,
            'advanced': 5
        }

    def _as_dataframe(self, data) -> pd.DataFrame:
        """Best-effort conversion to a pandas DataFrame (never raises)."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()
        
    def create_attack_planning_dashboard(self, target_data, attack_vectors):
        """
        Create comprehensive attack planning dashboard for red team operations.
        Implements advanced attack vector analysis and campaign planning.
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "Target Attack Surface", "Attack Vector Effectiveness", "Complexity vs Success Rate",
                "Gradient Obfuscation Techniques", "Evasion Probability Matrix", "Resource Requirements",
                "Timeline Planning", "Risk Assessment", "Success Prediction"
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "pie"}, {"type": "indicator"}]
            ]
        )
        
        # 1. Target Attack Surface Analysis
        attack_surface = pd.Series(target_data['attack_surface']).value_counts()
        
        fig.add_trace(
            go.Bar(
                x=attack_surface.index,
                y=attack_surface.values,
                marker_color=self.red_colors['primary'],
                name="Attack Surface"
            ),
            row=1, col=1
        )
        
        # 2. Attack Vector Effectiveness vs Difficulty
        effectiveness = attack_vectors['effectiveness_score']
        difficulty = attack_vectors['difficulty_score']
        vector_names = attack_vectors['vector_name']
        
        fig.add_trace(
            go.Scatter(
                x=difficulty,
                y=effectiveness,
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=effectiveness,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Effectiveness")
                ),
                text=vector_names,
                textposition='top center',
                name="Attack Vectors",
                hovertemplate='Vector: %{text}<br>Difficulty: %{x}<br>Effectiveness: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Complexity vs Success Rate Analysis
        complexity_scores = [self.complexity_levels.get(c, 3) for c in attack_vectors['complexity']]
        success_rates = attack_vectors['historical_success_rate']
        
        fig.add_trace(
            go.Scatter(
                x=complexity_scores,
                y=success_rates,
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.red_colors['danger'],
                    opacity=0.7
                ),
                name="Complexity vs Success",
                hovertemplate='Complexity: %{x}<br>Success Rate: %{y:.1%}<extra></extra>'
            ),
            row=1, col=3
        )
        
        # 4. Gradient Obfuscation Techniques Heatmap
        obfuscation_techniques = ['RL-Obfuscation', 'Cross-Modal', 'Temporal-Evolution', 'Gradient-Masking', 'Stochastic']
        target_types = ['CNN', 'RNN', 'Transformer', 'ResNet', 'BERT']

        # Derive technique effectiveness from attack_vectors when possible (no random placeholders).
        av = self._as_dataframe(attack_vectors)
        eff_map = {}
        if not av.empty and 'vector_name' in av.columns and 'effectiveness_score' in av.columns:
            try:
                eff_map = {
                    str(n): float(s)
                    for n, s in zip(av['vector_name'].tolist(), av['effectiveness_score'].tolist())
                }
            except Exception:
                eff_map = {}

        name_alias = {
            'Temporal-Evolution': ['Temporal-Evo', 'Temporal Evolution', 'Temporal_Evo'],
            'Gradient-Masking': ['Gradient-Mask', 'Gradient Masking', 'Gradient_Mask'],
        }
        obfuscation_matrix = []
        missing = []
        for tech in obfuscation_techniques:
            score = eff_map.get(tech)
            if score is None:
                for alt in name_alias.get(tech, []):
                    if alt in eff_map:
                        score = eff_map[alt]
                        break
            if score is None:
                score = 0.0
                missing.append(tech)
            # Accept either [0,1] or [0,100]
            if np.isfinite(score) and score > 1.0:
                score = score / 100.0
            score = float(np.clip(score if np.isfinite(score) else 0.0, 0.0, 1.0))
            obfuscation_matrix.append([score] * len(target_types))
        obfuscation_matrix = np.asarray(obfuscation_matrix, dtype=np.float64)
        
        fig.add_trace(
            go.Heatmap(
                z=obfuscation_matrix,
                x=target_types,
                y=obfuscation_techniques,
                colorscale='Reds',
                colorbar=dict(title="Effectiveness"),
                hovertemplate='Technique: %{y}<br>Target: %{x}<br>Effectiveness: %{z:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

        if missing:
            fig.add_annotation(
                text="Missing effectiveness_score for: " + ", ".join(missing) + " (set to 0).",
                showarrow=False,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.02,
                font=dict(size=9, color="#444"),
                row=2,
                col=1,
            )
        
        # 5. Evasion Probability Matrix
        defense_mechanisms = ['Gradient Clipping', 'Noise Addition', 'Differential Privacy', 'Adversarial Training']
        # No defensible way to infer defense-specific evasion probabilities from the inputs alone.
        # If you have a measured/provided matrix, pass it as attack_vectors['evasion_matrix'] (dict or ndarray).
        ev = None
        if isinstance(attack_vectors, dict):
            ev = attack_vectors.get('evasion_matrix')
        if ev is not None:
            try:
                evasion_matrix = np.asarray(ev, dtype=np.float64)
                fig.add_trace(
                    go.Heatmap(
                        z=evasion_matrix,
                        x=defense_mechanisms,
                        y=obfuscation_techniques,
                        colorscale='Oranges',
                        colorbar=dict(title="Evasion Probability"),
                        hovertemplate='Technique: %{y}<br>Defense: %{x}<br>Evasion Prob: %{z:.2f}<extra></extra>'
                    ),
                    row=2, col=2
                )
            except Exception:
                fig.add_annotation(
                    text="evasion_matrix provided but could not be parsed.",
                    showarrow=False,
                    xref="x domain",
                    yref="y domain",
                    x=0.5,
                    y=0.5,
                    font=dict(size=10, color="#444"),
                    row=2,
                    col=2,
                )
        else:
            fig.add_annotation(
                text="No evasion_matrix provided; not fabricating probabilities.",
                showarrow=False,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                font=dict(size=10, color="#444"),
                row=2,
                col=2,
            )
        
        # 6. Resource Requirements Analysis
        resource_types = ['Time (hours)', 'Compute (GPU-hours)', 'Storage (GB)', 'Personnel']
        resource_requirements = [24, 48, 100, 3]  # Example requirements
        
        fig.add_trace(
            go.Bar(
                x=resource_types,
                y=resource_requirements,
                marker_color=self.red_colors['warning'],
                name="Resource Requirements"
            ),
            row=2, col=3
        )
        
        # 7. Attack Timeline Planning
        timeline_phases = ['Reconnaissance', 'Initial Access', 'Persistence', 'Privilege Escalation', 'Exfiltration']
        phase_durations = [2, 4, 3, 6, 2]  # Hours for each phase
        cumulative_time = np.cumsum([0] + phase_durations[:-1])
        
        fig.add_trace(
            go.Scatter(
                x=cumulative_time,
                y=timeline_phases[:-1],
                mode='markers+lines',
                marker=dict(size=10, color=self.red_colors['primary']),
                line=dict(color=self.red_colors['primary'], width=3),
                name="Attack Timeline",
                hovertemplate='Phase: %{y}<br>Start Time: %{x}h<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 8. Risk Assessment Distribution
        risk_categories = ['Detection Risk', 'Attribution Risk', 'Technical Risk', 'Legal Risk']
        risk_levels = [0.3, 0.2, 0.4, 0.1]  # Risk probabilities
        
        fig.add_trace(
            go.Pie(
                labels=risk_categories,
                values=risk_levels,
                marker_colors=['#ff6b6b', '#ffa500', '#ff4500', '#8b0000'],
                name="Risk Assessment"
            ),
            row=3, col=2
        )
        
        # 9. Overall Success Prediction
        # Calculate weighted success probability
        vector_weights = effectiveness / np.sum(effectiveness)
        predicted_success = np.sum(vector_weights * success_rates) * 100
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=predicted_success,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Success Rate (%)"},
                delta={'reference': 70, 'relative': False},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.red_colors['primary']},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "darkred", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=3, col=3
        )
        
        fig.update_layout(
            title="Red Team Attack Planning Dashboard - Offensive Intelligence",
            height=1000,
            template="plotly_dark",
            font=dict(size=10),
            showlegend=True
        )
        
        return fig
    
    def create_vulnerability_assessment_dashboard(self, vulnerability_data, exploit_data):
        """
        Create comprehensive vulnerability assessment dashboard.
        Implements vulnerability prioritization and exploit effectiveness analysis.
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Vulnerability Severity Distribution", "Exploit Availability Timeline", "CVSS Score Analysis",
                "Attack Vector Mapping", "Exploitation Difficulty", "Patch Status Tracking"
            ],
            specs=[
                [{"type": "pie"}, {"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # 1. Vulnerability Severity Distribution
        severity_counts = pd.Series(vulnerability_data['severity']).value_counts()
        severity_colors = ['#8b0000', '#d62728', '#ff7f0e', '#ffbb78']
        
        fig.add_trace(
            go.Pie(
                labels=severity_counts.index,
                values=severity_counts.values,
                marker_colors=severity_colors,
                name="Severity Distribution"
            ),
            row=1, col=1
        )
        
        # 2. Exploit Availability Timeline
        exploit_timeline = pd.DataFrame(exploit_data)
        exploit_timeline['discovery_date'] = pd.to_datetime(exploit_timeline['discovery_date'])
        exploit_timeline['exploit_date'] = pd.to_datetime(exploit_timeline['exploit_date'])
        
        # Calculate time to exploit
        time_to_exploit = (exploit_timeline['exploit_date'] - exploit_timeline['discovery_date']).dt.days
        
        fig.add_trace(
            go.Scatter(
                x=exploit_timeline['discovery_date'],
                y=time_to_exploit,
                mode='markers',
                marker=dict(
                    size=8,
                    color=exploit_timeline['cvss_score'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="CVSS Score")
                ),
                name="Time to Exploit",
                hovertemplate='Discovery: %{x}<br>Days to Exploit: %{y}<br>CVSS: %{marker.color}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. CVSS Score Distribution
        cvss_scores = vulnerability_data['cvss_score']
        
        fig.add_trace(
            go.Histogram(
                x=cvss_scores,
                nbinsx=20,
                marker_color=self.red_colors['primary'],
                opacity=0.7,
                name="CVSS Distribution"
            ),
            row=1, col=3
        )
        
        # 4. Attack Vector Mapping
        attack_vectors = pd.Series(vulnerability_data['attack_vector']).value_counts()
        
        fig.add_trace(
            go.Bar(
                x=attack_vectors.index,
                y=attack_vectors.values,
                marker_color=self.red_colors['danger'],
                name="Attack Vectors"
            ),
            row=2, col=1
        )
        
        # 5. Exploitation Difficulty vs Impact
        difficulty = vulnerability_data['exploitation_difficulty']
        impact = vulnerability_data['impact_score']
        
        fig.add_trace(
            go.Scatter(
                x=difficulty,
                y=impact,
                mode='markers',
                marker=dict(
                    size=10,
                    color=cvss_scores,
                    colorscale='Reds',
                    opacity=0.7
                ),
                name="Difficulty vs Impact",
                hovertemplate='Difficulty: %{x}<br>Impact: %{y}<br>CVSS: %{marker.color}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 6. Patch Status Tracking
        patch_status = pd.Series(vulnerability_data['patch_status']).value_counts()
        patch_colors = ['green', 'orange', 'red', 'gray']
        
        fig.add_trace(
            go.Bar(
                x=patch_status.index,
                y=patch_status.values,
                marker_color=patch_colors[:len(patch_status)],
                name="Patch Status"
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title="Red Team Vulnerability Assessment Dashboard - Exploit Intelligence",
            height=800,
            template="plotly_dark",
            font=dict(size=10),
            showlegend=True
        )
        
        return fig
    
    def create_penetration_testing_analytics(self, pentest_data, campaign_metrics):
        """
        Create comprehensive penetration testing analytics dashboard.
        Implements campaign effectiveness tracking and methodology analysis.
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Campaign Success Rates", "Methodology Effectiveness", "Time to Compromise",
                "Detection Evasion Analysis", "Lateral Movement Paths", "Objective Achievement"
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "pie"}]
            ]
        )
        
        # 1. Campaign Success Rates by Type
        campaign_types = pd.Series(pentest_data['campaign_type']).value_counts()
        success_rates = []
        
        for campaign_type in campaign_types.index:
            type_data = pentest_data[pentest_data['campaign_type'] == campaign_type]
            # Treat any truthy values as success; avoid `== True` comparisons.
            success_rate = float(type_data['success'].astype(bool).mean() * 100.0)
            success_rates.append(success_rate)
        
        fig.add_trace(
            go.Bar(
                x=campaign_types.index,
                y=success_rates,
                marker_color=self.red_colors['primary'],
                name="Success Rates"
            ),
            row=1, col=1
        )
        
        # 2. Methodology Effectiveness
        methodologies = ['Social Engineering', 'Network Penetration', 'Web Application', 'Physical Security', 'Wireless']
        effectiveness_scores = [85, 78, 92, 65, 73]  # Example effectiveness percentages
        
        fig.add_trace(
            go.Bar(
                x=methodologies,
                y=effectiveness_scores,
                marker_color=self.red_colors['warning'],
                name="Methodology Effectiveness"
            ),
            row=1, col=2
        )
        
        # 3. Time to Compromise Distribution
        time_to_compromise = pentest_data['time_to_compromise_hours']
        
        fig.add_trace(
            go.Histogram(
                x=time_to_compromise,
                nbinsx=15,
                marker_color=self.red_colors['danger'],
                opacity=0.7,
                name="Time to Compromise"
            ),
            row=1, col=3
        )
        
        # 4. Detection Evasion vs Stealth Level
        stealth_level = pentest_data['stealth_level']
        detection_rate = pentest_data['detection_rate']
        
        fig.add_trace(
            go.Scatter(
                x=stealth_level,
                y=detection_rate,
                mode='markers',
                marker=dict(
                    size=8,
                    color=time_to_compromise,
                    colorscale='Reds_r',
                    showscale=True,
                    colorbar=dict(title="Time to Compromise")
                ),
                name="Evasion Analysis",
                hovertemplate='Stealth: %{x}<br>Detection Rate: %{y:.1%}<br>Time: %{marker.color}h<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 5. Lateral Movement Analysis
        initial_access = pentest_data['initial_access_method']
        lateral_moves = pentest_data['lateral_movement_count']
        
        fig.add_trace(
            go.Scatter(
                x=initial_access,
                y=lateral_moves,
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.red_colors['info'],
                    opacity=0.7
                ),
                name="Lateral Movement",
                hovertemplate='Access Method: %{x}<br>Lateral Moves: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 6. Objective Achievement Distribution
        objectives_achieved = pd.Series(pentest_data['objectives_achieved']).value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=[f"{obj} Objectives" for obj in objectives_achieved.index],
                values=objectives_achieved.values,
                marker_colors=px.colors.sequential.Reds_r,
                name="Objectives Achieved"
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title="Red Team Penetration Testing Analytics - Campaign Intelligence",
            height=800,
            template="plotly_dark",
            font=dict(size=10),
            showlegend=True
        )
        
        return fig
    
    def create_offensive_operations_dashboard(self, operations_data, target_intelligence):
        """
        Create comprehensive offensive operations dashboard.
        Implements operational security and target intelligence analysis.
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "Operation Status Overview", "Target Intelligence Map", "OPSEC Risk Assessment",
                "Attack Chain Analysis", "Resource Utilization", "Countermeasure Effectiveness",
                "Intelligence Gathering", "Operational Timeline", "Mission Success Metrics"
            ],
            specs=[
                [{"type": "pie"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "indicator"}]
            ]
        )
        
        # 1. Operation Status Overview
        operation_status = pd.Series(operations_data['status']).value_counts()
        status_colors = ['green', 'orange', 'red', 'blue', 'gray']
        
        fig.add_trace(
            go.Pie(
                labels=operation_status.index,
                values=operation_status.values,
                marker_colors=status_colors[:len(operation_status)],
                name="Operation Status"
            ),
            row=1, col=1
        )
        
        # 2. Target Intelligence Map
        target_value = target_intelligence['intelligence_value']
        target_difficulty = target_intelligence['access_difficulty']
        target_names = target_intelligence['target_name']
        
        fig.add_trace(
            go.Scatter(
                x=target_difficulty,
                y=target_value,
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=target_value,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Intelligence Value")
                ),
                text=target_names,
                textposition='top center',
                name="Target Intelligence",
                hovertemplate='Target: %{text}<br>Difficulty: %{x}<br>Value: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. OPSEC Risk Assessment
        opsec_categories = ['Communication', 'Infrastructure', 'Personnel', 'Technical', 'Physical']
        risk_scores = [3, 2, 4, 3, 1]  # Risk levels 1-5
        
        fig.add_trace(
            go.Bar(
                x=opsec_categories,
                y=risk_scores,
                marker_color=['red' if score >= 4 else 'orange' if score >= 3 else 'green' for score in risk_scores],
                name="OPSEC Risk"
            ),
            row=1, col=3
        )
        
        # 4. Attack Chain Analysis
        attack_phases = ['Reconnaissance', 'Weaponization', 'Delivery', 'Exploitation', 'Installation', 'C2', 'Actions']
        success_rates = [95, 88, 82, 75, 70, 85, 78]  # Success rate for each phase
        
        fig.add_trace(
            go.Bar(
                x=attack_phases,
                y=success_rates,
                marker_color=self.red_colors['primary'],
                name="Attack Chain Success"
            ),
            row=2, col=1
        )
        
        # 5. Resource Utilization
        resource_categories = ['Human Resources', 'Technical Tools', 'Infrastructure', 'Time Investment']
        utilization_percentages = [75, 85, 60, 90]
        
        fig.add_trace(
            go.Bar(
                x=resource_categories,
                y=utilization_percentages,
                marker_color=self.red_colors['warning'],
                name="Resource Utilization"
            ),
            row=2, col=2
        )
        
        # 6. Countermeasure Effectiveness Matrix
        countermeasures = ['Firewall', 'IDS/IPS', 'EDR', 'SIEM', 'User Training']
        attack_types = ['Network', 'Endpoint', 'Social Eng', 'Web App']
        cm = None
        if isinstance(operations_data, dict):
            cm = operations_data.get('countermeasure_effectiveness_matrix')
        if cm is not None:
            try:
                effectiveness_matrix = np.asarray(cm, dtype=np.float64)
                fig.add_trace(
                    go.Heatmap(
                        z=effectiveness_matrix,
                        x=attack_types,
                        y=countermeasures,
                        colorscale='Reds_r',  # Inverted so red = less effective
                        colorbar=dict(title="Effectiveness"),
                        hovertemplate='Countermeasure: %{y}<br>Attack Type: %{x}<br>Effectiveness: %{z:.2f}<extra></extra>'
                    ),
                    row=2, col=3
                )
            except Exception:
                fig.add_annotation(
                    text="countermeasure_effectiveness_matrix provided but could not be parsed.",
                    showarrow=False,
                    xref="x domain",
                    yref="y domain",
                    x=0.5,
                    y=0.5,
                    font=dict(size=10, color="#ddd"),
                    row=2,
                    col=3,
                )
        else:
            fig.add_annotation(
                text="No countermeasure_effectiveness_matrix provided; not fabricating effectiveness.",
                showarrow=False,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                font=dict(size=10, color="#ddd"),
                row=2,
                col=3,
            )
        
        # 7. Intelligence Gathering Metrics
        intel_sources = ['OSINT', 'HUMINT', 'SIGINT', 'Technical Recon', 'Social Media']
        intel_quality = [85, 70, 90, 95, 75]  # Quality scores
        
        fig.add_trace(
            go.Bar(
                x=intel_sources,
                y=intel_quality,
                marker_color=self.red_colors['info'],
                name="Intelligence Quality"
            ),
            row=3, col=1
        )
        
        # 8. Operational Timeline
        # Operational timeline should come from real operations events (timestamps). If not present, don't fabricate.
        op_df = self._as_dataframe(operations_data)
        if not op_df.empty and 'timestamp' in op_df.columns:
            try:
                ts = pd.to_datetime(op_df['timestamp'], errors='coerce').dropna()
                s = ts.dt.floor('1D').value_counts().sort_index()
                fig.add_trace(
                    go.Scatter(
                        x=s.index,
                        y=s.values,
                        mode='lines+markers',
                        line=dict(color=self.red_colors['primary'], width=2),
                        marker=dict(size=6),
                        name="Ops events/day"
                    ),
                    row=3, col=2
                )
            except Exception:
                fig.add_annotation(
                    text="Operations timestamps present but could not be aggregated.",
                    showarrow=False,
                    xref="x domain",
                    yref="y domain",
                    x=0.5,
                    y=0.5,
                    font=dict(size=10, color="#ddd"),
                    row=3,
                    col=2,
                )
        else:
            fig.add_annotation(
                text="No operations timestamps provided; operational intensity not plotted.",
                showarrow=False,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                font=dict(size=10, color="#ddd"),
                row=3,
                col=2,
            )
        
        # 9. Mission Success Metrics
        # Calculate overall mission success score
        mission_success = np.mean([
            np.mean(success_rates) / 100,  # Attack chain success
            (100 - np.mean(risk_scores) * 20) / 100,  # OPSEC (inverted)
            np.mean(utilization_percentages) / 100,  # Resource utilization
            np.mean(intel_quality) / 100  # Intelligence quality
        ]) * 100
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=mission_success,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Mission Success Score"},
                delta={'reference': 80, 'relative': False},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.red_colors['primary']},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "darkred", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=3, col=3
        )
        
        fig.update_layout(
            title="Red Team Offensive Operations Dashboard - Mission Intelligence",
            height=1200,
            template="plotly_dark",
            font=dict(size=9),
            showlegend=True
        )
        
        return fig

def demonstrate_red_team_intelligence():
    """Demonstrate the red team intelligence engine."""
    print("⚔️ NeurInSpectre Red Team Intelligence Engine Demo")
    print("=" * 60)
    
    # Initialize red team engine
    red_team = RedTeamIntelligenceEngine()
    
    # Generate synthetic data for demonstrations
    np.random.seed(42)
    
    # 1. Attack Planning Dashboard
    print("📊 Creating attack planning dashboard...")
    
    # Generate target data
    target_data = {
        'target_id': [f"TGT-{i:03d}" for i in range(1, 51)],
        'attack_surface': np.random.choice(['Network', 'Web App', 'Mobile', 'IoT', 'Cloud'], 50)
    }
    
    # Generate attack vector data
    attack_vectors = pd.DataFrame({
        'vector_name': ['RL-Obfuscation', 'Cross-Modal', 'Temporal-Evo', 'Gradient-Mask', 'Stochastic', 'Traditional'],
        'effectiveness_score': [0.92, 0.85, 0.78, 0.88, 0.75, 0.65],
        'difficulty_score': [4.5, 4.2, 3.8, 3.5, 3.2, 2.5],
        'complexity': ['advanced', 'advanced', 'complex', 'complex', 'moderate', 'simple'],
        'historical_success_rate': [0.85, 0.78, 0.82, 0.75, 0.88, 0.92]
    })
    
    fig1 = red_team.create_attack_planning_dashboard(target_data, attack_vectors)
    fig1.write_html("neurinspectre_attack_planning.html")
    print("   ✅ Saved: neurinspectre_attack_planning.html")
    
    # 2. Vulnerability Assessment Dashboard
    print("📊 Creating vulnerability assessment dashboard...")
    
    # Generate vulnerability data
    vulnerability_data = pd.DataFrame({
        'vuln_id': [f"CVE-2024-{i:05d}" for i in range(1, 101)],
        'severity': np.random.choice(['Critical', 'High', 'Medium', 'Low'], 100, p=[0.1, 0.3, 0.4, 0.2]),
        'cvss_score': np.random.uniform(1.0, 10.0, 100),
        'attack_vector': np.random.choice(['Network', 'Adjacent', 'Local', 'Physical'], 100),
        'exploitation_difficulty': np.random.uniform(1, 5, 100),
        'impact_score': np.random.uniform(1, 10, 100),
        'patch_status': np.random.choice(['Patched', 'Partial', 'Unpatched', 'Unknown'], 100)
    })
    
    # Generate exploit data
    exploit_data = []
    for i in range(50):
        discovery_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
        exploit_date = discovery_date + timedelta(days=np.random.randint(1, 180))
        exploit_data.append({
            'vuln_id': f"CVE-2024-{i:05d}",
            'discovery_date': discovery_date,
            'exploit_date': exploit_date,
            'cvss_score': np.random.uniform(5.0, 10.0)
        })
    
    fig2 = red_team.create_vulnerability_assessment_dashboard(vulnerability_data, exploit_data)
    fig2.write_html("neurinspectre_vulnerability_assessment.html")
    print("   ✅ Saved: neurinspectre_vulnerability_assessment.html")
    
    # 3. Penetration Testing Analytics
    print("📊 Creating penetration testing analytics...")
    
    # Generate pentest data
    pentest_data = pd.DataFrame({
        'campaign_id': [f"PT-{i:03d}" for i in range(1, 101)],
        'campaign_type': np.random.choice(['External', 'Internal', 'Wireless', 'Social Engineering'], 100),
        'success': np.random.choice([True, False], 100, p=[0.75, 0.25]),
        'time_to_compromise_hours': np.random.exponential(12, 100),
        'stealth_level': np.random.uniform(1, 5, 100),
        'detection_rate': np.random.uniform(0, 0.5, 100),
        'initial_access_method': np.random.choice(['Phishing', 'Exploit', 'Credential Stuffing', 'Physical'], 100),
        'lateral_movement_count': np.random.poisson(3, 100),
        'objectives_achieved': np.random.randint(1, 6, 100)
    })
    
    campaign_metrics = {
        'total_campaigns': len(pentest_data),
        'success_rate': pentest_data['success'].mean(),
        'avg_time_to_compromise': pentest_data['time_to_compromise_hours'].mean()
    }
    
    fig3 = red_team.create_penetration_testing_analytics(pentest_data, campaign_metrics)
    fig3.write_html("neurinspectre_penetration_testing.html")
    print("   ✅ Saved: neurinspectre_penetration_testing.html")
    
    # 4. Offensive Operations Dashboard
    print("📊 Creating offensive operations dashboard...")
    
    # Generate operations data
    operations_data = pd.DataFrame({
        'operation_id': [f"OP-{i:03d}" for i in range(1, 51)],
        'status': np.random.choice(['Active', 'Planning', 'Completed', 'On Hold', 'Cancelled'], 50),
        'priority': np.random.choice(['Critical', 'High', 'Medium', 'Low'], 50),
        'resource_allocation': np.random.uniform(0.3, 1.0, 50)
    })
    
    # Generate target intelligence data
    target_intelligence = pd.DataFrame({
        'target_name': [f"Target-{chr(65+i)}" for i in range(15)],
        'intelligence_value': np.random.uniform(3, 10, 15),
        'access_difficulty': np.random.uniform(1, 8, 15)
    })
    
    fig4 = red_team.create_offensive_operations_dashboard(operations_data, target_intelligence)
    fig4.write_html("neurinspectre_offensive_operations.html")
    print("   ✅ Saved: neurinspectre_offensive_operations.html")
    
    print("\n⚔️ Red Team Intelligence Engine Demo Complete!")
    print("📁 Generated Files:")
    print("   • neurinspectre_attack_planning.html")
    print("   • neurinspectre_vulnerability_assessment.html")
    print("   • neurinspectre_penetration_testing.html")
    print("   • neurinspectre_offensive_operations.html")
    
    return red_team

if __name__ == "__main__":
    red_team = demonstrate_red_team_intelligence()

