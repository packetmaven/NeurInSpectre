"""
NeurInSpectre: Blue Team Operational Intelligence Tools
Enhanced with AI-Native SOC and Defensive Analytics (2025)

This module implements comprehensive blue team operational intelligence tools for
gradient obfuscation defense, incorporating latest research in AI-powered cybersecurity
operations, incident response automation, and threat hunting analytics.

Research Integration:
- AI-Native SOC operational workflows (June 2025)
- Automated incident response and threat hunting (May 2025)
- Advanced behavioral analytics for defensive operations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Avoid process-global warning suppression at import time. If you want to silence warnings,
# do it in your application entry point.

class BlueTeamIntelligenceEngine:
    """
    Advanced blue team operational intelligence engine for defensive cybersecurity operations.
    Implements AI-Native SOC capabilities and automated threat response.
    """
    
    def __init__(self):
        """Initialize blue team intelligence engine."""
        self.threat_database = []
        self.incident_history = []
        self.defense_metrics = {}
        self.alert_thresholds = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
        
        # Blue team color scheme
        self.blue_colors = {
            'primary': '#1f77b4',
            'secondary': '#aec7e8', 
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'info': '#17becf'
        }

    def _as_dataframe(self, data) -> pd.DataFrame:
        """Best-effort conversion to a pandas DataFrame (never raises)."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()
        
    def create_incident_response_dashboard(self, incident_data, timeline_data):
        """
        Create comprehensive incident response dashboard for blue team operations.
        Implements AI-Native SOC incident management visualization.
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "Incident Timeline", "Severity Distribution", "Response Time Analysis",
                "Attack Vector Breakdown", "Mitigation Effectiveness", "Resource Allocation",
                "Threat Actor Profiling", "Recovery Metrics", "Lessons Learned"
            ],
            specs=[
                [{"colspan": 2}, None, {"type": "pie"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "indicator"}, {"type": "table"}]
            ]
        )
        
        # 1. Incident Timeline (spans 2 columns)
        timeline_df = pd.DataFrame(timeline_data)
        timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])
        
        # Create timeline with different phases
        phases = ['Detection', 'Analysis', 'Containment', 'Eradication', 'Recovery']
        phase_colors = ['red', 'orange', 'yellow', 'blue', 'green']
        
        for i, phase in enumerate(phases):
            phase_data = timeline_df[timeline_df['phase'] == phase]
            if not phase_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=phase_data['timestamp'],
                        y=[phase] * len(phase_data),
                        mode='markers+lines',
                        name=phase,
                        marker=dict(size=10, color=phase_colors[i]),
                        line=dict(color=phase_colors[i], width=3),
                        hovertemplate=f'Phase: {phase}<br>Time: %{{x}}<br>Action: %{{text}}<extra></extra>',
                        text=phase_data['action']
                    ),
                    row=1, col=1
                )
        
        # 2. Severity Distribution
        severity_counts = pd.Series(incident_data['severity']).value_counts()
        fig.add_trace(
            go.Pie(
                labels=severity_counts.index,
                values=severity_counts.values,
                marker_colors=['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c'],
                name="Severity Distribution"
            ),
            row=1, col=3
        )
        
        # 3. Attack Vector Breakdown
        attack_vectors = pd.Series(incident_data['attack_vector']).value_counts()
        fig.add_trace(
            go.Bar(
                x=attack_vectors.index,
                y=attack_vectors.values,
                marker_color=self.blue_colors['primary'],
                name="Attack Vectors"
            ),
            row=2, col=1
        )
        
        # 4. Response Time vs Severity
        response_times = incident_data['response_time_hours']
        severities = incident_data['severity']
        
        severity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        severity_numeric = [severity_map.get(s, 0) for s in severities]
        
        fig.add_trace(
            go.Scatter(
                x=severity_numeric,
                y=response_times,
                mode='markers',
                marker=dict(
                    size=8,
                    color=response_times,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Response Time (hrs)")
                ),
                name="Response Time Analysis",
                hovertemplate='Severity: %{text}<br>Response Time: %{y:.1f} hrs<extra></extra>',
                text=severities
            ),
            row=2, col=2
        )
        
        # 5. Mitigation Effectiveness
        mitigation_methods = ['Isolation', 'Patching', 'Rule Update', 'Monitoring', 'Training']
        effectiveness_scores = [85, 92, 78, 88, 75]  # Example effectiveness percentages
        
        fig.add_trace(
            go.Bar(
                x=mitigation_methods,
                y=effectiveness_scores,
                marker_color=self.blue_colors['success'],
                name="Mitigation Effectiveness"
            ),
            row=2, col=3
        )
        
        # 6. Threat Actor Profiling Heatmap
        threat_actors = ['APT-28', 'Lazarus', 'FIN7', 'Carbanak', 'Unknown']
        attack_techniques = ['RL-Obfuscation', 'Cross-Modal', 'Temporal-Evo', 'Gradient-Mask']

        # Data-driven activity matrix if incident_data includes actor/technique fields.
        incident_df = self._as_dataframe(incident_data)
        matrix_source = "defaults"
        if not incident_df.empty and 'threat_actor' in incident_df.columns and 'attack_technique' in incident_df.columns:
            # Use observed categories to avoid fabricating activity.
            threat_actors = sorted([str(x) for x in incident_df['threat_actor'].dropna().unique().tolist()])
            attack_techniques = sorted([str(x) for x in incident_df['attack_technique'].dropna().unique().tolist()])
            ct = pd.crosstab(incident_df['threat_actor'], incident_df['attack_technique'])
            activity_matrix = ct.reindex(index=threat_actors, columns=attack_techniques, fill_value=0).to_numpy()
            matrix_source = "observed_counts"
        else:
            activity_matrix = np.zeros((len(threat_actors), len(attack_techniques)), dtype=np.int64)
            matrix_source = "missing_fields"
        
        fig.add_trace(
            go.Heatmap(
                z=activity_matrix,
                x=attack_techniques,
                y=threat_actors,
                colorscale='Blues',
                colorbar=dict(title="Activity Level"),
                hovertemplate='Actor: %{y}<br>Technique: %{x}<br>Activity: %{z}<extra></extra>'
            ),
            row=3, col=1
        )

        if matrix_source != "observed_counts":
            fig.add_annotation(
                text="No threat_actor/attack_technique fields provided; showing empty matrix.",
                showarrow=False,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                font=dict(size=10, color="#444"),
                row=3,
                col=1,
            )
        
        # 7. Recovery Time Indicator
        avg_recovery_time = np.mean(incident_data['recovery_time_hours'])
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=avg_recovery_time,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg Recovery Time (hrs)"},
                delta={'reference': 24, 'relative': True},
                gauge={
                    'axis': {'range': [None, 72]},
                    'bar': {'color': self.blue_colors['primary']},
                    'steps': [
                        {'range': [0, 12], 'color': "lightgreen"},
                        {'range': [12, 24], 'color': "yellow"},
                        {'range': [24, 48], 'color': "orange"},
                        {'range': [48, 72], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 48
                    }
                }
            ),
            row=3, col=2
        )
        
        # 8. Lessons Learned Table
        lessons_data = [
            ['Faster Detection', 'Implement ML-based anomaly detection', 'High'],
            ['Better Isolation', 'Automated network segmentation', 'Medium'],
            ['Improved Training', 'Regular red team exercises', 'High'],
            ['Tool Integration', 'SOAR platform deployment', 'Medium']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Improvement Area', 'Action Item', 'Priority'],
                           fill_color=self.blue_colors['primary'],
                           font=dict(color='white')),
                cells=dict(values=list(zip(*lessons_data)),
                          fill_color='lightgray')
            ),
            row=3, col=3
        )
        
        fig.update_layout(
            title="Blue Team Incident Response Dashboard - AI-Native SOC",
            height=1000,
            template="plotly_white",
            font=dict(size=10),
            showlegend=True
        )
        
        return fig
    
    def create_threat_hunting_analytics(self, hunting_data, ioc_data):
        """
        Create advanced threat hunting analytics dashboard.
        Implements behavioral analytics and proactive threat detection.
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Hunt Campaign Progress", "IOC Detection Timeline", "Behavioral Anomalies",
                "Threat Landscape Map", "Hunt Effectiveness Metrics", "Proactive Alerts"
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}, {"type": "histogram"}],
                [{"type": "pie"}, {"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Hunt Campaign Progress
        campaigns = hunting_data['campaign_name'].unique()
        campaign_progress = []
        
        for campaign in campaigns:
            campaign_data = hunting_data[hunting_data['campaign_name'] == campaign]
            progress = len(campaign_data[campaign_data['status'] == 'completed']) / len(campaign_data) * 100
            campaign_progress.append(progress)
        
        fig.add_trace(
            go.Bar(
                x=campaigns,
                y=campaign_progress,
                marker_color=self.blue_colors['info'],
                name="Campaign Progress %"
            ),
            row=1, col=1
        )
        
        # 2. IOC Detection Timeline
        ioc_timeline = pd.DataFrame(ioc_data)
        ioc_timeline['detection_time'] = pd.to_datetime(ioc_timeline['detection_time'])
        
        # Group by IOC type
        ioc_types = ioc_timeline['ioc_type'].unique()
        colors = px.colors.qualitative.Set1[:len(ioc_types)]
        
        for i, ioc_type in enumerate(ioc_types):
            type_data = ioc_timeline[ioc_timeline['ioc_type'] == ioc_type]
            fig.add_trace(
                go.Scatter(
                    x=type_data['detection_time'],
                    y=[ioc_type] * len(type_data),
                    mode='markers',
                    marker=dict(size=8, color=colors[i]),
                    name=f"IOC: {ioc_type}",
                    hovertemplate=f'IOC Type: {ioc_type}<br>Time: %{{x}}<br>Confidence: %{{text}}<extra></extra>',
                    text=type_data['confidence']
                ),
                row=1, col=2
            )
        
        # 3. Behavioral Anomalies
        anomaly_scores = hunting_data['anomaly_score']
        normal_threshold = 0.3
        
        fig.add_trace(
            go.Histogram(
                x=anomaly_scores,
                nbinsx=30,
                marker_color=self.blue_colors['warning'],
                opacity=0.7,
                name="Anomaly Distribution"
            ),
            row=1, col=3
        )
        
        # Add threshold line
        fig.add_vline(
            x=normal_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Anomaly Threshold",
            row=1, col=3
        )
        
        # 4. Threat Landscape Map (Geographic or Network-based)
        threat_locations = hunting_data['source_location'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=threat_locations.index,
                values=threat_locations.values,
                hole=0.4,
                marker_colors=px.colors.sequential.Blues_r,
                name="Threat Geography"
            ),
            row=2, col=1
        )
        
        # 5. Hunt Effectiveness Metrics (data-derived, not ground-truth claims)
        # With typical IOC feeds we don't have labels for TP/FP/TN/FN. Instead, report
        # descriptive metrics that are actually computable from the inputs.
        num_campaigns = int(hunting_data['campaign_name'].nunique()) if 'campaign_name' in hunting_data.columns else 0
        num_iocs = int(len(ioc_timeline)) if not ioc_timeline.empty else 0
        mean_conf = float(ioc_timeline['confidence'].mean()) if 'confidence' in ioc_timeline.columns and num_iocs > 0 else 0.0
        high_conf_frac = float((ioc_timeline['confidence'] >= 0.9).mean()) if 'confidence' in ioc_timeline.columns and num_iocs > 0 else 0.0

        metrics_df = pd.DataFrame({
            'Metric': ['Campaigns', 'IOCs', 'Mean IOC confidence', 'High-confidence IOC fraction (≥0.9)'],
            'Value': [float(num_campaigns), float(num_iocs), float(mean_conf), float(high_conf_frac)],
        })
        
        fig.add_trace(
            go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Value'],
                marker_color=[self.blue_colors['success'], self.blue_colors['info'], self.blue_colors['primary']],
                name="Hunt Effectiveness"
            ),
            row=2, col=2
        )
        
        # 6. Proactive Alerts Timeline
        if not ioc_timeline.empty and 'detection_time' in ioc_timeline.columns:
            # Use IOC detection counts as a proxy for alert volume (data-derived).
            s = (
                ioc_timeline
                .dropna(subset=['detection_time'])
                .set_index('detection_time')
                .resample('4H')
                .size()
            )
            alert_times = s.index
            alert_counts = s.values
        else:
            alert_times = None
            alert_counts = None
        
        if alert_times is not None and alert_counts is not None:
            fig.add_trace(
                go.Scatter(
                    x=alert_times,
                    y=alert_counts,
                    mode='lines+markers',
                    line=dict(color=self.blue_colors['danger'], width=2),
                    marker=dict(size=6),
                    name="IOC-derived alerts"
                ),
                row=2, col=3
            )
        else:
            fig.add_annotation(
                text="No IOC timeline provided; cannot derive alert counts.",
                showarrow=False,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                font=dict(size=10, color="#444"),
                row=2,
                col=3,
            )
        
        fig.update_layout(
            title="Blue Team Threat Hunting Analytics - Proactive Defense",
            height=800,
            template="plotly_white",
            font=dict(size=10),
            showlegend=True
        )
        
        return fig
    
    def create_defensive_posture_assessment(self, security_metrics, control_effectiveness):
        """
        Create comprehensive defensive posture assessment dashboard.
        Implements security control effectiveness and gap analysis.
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Security Control Effectiveness", "Risk Heat Map", "Compliance Status",
                "Defense Maturity Model", "Vulnerability Trends", "Investment ROI"
            ],
            specs=[
                [{"type": "scatterpolar"}, {"type": "heatmap"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "indicator"}]
            ]
        )
        
        # 1. Security Control Effectiveness (Radar Chart)
        control_categories = list(control_effectiveness.keys())
        effectiveness_scores = list(control_effectiveness.values())
        
        fig.add_trace(
            go.Scatterpolar(
                r=effectiveness_scores,
                theta=control_categories,
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.3)',
                line=dict(color=self.blue_colors['primary'], width=2),
                name="Current Effectiveness"
            ),
            row=1, col=1
        )
        
        # Add target effectiveness
        target_scores = [90] * len(control_categories)
        fig.add_trace(
            go.Scatterpolar(
                r=target_scores,
                theta=control_categories,
                fill='toself',
                fillcolor='rgba(44, 160, 44, 0.2)',
                line=dict(color=self.blue_colors['success'], width=2, dash='dash'),
                name="Target Effectiveness"
            ),
            row=1, col=1
        )
        
        # 2. Risk Heat Map
        risk_categories = ['Network', 'Endpoint', 'Application', 'Data', 'Identity']
        threat_types = ['RL-Obfuscation', 'Cross-Modal', 'Temporal-Evo', 'Traditional']

        def _norm01(v) -> float:
            try:
                x = float(v)
            except Exception:
                return 0.0
            if not np.isfinite(x):
                return 0.0
            # Accept [0,100] or [0,1]
            if x > 1.0:
                x = x / 100.0
            return float(np.clip(x, 0.0, 1.0))

        # Derive category risks from provided metrics where available (no random placeholders).
        sm = security_metrics
        if isinstance(security_metrics, dict) and 'metrics' in security_metrics and isinstance(security_metrics['metrics'], dict):
            sm = security_metrics['metrics']
        sm = sm if isinstance(sm, dict) else {}

        # Derive an overall baseline from control_effectiveness if we can't map categories.
        ce = control_effectiveness if isinstance(control_effectiveness, dict) else {}
        ce_vals = [_norm01(v) for v in ce.values()] if ce else []
        baseline_eff = float(np.mean(ce_vals)) if ce_vals else 0.0

        # Simple mapping from categories to common metric keys.
        cat_keys = {
            'Network': ['network_security', 'network', 'net'],
            'Endpoint': ['endpoint_protection', 'endpoint', 'edr'],
            'Application': ['application_security', 'appsec', 'application'],
            'Data': ['data_protection', 'data', 'dataloss'],
            'Identity': ['identity_management', 'identity', 'iam'],
        }
        cat_eff = []
        for cat in risk_categories:
            eff = None
            for k in cat_keys.get(cat, []):
                if k in sm:
                    eff = _norm01(sm.get(k))
                    break
            if eff is None:
                eff = baseline_eff
            cat_eff.append(float(eff))

        cat_risk = 1.0 - np.asarray(cat_eff, dtype=np.float64)
        risk_matrix = np.clip(cat_risk[:, None] * np.ones((1, len(threat_types))), 0.0, 1.0)
        
        fig.add_trace(
            go.Heatmap(
                z=risk_matrix,
                x=threat_types,
                y=risk_categories,
                colorscale='Reds',
                colorbar=dict(title="Risk Level"),
                hovertemplate='Category: %{y}<br>Threat: %{x}<br>Risk: %{z:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Compliance Status
        compliance_frameworks = ['NIST', 'ISO 27001', 'SOC 2', 'PCI DSS', 'GDPR']
        compliance_scores = [85, 92, 78, 88, 95]
        
        colors = ['red' if score < 80 else 'orange' if score < 90 else 'green' for score in compliance_scores]
        
        fig.add_trace(
            go.Bar(
                x=compliance_frameworks,
                y=compliance_scores,
                marker_color=colors,
                name="Compliance Scores"
            ),
            row=1, col=3
        )
        
        # 4. Defense Maturity Model
        maturity_levels = ['Initial', 'Managed', 'Defined', 'Quantitatively Managed', 'Optimizing']
        current_maturity = [20, 35, 60, 75, 40]  # Percentage in each level
        target_maturity = [5, 15, 25, 35, 20]
        
        fig.add_trace(
            go.Scatter(
                x=maturity_levels,
                y=current_maturity,
                mode='lines+markers',
                name='Current Maturity',
                line=dict(color=self.blue_colors['primary'], width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=maturity_levels,
                y=target_maturity,
                mode='lines+markers',
                name='Target Maturity',
                line=dict(color=self.blue_colors['success'], width=3, dash='dash'),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        # 5. Vulnerability Trends
        months = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
        vt = None
        if isinstance(security_metrics, dict):
            vt = security_metrics.get('vulnerability_trends')
        if isinstance(vt, dict) and all(k in vt for k in ('months', 'critical', 'high')):
            try:
                months = pd.to_datetime(vt['months'])
                critical_vulns = np.asarray(vt['critical'], dtype=np.float64)
                high_vulns = np.asarray(vt['high'], dtype=np.float64)
                fig.add_trace(
                    go.Scatter(
                        x=months,
                        y=critical_vulns,
                        mode='lines+markers',
                        name='Critical Vulnerabilities',
                        line=dict(color='red', width=2),
                        marker=dict(size=6)
                    ),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Scatter(
                        x=months,
                        y=high_vulns,
                        mode='lines+markers',
                        name='High Vulnerabilities',
                        line=dict(color='orange', width=2),
                        marker=dict(size=6)
                    ),
                    row=2, col=2
                )
            except Exception:
                fig.add_annotation(
                    text="Vulnerability trends provided but could not be parsed.",
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
                text="No vulnerability_trends provided; not plotting fabricated counts.",
                showarrow=False,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                font=dict(size=10, color="#444"),
                row=2,
                col=2,
            )
        
        # 6. Security Investment ROI
        total_investment = 2500000  # $2.5M
        incidents_prevented = 45
        avg_incident_cost = 150000  # $150K per incident
        roi_value = (incidents_prevented * avg_incident_cost - total_investment) / total_investment * 100
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=roi_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Security Investment ROI (%)"},
                delta={'reference': 200, 'relative': False},
                gauge={
                    'axis': {'range': [None, 500]},
                    'bar': {'color': self.blue_colors['success']},
                    'steps': [
                        {'range': [0, 100], 'color': "lightgray"},
                        {'range': [100, 200], 'color': "yellow"},
                        {'range': [200, 300], 'color': "lightgreen"},
                        {'range': [300, 500], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 150
                    }
                }
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title="Blue Team Defensive Posture Assessment - Security Effectiveness",
            height=800,
            template="plotly_white",
            font=dict(size=10),
            showlegend=True
        )
        
        return fig
    
    def create_soc_operations_dashboard(self, alert_data, analyst_metrics):
        """
        Create comprehensive SOC operations dashboard.
        Implements AI-Native SOC operational metrics and analyst performance tracking.
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "Alert Volume Trends", "Alert Severity Distribution", "MTTR by Severity",
                "Analyst Workload", "False Positive Rate", "Escalation Patterns",
                "Tool Effectiveness", "Shift Performance", "Automation Impact"
            ],
            specs=[
                [{"secondary_y": True}, {"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"secondary_y": True}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "bar"}, {"type": "indicator"}]
            ]
        )
        
        # 1. Alert Volume Trends with Resolution Rate
        alert_timeline = pd.DataFrame(alert_data)
        alert_timeline['timestamp'] = pd.to_datetime(alert_timeline['timestamp'])
        
        # Group by hour
        hourly_alerts = alert_timeline.groupby(alert_timeline['timestamp'].dt.hour).agg({
            'alert_id': 'count',
            'resolved': 'sum'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=hourly_alerts['timestamp'],
                y=hourly_alerts['alert_id'],
                mode='lines+markers',
                name='Total Alerts',
                line=dict(color=self.blue_colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        resolution_rate = (hourly_alerts['resolved'] / hourly_alerts['alert_id'] * 100).fillna(0)
        fig.add_trace(
            go.Scatter(
                x=hourly_alerts['timestamp'],
                y=resolution_rate,
                mode='lines+markers',
                name='Resolution Rate %',
                line=dict(color=self.blue_colors['success'], width=2),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Alert Severity Distribution
        severity_counts = alert_timeline['severity'].value_counts()
        severity_colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c']
        
        fig.add_trace(
            go.Pie(
                labels=severity_counts.index,
                values=severity_counts.values,
                marker_colors=severity_colors,
                name="Severity Distribution"
            ),
            row=1, col=2
        )
        
        # 3. Mean Time to Resolution (MTTR) by Severity
        mttr_by_severity = alert_timeline.groupby('severity')['resolution_time_minutes'].mean()
        
        fig.add_trace(
            go.Bar(
                x=mttr_by_severity.index,
                y=mttr_by_severity.values,
                marker_color=severity_colors,
                name="MTTR by Severity"
            ),
            row=1, col=3
        )
        
        # 4. Analyst Workload Distribution
        analyst_workload = pd.Series(analyst_metrics['alerts_handled']).value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(
                x=[f"Analyst {i}" for i in range(len(analyst_workload))],
                y=analyst_workload.values,
                marker_color=self.blue_colors['info'],
                name="Analyst Workload"
            ),
            row=2, col=1
        )
        
        # 5. False Positive Rate Trends (only if labels exist; no synthetic timelines)
        fp_series = None
        if 'is_false_positive' in alert_timeline.columns:
            fp_series = alert_timeline.set_index('timestamp')['is_false_positive'].resample('1D').mean()
        elif 'false_positive' in alert_timeline.columns:
            fp_series = alert_timeline.set_index('timestamp')['false_positive'].resample('1D').mean()

        if fp_series is not None and len(fp_series) > 0:
            fig.add_trace(
                go.Scatter(
                    x=fp_series.index,
                    y=fp_series.values,
                    mode='lines+markers',
                    name='False Positive Rate',
                    line=dict(color=self.blue_colors['warning'], width=2)
                ),
                row=2, col=2
            )
        else:
            fig.add_annotation(
                text="No false-positive labels provided; FP rate not plotted.",
                showarrow=False,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                font=dict(size=10, color="#444"),
                row=2,
                col=2,
            )
        
        # 6. Escalation Patterns (Sankey Diagram)
        # Note: Simplified representation due to subplot limitations
        escalation_data = {
            'L1 → L2': 150,
            'L1 → Resolved': 300,
            'L2 → L3': 50,
            'L2 → Resolved': 100,
            'L3 → Incident': 20,
            'L3 → Resolved': 30
        }
        
        # Convert to bar chart for subplot compatibility
        fig.add_trace(
            go.Bar(
                x=list(escalation_data.keys()),
                y=list(escalation_data.values()),
                marker_color=self.blue_colors['secondary'],
                name="Escalation Patterns"
            ),
            row=2, col=3
        )
        
        # 7. Tool Effectiveness Heatmap
        tools = ['SIEM', 'EDR', 'SOAR', 'TIP', 'UEBA']
        metrics = ['Detection', 'Investigation', 'Response', 'Recovery']
        tm = None
        if isinstance(analyst_metrics, dict):
            tm = analyst_metrics.get('tool_effectiveness')
        if isinstance(tm, dict) and all(k in tm for k in ('tools', 'metrics', 'matrix')):
            try:
                tools = [str(x) for x in tm['tools']]
                metrics = [str(x) for x in tm['metrics']]
                effectiveness_matrix = np.asarray(tm['matrix'], dtype=np.float64)
                fig.add_trace(
                    go.Heatmap(
                        z=effectiveness_matrix,
                        x=metrics,
                        y=tools,
                        colorscale='Blues',
                        colorbar=dict(title="Effectiveness"),
                        hovertemplate='Tool: %{y}<br>Metric: %{x}<br>Score: %{z:.2f}<extra></extra>'
                    ),
                    row=3, col=1
                )
            except Exception:
                fig.add_annotation(
                    text="tool_effectiveness provided but could not be parsed.",
                    showarrow=False,
                    xref="x domain",
                    yref="y domain",
                    x=0.5,
                    y=0.5,
                    font=dict(size=10, color="#444"),
                    row=3,
                    col=1,
                )
        else:
            fig.add_annotation(
                text="No tool_effectiveness provided; heatmap not plotted.",
                showarrow=False,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                font=dict(size=10, color="#444"),
                row=3,
                col=1,
            )
        
        # 8. Shift Performance Comparison
        shifts = ['Day', 'Evening', 'Night']
        performance_metrics = {
            'Alerts Handled': [450, 320, 180],
            'Avg Resolution Time': [25, 35, 45],
            'Escalation Rate': [15, 20, 25]
        }
        
        for metric, values in performance_metrics.items():
            fig.add_trace(
                go.Bar(
                    x=shifts,
                    y=values,
                    name=metric,
                    offsetgroup=metric
                ),
                row=3, col=2
            )
        
        # 9. Automation Impact Indicator
        automation_savings = 65  # Percentage of manual work automated
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=automation_savings,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Automation Impact (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.blue_colors['success']},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "blue", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ),
            row=3, col=3
        )
        
        fig.update_layout(
            title="Blue Team SOC Operations Dashboard - AI-Native Operations",
            height=1200,
            template="plotly_white",
            font=dict(size=9),
            showlegend=True
        )
        
        return fig

def demonstrate_blue_team_intelligence():
    """Demonstrate the blue team intelligence engine."""
    print("🛡️ NeurInSpectre Blue Team Intelligence Engine Demo")
    print("=" * 60)
    
    # Initialize blue team engine
    blue_team = BlueTeamIntelligenceEngine()
    
    # Generate synthetic data for demonstrations
    np.random.seed(42)
    
    # 1. Incident Response Dashboard
    print("📊 Creating incident response dashboard...")
    
    # Generate incident data
    incident_data = {
        'incident_id': [f"INC-{i:04d}" for i in range(1, 51)],
        'severity': np.random.choice(['Low', 'Medium', 'High', 'Critical'], 50, p=[0.3, 0.4, 0.2, 0.1]),
        'attack_vector': np.random.choice(['RL-Obfuscation', 'Cross-Modal', 'Temporal-Evolution', 'Traditional'], 50),
        'response_time_hours': np.random.exponential(2, 50),
        'recovery_time_hours': np.random.exponential(8, 50)
    }
    
    # Generate timeline data
    timeline_data = []
    for i in range(100):
        timeline_data.append({
            'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 72)),
            'phase': np.random.choice(['Detection', 'Analysis', 'Containment', 'Eradication', 'Recovery']),
            'action': f"Action {i+1}"
        })
    
    fig1 = blue_team.create_incident_response_dashboard(incident_data, timeline_data)
    fig1.write_html("neurinspectre_incident_response.html")
    print("   ✅ Saved: neurinspectre_incident_response.html")
    
    # 2. Threat Hunting Analytics
    print("📊 Creating threat hunting analytics...")
    
    # Generate hunting data
    hunting_data = pd.DataFrame({
        'campaign_name': np.random.choice(['Hunt-Alpha', 'Hunt-Beta', 'Hunt-Gamma'], 100),
        'status': np.random.choice(['active', 'completed', 'pending'], 100, p=[0.3, 0.6, 0.1]),
        'anomaly_score': np.random.beta(2, 5, 100),
        'source_location': np.random.choice(['Internal', 'External-US', 'External-EU', 'External-APAC'], 100)
    })
    
    # Generate IOC data
    ioc_data = []
    for i in range(50):
        ioc_data.append({
            'detection_time': datetime.now() - timedelta(hours=np.random.randint(0, 168)),
            'ioc_type': np.random.choice(['IP', 'Domain', 'Hash', 'URL']),
            'confidence': np.random.uniform(0.5, 1.0)
        })
    
    fig2 = blue_team.create_threat_hunting_analytics(hunting_data, ioc_data)
    fig2.write_html("neurinspectre_threat_hunting.html")
    print("   ✅ Saved: neurinspectre_threat_hunting.html")
    
    # 3. Defensive Posture Assessment
    print("📊 Creating defensive posture assessment...")
    
    security_metrics = {
        'network_security': 85,
        'endpoint_protection': 92,
        'identity_management': 78,
        'data_protection': 88,
        'incident_response': 90
    }
    
    control_effectiveness = {
        'Prevention': 85,
        'Detection': 90,
        'Response': 88,
        'Recovery': 82,
        'Monitoring': 95,
        'Analysis': 87
    }
    
    fig3 = blue_team.create_defensive_posture_assessment(security_metrics, control_effectiveness)
    fig3.write_html("neurinspectre_defensive_posture.html")
    print("   ✅ Saved: neurinspectre_defensive_posture.html")
    
    # 4. SOC Operations Dashboard
    print("📊 Creating SOC operations dashboard...")
    
    # Generate alert data
    alert_data = []
    for i in range(1000):
        alert_data.append({
            'alert_id': f"ALT-{i:06d}",
            'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 168)),
            'severity': np.random.choice(['Low', 'Medium', 'High', 'Critical'], p=[0.4, 0.3, 0.2, 0.1]),
            'resolved': np.random.choice([0, 1], p=[0.2, 0.8]),
            'resolution_time_minutes': np.random.exponential(45)
        })
    
    # Generate analyst metrics
    analyst_metrics = {
        'alerts_handled': np.random.poisson(50, 20),
        'resolution_time': np.random.exponential(30, 20),
        'escalation_rate': np.random.uniform(0.1, 0.3, 20)
    }
    
    fig4 = blue_team.create_soc_operations_dashboard(alert_data, analyst_metrics)
    fig4.write_html("neurinspectre_soc_operations.html")
    print("   ✅ Saved: neurinspectre_soc_operations.html")
    
    print("\n🛡️ Blue Team Intelligence Engine Demo Complete!")
    print("📁 Generated Files:")
    print("   • neurinspectre_incident_response.html")
    print("   • neurinspectre_threat_hunting.html")
    print("   • neurinspectre_defensive_posture.html")
    print("   • neurinspectre_soc_operations.html")
    
    return blue_team

if __name__ == "__main__":
    blue_team = demonstrate_blue_team_intelligence()

