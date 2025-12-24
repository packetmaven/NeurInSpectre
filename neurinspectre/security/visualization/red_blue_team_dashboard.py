"""
Red & Blue Team Visualization Dashboard
Comprehensive visualization system for adversarial security analysis
Designed for red/blue team security analysis workflows.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Tuple, Any
from pathlib import Path
import time
from dataclasses import dataclass
import warnings

@dataclass
class SecurityAssessment:
    """Security assessment results"""
    threat_level: str
    attack_type: str
    confidence: float
    detection_methods: List[str]
    evasion_techniques: List[str]
    recommendations: Dict[str, List[str]]
    metadata: Dict[str, Any]

class RedBlueTeamDashboard:
    """
    Comprehensive visualization dashboard for red and blue team analysis
    """
    
    def __init__(self, figsize: Tuple[int, int] = (20, 12)):
        self.figsize = figsize

        self.colors = {
            'red_team': '#FF4444',
            'blue_team': '#4444FF',
            'threat_high': '#FF2222',
            'threat_medium': '#FF8800',
            'threat_low': '#FFCC00',
            'success': '#00AA00',
            'warning': '#FF8800',
            'info': '#4488FF',
            'background': '#2E2E2E',
            'text': '#FFFFFF'
        }

        # Per-instance plotting style (applied via context managers at render time).
        self._mpl_style = 'seaborn-v0_8-dark'
        self._sns_palette = "husl"
        self._mpl_rc = {
            'figure.facecolor': self.colors['background'],
            'axes.facecolor': self.colors['background'],
            'text.color': self.colors['text'],
            'axes.labelcolor': self.colors['text'],
            'xtick.color': self.colors['text'],
            'ytick.color': self.colors['text'],
            'legend.facecolor': self.colors['background'],
            'legend.edgecolor': self.colors['text'],
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
    
    def create_comprehensive_dashboard(self, attack_data: Dict[str, np.ndarray], 
                                     detection_results: Dict[str, SecurityAssessment],
                                     save_path: str = "security_dashboard.png") -> None:
        """Create comprehensive dashboard for red and blue teams"""
        with self._plot_context():
            fig = plt.figure(figsize=self.figsize)
            fig.suptitle(
                'RED TEAM & BLUE TEAM SECURITY ANALYSIS DASHBOARD',
                fontsize=20,
                fontweight='bold',
                y=0.98,
            )
            
            # Create grid layout
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # 1. Threat Landscape Overview (top-left)
            ax1 = fig.add_subplot(gs[0, 0:2])
            self._plot_threat_landscape(ax1, detection_results)
            
            # 2. Attack Pattern Analysis (top-right)
            ax2 = fig.add_subplot(gs[0, 2:4])
            self._plot_attack_patterns(ax2, attack_data)
            
            # 3. Detection Effectiveness (middle-left)
            ax3 = fig.add_subplot(gs[1, 0:2])
            self._plot_detection_effectiveness(ax3, detection_results)
            
            # 4. Evasion Techniques Analysis (middle-right)
            ax4 = fig.add_subplot(gs[1, 2:4])
            self._plot_evasion_techniques(ax4, detection_results)
            
            # 5. Red Team Intelligence (bottom-left)
            ax5 = fig.add_subplot(gs[2, 0:2])
            self._plot_red_team_intelligence(ax5, attack_data, detection_results)
            
            # 6. Blue Team Intelligence (bottom-right)
            ax6 = fig.add_subplot(gs[2, 2:4])
            self._plot_blue_team_intelligence(ax6, detection_results)
            
            # 7. Actionable Recommendations (bottom span)
            ax7 = fig.add_subplot(gs[3, :])
            self._plot_actionable_recommendations(ax7, detection_results)
            
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This figure includes Axes that are not compatible with tight_layout.*",
                    category=UserWarning,
                )
                plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.colors['background'])
        
        print(f"🎯 Dashboard saved to: {save_path}")
        
        # Also create separate focused dashboards
        self._create_red_team_focused_dashboard(attack_data, detection_results)
        self._create_blue_team_focused_dashboard(attack_data, detection_results)
    
    def _plot_threat_landscape(self, ax, detection_results: Dict[str, SecurityAssessment]):
        """Plot threat landscape overview"""
        
        ax.set_title('THREAT LANDSCAPE OVERVIEW', fontsize=14, fontweight='bold')
        
        # Extract threat levels
        threat_levels = []
        attack_types = []
        confidence_scores = []
        
        for name, result in detection_results.items():
            threat_levels.append(result.threat_level)
            attack_types.append(result.attack_type)
            confidence_scores.append(result.confidence)
        
        # Create bubble chart
        y_pos = np.arange(len(attack_types))
        
        # Map threat levels to colors and sizes
        colors = []
        sizes = []
        for threat in threat_levels:
            if threat == 'HIGH':
                colors.append(self.colors['threat_high'])
                sizes.append(300)
            elif threat == 'MEDIUM':
                colors.append(self.colors['threat_medium'])
                sizes.append(200)
            else:
                colors.append(self.colors['threat_low'])
                sizes.append(100)
        
        ax.scatter(
            confidence_scores,
            y_pos,
            c=colors,
            s=sizes,
            alpha=0.7,
            edgecolors='white',
            linewidth=2,
        )
        
        ax.set_xlabel('Detection Confidence', fontsize=12)
        ax.set_ylabel('Attack Types', fontsize=12)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([t.replace('_', ' ').title() for t in attack_types])
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], s=300, c=self.colors['threat_high'], label='HIGH THREAT'),
            plt.scatter([], [], s=200, c=self.colors['threat_medium'], label='MEDIUM THREAT'),
            plt.scatter([], [], s=100, c=self.colors['threat_low'], label='LOW THREAT')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    def _plot_attack_patterns(self, ax, attack_data: Dict[str, np.ndarray]):
        """Plot attack pattern analysis"""
        
        ax.set_title('ATTACK PATTERN ANALYSIS', fontsize=14, fontweight='bold')
        
        # Calculate pattern characteristics
        pattern_metrics = {}
        for name, data in attack_data.items():
            if len(data.shape) == 2:
                # For 2D data, calculate complexity metrics
                mean_activation = np.mean(data, axis=0)
                variance = np.var(data, axis=0)
                pattern_metrics[name] = {
                    'complexity': np.mean(variance),
                    'consistency': 1.0 - np.std(variance) / np.mean(variance) if np.mean(variance) > 0 else 0,
                    'amplitude': np.max(np.abs(mean_activation))
                }
            else:
                # For 1D data
                pattern_metrics[name] = {
                    'complexity': np.var(data),
                    'consistency': 1.0 - np.std(data) / np.mean(data) if np.mean(data) > 0 else 0,
                    'amplitude': np.max(np.abs(data))
                }
        
        # Create radar chart for pattern analysis
        categories = list(pattern_metrics.keys())
        metrics = ['complexity', 'consistency', 'amplitude']
        
        # Normalize metrics
        normalized_data = {}
        for metric in metrics:
            values = [pattern_metrics[cat][metric] for cat in categories]
            max_val = max(values) if max(values) > 0 else 1
            normalized_data[metric] = [v/max_val for v in values]
        
        # Create stacked bar chart
        x = np.arange(len(categories))
        width = 0.25
        
        ax.bar(
            x - width,
            normalized_data['complexity'],
            width,
            label='Complexity',
            color=self.colors['red_team'],
            alpha=0.7,
        )
        ax.bar(
            x,
            normalized_data['consistency'],
            width,
            label='Consistency',
            color=self.colors['blue_team'],
            alpha=0.7,
        )
        ax.bar(
            x + width,
            normalized_data['amplitude'],
            width,
            label='Amplitude',
            color=self.colors['warning'],
            alpha=0.7,
        )
        
        ax.set_xlabel('Attack Types', fontsize=12)
        ax.set_ylabel('Normalized Metrics', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_detection_effectiveness(self, ax, detection_results: Dict[str, SecurityAssessment]):
        """Plot detection effectiveness analysis"""
        
        ax.set_title('DETECTION EFFECTIVENESS', fontsize=14, fontweight='bold')
        
        # Extract detection methods and their effectiveness
        method_effectiveness = {}
        for name, result in detection_results.items():
            for method in result.detection_methods:
                if method not in method_effectiveness:
                    method_effectiveness[method] = []
                method_effectiveness[method].append(result.confidence)
        
        # Calculate average effectiveness per method
        methods = list(method_effectiveness.keys())
        avg_effectiveness = [np.mean(method_effectiveness[method]) for method in methods]
        std_effectiveness = [np.std(method_effectiveness[method]) for method in methods]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(methods))
        bars = ax.barh(y_pos, avg_effectiveness, xerr=std_effectiveness, 
                      capsize=5, alpha=0.7, color=self.colors['blue_team'])
        
        ax.set_xlabel('Detection Effectiveness', fontsize=12)
        ax.set_ylabel('Detection Methods', fontsize=12)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([m.replace('_', ' ').title() for m in methods])
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add effectiveness labels
        for i, (bar, eff) in enumerate(zip(bars, avg_effectiveness)):
            ax.text(eff + 0.02, i, f'{eff:.2f}', 
                   va='center', fontweight='bold')
    
    def _plot_evasion_techniques(self, ax, detection_results: Dict[str, SecurityAssessment]):
        """Plot evasion techniques analysis"""
        
        ax.set_title('EVASION TECHNIQUES ANALYSIS', fontsize=14, fontweight='bold')
        
        # Extract evasion techniques
        evasion_counts = {}
        for name, result in detection_results.items():
            for technique in result.evasion_techniques:
                evasion_counts[technique] = evasion_counts.get(technique, 0) + 1
        
        # Create pie chart
        if evasion_counts:
            techniques = list(evasion_counts.keys())
            counts = list(evasion_counts.values())
            
            # Custom colors for pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(techniques)))
            
            wedges, texts, autotexts = ax.pie(counts, labels=techniques, autopct='%1.1f%%',
                                            startangle=90, colors=colors, 
                                            textprops={'fontsize': 10})
            
            # Enhance text visibility
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax.text(0.5, 0.5, 'No evasion techniques detected', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, style='italic')
    
    def _plot_red_team_intelligence(self, ax, attack_data: Dict[str, np.ndarray], 
                                   detection_results: Dict[str, SecurityAssessment]):
        """Plot red team intelligence and recommendations"""
        
        ax.set_title('RED TEAM INTELLIGENCE', fontsize=14, fontweight='bold', 
                    color=self.colors['red_team'])
        
        # Create actionable intelligence for red team
        red_intel = []
        for name, result in detection_results.items():
            if 'red_team' in result.recommendations:
                red_intel.extend(result.recommendations['red_team'])
        
        if not red_intel:
            red_intel = [
                "Increase attack sophistication",
                "Vary attack patterns",
                "Use adaptive evasion",
                "Analyze detection patterns",
                "Implement real-time adaptation",
            ]
        
        # Create text visualization
        ax.axis('off')
        
        # Title box
        title_box = FancyBboxPatch((0.02, 0.85), 0.96, 0.12, 
                                  boxstyle="round,pad=0.02",
                                  facecolor=self.colors['red_team'], 
                                  edgecolor='white', linewidth=2, alpha=0.8)
        ax.add_patch(title_box)
        ax.text(0.5, 0.91, 'OFFENSIVE RECOMMENDATIONS', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               color='white', transform=ax.transAxes)
        
        # Intelligence items
        y_positions = np.linspace(0.75, 0.1, len(red_intel[:5]))
        for i, intel in enumerate(red_intel[:5]):
            # Create box for each item
            item_box = FancyBboxPatch((0.02, y_positions[i]-0.05), 0.96, 0.08,
                                     boxstyle="round,pad=0.01",
                                     facecolor=self.colors['red_team'], 
                                     alpha=0.3, edgecolor=self.colors['red_team'])
            ax.add_patch(item_box)
            
            ax.text(0.05, y_positions[i], intel, 
                   va='center', fontsize=10, fontweight='bold',
                   color=self.colors['red_team'], transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _plot_blue_team_intelligence(self, ax, detection_results: Dict[str, SecurityAssessment]):
        """Plot blue team intelligence and recommendations"""
        
        ax.set_title('BLUE TEAM INTELLIGENCE', fontsize=14, fontweight='bold', 
                    color=self.colors['blue_team'])
        
        # Create actionable intelligence for blue team
        blue_intel = []
        for name, result in detection_results.items():
            if 'blue_team' in result.recommendations:
                blue_intel.extend(result.recommendations['blue_team'])
        
        if not blue_intel:
            blue_intel = [
                "Strengthen detection algorithms",
                "Improve monitoring coverage",
                "Enhance anomaly detection",
                "Implement real-time alerts",
                "Focus on high-risk vectors",
            ]
        
        # Create text visualization
        ax.axis('off')
        
        # Title box
        title_box = FancyBboxPatch((0.02, 0.85), 0.96, 0.12, 
                                  boxstyle="round,pad=0.02",
                                  facecolor=self.colors['blue_team'], 
                                  edgecolor='white', linewidth=2, alpha=0.8)
        ax.add_patch(title_box)
        ax.text(0.5, 0.91, 'DEFENSIVE RECOMMENDATIONS', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               color='white', transform=ax.transAxes)
        
        # Intelligence items
        y_positions = np.linspace(0.75, 0.1, len(blue_intel[:5]))
        for i, intel in enumerate(blue_intel[:5]):
            # Create box for each item
            item_box = FancyBboxPatch((0.02, y_positions[i]-0.05), 0.96, 0.08,
                                     boxstyle="round,pad=0.01",
                                     facecolor=self.colors['blue_team'], 
                                     alpha=0.3, edgecolor=self.colors['blue_team'])
            ax.add_patch(item_box)
            
            ax.text(0.05, y_positions[i], intel, 
                   va='center', fontsize=10, fontweight='bold',
                   color=self.colors['blue_team'], transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _plot_actionable_recommendations(self, ax, detection_results: Dict[str, SecurityAssessment]):
        """Plot actionable recommendations for both teams"""
        
        ax.set_title('ACTIONABLE INTELLIGENCE & NEXT STEPS', fontsize=14, fontweight='bold')
        
        # Create side-by-side recommendations
        ax.axis('off')
        
        # Red team section
        red_section = FancyBboxPatch((0.02, 0.1), 0.46, 0.8, 
                                   boxstyle="round,pad=0.02",
                                   facecolor=self.colors['red_team'], 
                                   alpha=0.2, edgecolor=self.colors['red_team'], linewidth=2)
        ax.add_patch(red_section)
        
        ax.text(0.25, 0.85, 'RED TEAM NEXT STEPS', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               color=self.colors['red_team'], transform=ax.transAxes)
        
        red_steps = [
            "1. Analyze detection blind spots",
            "2. Develop advanced evasion techniques",
            "3. Test multi-vector attacks",
            "4. Implement adaptive strategies",
            "5. Document attack methodologies"
        ]
        
        y_pos = 0.75
        for step in red_steps:
            ax.text(0.04, y_pos, step, va='center', fontsize=10,
                   color=self.colors['red_team'], transform=ax.transAxes)
            y_pos -= 0.12
        
        # Blue team section
        blue_section = FancyBboxPatch((0.52, 0.1), 0.46, 0.8, 
                                    boxstyle="round,pad=0.02",
                                    facecolor=self.colors['blue_team'], 
                                    alpha=0.2, edgecolor=self.colors['blue_team'], linewidth=2)
        ax.add_patch(blue_section)
        
        ax.text(0.75, 0.85, 'BLUE TEAM NEXT STEPS', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               color=self.colors['blue_team'], transform=ax.transAxes)
        
        blue_steps = [
            "1. Strengthen detection algorithms",
            "2. Implement real-time monitoring",
            "3. Enhance threat intelligence",
            "4. Deploy adaptive defenses",
            "5. Improve incident response"
        ]
        
        y_pos = 0.75
        for step in blue_steps:
            ax.text(0.54, y_pos, step, va='center', fontsize=10,
                   color=self.colors['blue_team'], transform=ax.transAxes)
            y_pos -= 0.12
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _create_red_team_focused_dashboard(self, attack_data: Dict[str, np.ndarray], 
                                         detection_results: Dict[str, SecurityAssessment]):
        """Create focused dashboard for red team"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RED TEAM FOCUSED ANALYSIS', fontsize=18, fontweight='bold',
                    color=self.colors['red_team'])
        
        # Attack success rates
        ax1 = axes[0, 0]
        success_rates = []
        attack_names = []
        for name, result in detection_results.items():
            success_rates.append(1.0 - result.confidence)  # Lower detection = higher success
            attack_names.append(name.replace('_', ' ').title())
        
        bars = ax1.bar(attack_names, success_rates, color=self.colors['red_team'], alpha=0.7)
        ax1.set_title('Attack Success Rates', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Evasion technique effectiveness
        ax2 = axes[0, 1]
        evasion_effectiveness = {}
        for name, result in detection_results.items():
            for technique in result.evasion_techniques:
                if technique not in evasion_effectiveness:
                    evasion_effectiveness[technique] = []
                evasion_effectiveness[technique].append(1.0 - result.confidence)
        
        if evasion_effectiveness:
            techniques = list(evasion_effectiveness.keys())
            avg_effectiveness = [np.mean(evasion_effectiveness[tech]) for tech in techniques]
            
            ax2.pie(avg_effectiveness, labels=techniques, autopct='%1.1f%%',
                   colors=plt.cm.Reds(np.linspace(0.3, 0.9, len(techniques))))
            ax2.set_title('Evasion Technique Effectiveness', fontsize=14, fontweight='bold')
        
        # Attack pattern complexity
        ax3 = axes[1, 0]
        complexities = []
        names = []
        for name, data in attack_data.items():
            if len(data.shape) == 2:
                complexity = np.mean(np.var(data, axis=0))
            else:
                complexity = np.var(data)
            complexities.append(complexity)
            names.append(name.replace('_', ' ').title())
        
        ax3.scatter(range(len(names)), complexities, s=200, 
                   color=self.colors['red_team'], alpha=0.7, edgecolors='white', linewidth=2)
        ax3.set_title('Attack Pattern Complexity', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Attack Types')
        ax3.set_ylabel('Complexity Score')
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Detection blind spots
        ax4 = axes[1, 1]
        blind_spots = []
        for name, result in detection_results.items():
            if result.confidence < 0.5:  # Low confidence = potential blind spot
                blind_spots.append((name, result.confidence))
        
        if blind_spots:
            names, confidences = zip(*blind_spots)
            ax4.barh(range(len(names)), [1-c for c in confidences], 
                    color=self.colors['red_team'], alpha=0.7)
            ax4.set_title('Detection Blind Spots', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Blind Spot Severity')
            ax4.set_yticks(range(len(names)))
            ax4.set_yticklabels([n.replace('_', ' ').title() for n in names])
        else:
            ax4.text(0.5, 0.5, 'No significant blind spots detected', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Detection Blind Spots', fontsize=14, fontweight='bold')
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout.*",
                category=UserWarning,
            )
            plt.tight_layout()
        plt.savefig('red_team_dashboard.png', dpi=300, bbox_inches='tight',
                   facecolor=self.colors['background'])
        print("🔴 Red team dashboard saved to: red_team_dashboard.png")
    
    def _create_blue_team_focused_dashboard(self, attack_data: Dict[str, np.ndarray], 
                                          detection_results: Dict[str, SecurityAssessment]):
        """Create focused dashboard for blue team"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('BLUE TEAM FOCUSED ANALYSIS', fontsize=18, fontweight='bold',
                    color=self.colors['blue_team'])
        
        # Detection accuracy
        ax1 = axes[0, 0]
        accuracies = []
        attack_names = []
        for name, result in detection_results.items():
            accuracies.append(result.confidence)
            attack_names.append(name.replace('_', ' ').title())
        
        ax1.bar(attack_names, accuracies, color=self.colors['blue_team'], alpha=0.7)
        ax1.set_title('Detection Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy Score')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add threshold line
        ax1.axhline(y=0.7, color=self.colors['warning'], linestyle='--', 
                   linewidth=2, label='Minimum Threshold')
        ax1.legend()
        
        # Threat level distribution
        ax2 = axes[0, 1]
        threat_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for name, result in detection_results.items():
            threat_counts[result.threat_level] += 1
        
        colors = [self.colors['threat_high'], self.colors['threat_medium'], self.colors['threat_low']]
        ax2.pie(threat_counts.values(), labels=threat_counts.keys(), autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax2.set_title('Threat Level Distribution', fontsize=14, fontweight='bold')
        
        # Coverage analysis
        ax3 = axes[1, 0]
        coverage_data = {}
        for name, result in detection_results.items():
            for method in result.detection_methods:
                if method not in coverage_data:
                    coverage_data[method] = 0
                coverage_data[method] += 1
        
        if coverage_data:
            methods = list(coverage_data.keys())
            counts = list(coverage_data.values())
            
            ax3.bar(methods, counts, color=self.colors['blue_team'], alpha=0.7)
            ax3.set_title('Detection Method Coverage', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Number of Attacks Covered')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Response time analysis
        ax4 = axes[1, 1]
        response_times = []
        attack_types = []
        for name, result in detection_results.items():
            rt = None
            meta = getattr(result, "metadata", None)
            if isinstance(meta, dict):
                rt = (
                    meta.get("processing_time_seconds")
                    or meta.get("processing_time")
                    or meta.get("response_time_seconds")
                )
            if rt is None:
                continue
            try:
                rt_f = float(rt)
            except Exception:
                continue
            if not np.isfinite(rt_f) or rt_f < 0.0:
                continue
            response_times.append(rt_f)
            attack_types.append(name.replace('_', ' ').title())

        ax4.set_title('Response Time Analysis', fontsize=14, fontweight='bold')
        if response_times:
            ax4.scatter(
                response_times,
                range(len(attack_types)),
                s=200,
                color=self.colors['blue_team'],
                alpha=0.7,
                edgecolors='white',
                linewidth=2,
            )
            ax4.set_xlabel('Processing / Response Time (seconds)')
            ax4.set_yticks(range(len(attack_types)))
            ax4.set_yticklabels(attack_types)
            ax4.grid(True, alpha=0.3)

            # Add SLA line (reference only)
            ax4.axvline(
                x=3.0,
                color=self.colors['warning'],
                linestyle='--',
                linewidth=2,
                label='SLA Threshold',
            )
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No response-time metadata available.', ha='center', va='center')
            ax4.set_axis_off()
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout.*",
                category=UserWarning,
            )
            plt.tight_layout()
        plt.savefig('blue_team_dashboard.png', dpi=300, bbox_inches='tight',
                   facecolor=self.colors['background'])
        print("🔵 Blue team dashboard saved to: blue_team_dashboard.png")
    
    def create_attack_timeline(self, attack_data: Dict[str, np.ndarray], 
                              detection_results: Dict[str, SecurityAssessment],
                              save_path: str = "attack_timeline.png"):
        """Create attack timeline visualization"""
        
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.suptitle('ATTACK TIMELINE & DETECTION RESPONSE', fontsize=18, fontweight='bold')
        
        # Build timeline data from assessment timestamps and optional processing-time metadata.
        timeline_data = []
        
        for i, (name, result) in enumerate(detection_results.items()):
            detection_time = float(getattr(result, "timestamp", time.time()))

            rt = 0.0
            meta = getattr(result, "metadata", None)
            if isinstance(meta, dict):
                rt_raw = (
                    meta.get("processing_time_seconds")
                    or meta.get("processing_time")
                    or meta.get("response_time_seconds")
                )
                try:
                    rt = float(rt_raw) if rt_raw is not None else 0.0
                except Exception:
                    rt = 0.0

            if not np.isfinite(rt) or rt < 0.0:
                rt = 0.0
            attack_time = detection_time - rt
            
            timeline_data.append({
                'attack_name': name,
                'attack_time': attack_time,
                'detection_time': detection_time,
                'threat_level': result.threat_level,
                'confidence': result.confidence
            })
        
        # Plot timeline
        for i, data in enumerate(timeline_data):
            attack_time = data['attack_time']
            detection_time = data['detection_time']
            
            # Attack start
            ax.scatter(attack_time, i, s=200, color=self.colors['red_team'], 
                      marker='v', label='Attack Start' if i == 0 else "")
            
            # Detection point
            ax.scatter(detection_time, i, s=200, color=self.colors['blue_team'], 
                      marker='o', label='Detection' if i == 0 else "")
            
            # Connection line
            ax.plot([attack_time, detection_time], [i, i], 
                   color=self.colors['warning'], linewidth=3, alpha=0.7)
            
            # Add attack name
            ax.text(attack_time - 30, i, data['attack_name'].replace('_', ' ').title(), 
                   ha='right', va='center', fontsize=10)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Attack Sequence')
        ax.set_yticks(range(len(timeline_data)))
        ax.set_yticklabels([f"Attack {i+1}" for i in range(len(timeline_data))])
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format time axis
        times = [data['attack_time'] for data in timeline_data] + [data['detection_time'] for data in timeline_data]
        ax.set_xlim(min(times) - 60, max(times) + 60)
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout.*",
                category=UserWarning,
            )
            plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=self.colors['background'])
        print(f"⏱️ Timeline saved to: {save_path}")

def create_comprehensive_report(attack_data: Dict[str, np.ndarray], 
                               detection_results: Dict[str, SecurityAssessment],
                               output_dir: str = "security_reports"):
    """Create comprehensive security report"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create dashboard
    dashboard = RedBlueTeamDashboard()
    
    # Main dashboard
    dashboard.create_comprehensive_dashboard(
        attack_data, detection_results, 
        str(output_path / "comprehensive_dashboard.png")
    )
    
    # Timeline
    dashboard.create_attack_timeline(
        attack_data, detection_results,
        str(output_path / "attack_timeline.png")
    )
    
    # Generate text report
    report_content = generate_text_report(attack_data, detection_results)
    
    with open(output_path / "security_analysis_report.md", 'w') as f:
        f.write(report_content)
    
    print(f"📊 Comprehensive security report created in: {output_path}")
    
    return output_path

def generate_text_report(attack_data: Dict[str, np.ndarray], 
                        detection_results: Dict[str, SecurityAssessment]) -> str:
    """Generate comprehensive text report"""
    
    report = f"""# 🔴🔵 RED & BLUE TEAM SECURITY ANALYSIS REPORT

## Executive Summary

This report summarizes {len(attack_data)} scenarios analyzed by NeurInSpectre.
Results are heuristic and input-dependent; validate conclusions with independent telemetry and controlled tests.

"""
    
    # Add attack summaries
    for name, result in detection_results.items():
        report += f"### {name.replace('_', ' ').title()}\n"
        report += f"- **Threat Level**: {result.threat_level}\n"
        report += f"- **Detection Confidence**: {result.confidence:.2f}\n"
        report += f"- **Attack Type**: {result.attack_type}\n"
        report += f"- **Detection Methods**: {', '.join(result.detection_methods)}\n"
        report += f"- **Evasion Techniques**: {', '.join(result.evasion_techniques)}\n\n"
    
    # Red team recommendations
    report += "## 🔴 RED TEAM RECOMMENDATIONS\n\n"
    report += "### Immediate Actions:\n"
    report += "1. **Review detector triggers**: Focus on low-confidence detections and ambiguous indicators\n"
    report += "2. **Design controlled test cases**: Validate detector boundaries using explicit, authorized evaluations\n"
    report += "3. **Measure robustness**: Track false positives/negatives across realistic distribution shifts\n"
    report += "4. **Document evaluation protocols**: Keep inputs, thresholds, and outputs reproducible\n\n"
    
    # Blue team recommendations
    report += "## 🔵 BLUE TEAM RECOMMENDATIONS\n\n"
    report += "### Immediate Actions:\n"
    report += "1. **Strengthen Detection Algorithms**: Improve low-confidence detections\n"
    report += "2. **Implement Real-Time Monitoring**: Deploy continuous threat detection\n"
    report += "3. **Enhance Threat Intelligence**: Integrate latest attack patterns\n"
    report += "4. **Deploy Adaptive Defenses**: Use machine learning for dynamic protection\n"
    report += "5. **Improve Response Times**: Optimize incident response workflows\n\n"
    
    # Technical details
    report += "## 📊 TECHNICAL ANALYSIS\n\n"
    
    denom = max(1, len(detection_results))
    high_threat_count = sum(1 for r in detection_results.values() if r.threat_level == 'HIGH')
    medium_threat_count = sum(1 for r in detection_results.values() if r.threat_level == 'MEDIUM')
    low_threat_count = sum(1 for r in detection_results.values() if r.threat_level == 'LOW')
    
    report += "### Threat Distribution:\n"
    report += f"- **High Threat**: {high_threat_count} scenarios ({high_threat_count/denom*100:.1f}%)\n"
    report += f"- **Medium Threat**: {medium_threat_count} scenarios ({medium_threat_count/denom*100:.1f}%)\n"
    report += f"- **Low Threat**: {low_threat_count} scenarios ({low_threat_count/denom*100:.1f}%)\n\n"
    
    avg_confidence = np.mean([r.confidence for r in detection_results.values()])
    report += "### Detection Performance:\n"
    report += f"- **Average Detection Confidence**: {avg_confidence:.2f}\n"
    report += f"- **High-confidence fraction (confidence > 0.7)**: {sum(1 for r in detection_results.values() if r.confidence > 0.7)/denom*100:.1f}%\n\n"
    
    report += "## 🎯 NEXT STEPS\n\n"
    report += "### For Red Teams:\n"
    report += "1. Focus on attacks with <70% detection confidence\n"
    report += "2. Develop hybrid attack strategies\n"
    report += "3. Test against updated defense mechanisms\n\n"
    
    report += "### For Blue Teams:\n"
    report += "1. Prioritize high-threat, low-confidence scenarios\n"
    report += "2. Implement multi-layer defense strategies\n"
    report += "3. Establish continuous monitoring protocols\n\n"
    
    report += f"---\n*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"
    report += "*Generated by NeurInSpectre from provided inputs and configured thresholds.*\n"
    
    return report

if __name__ == "__main__":
    # Example usage
    print("🎯 Red & Blue Team Dashboard System Ready!")
    print("Use create_comprehensive_report() to generate full analysis") 