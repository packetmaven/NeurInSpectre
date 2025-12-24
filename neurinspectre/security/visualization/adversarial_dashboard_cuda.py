#!/usr/bin/env python3
"""
CUDA-Accelerated Adversarial Robustness Dashboard
Comprehensive adversarial attack analysis and visualization using NVIDIA GPU acceleration
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json

# Avoid process-global warning suppression at import time. Silence warnings in your app entry point if desired.

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    import numpy as cp

try:
    import torchattacks
    TORCHATTACKS_AVAILABLE = True
except ImportError:
    TORCHATTACKS_AVAILABLE = False

class CUDAAdversarialRobustnessDashboard:
    """CUDA-accelerated adversarial robustness analysis and visualization"""
    
    def __init__(self, device_id: int = 0):
        """Initialize CUDA adversarial robustness dashboard
        
        Args:
            device_id: CUDA device ID
        """
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
            self.use_cuda = True
            print(f"✅ CUDA device {device_id} initialized for adversarial analysis")
        else:
            self.device = torch.device("cpu")
            self.use_cuda = False
            print("⚠️ CUDA not available. Using CPU mode.")
        
        if self.use_cuda and CUDA_AVAILABLE:
            try:
                self.cupy_device = cp.cuda.Device(device_id)
                # Configure CUDA optimization
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Memory optimization
                self.memory_pool = cp.get_default_memory_pool()
                self.memory_pool.set_limit(size=2**30)  # 1GB limit
                print("🚀 CUDA optimizations enabled")
            except Exception as e:
                print(f"⚠️ CUDA optimization failed: {e}")
                self.use_cuda = False
        
        # Initialize adversarial attack methods
        self.attacks = {}
        
    def initialize_attacks(self, model: torch.nn.Module):
        """Initialize adversarial attack methods
        
        Args:
            model: Target neural network model
        """
        model = model.to(self.device)
        
        if TORCHATTACKS_AVAILABLE:
            try:
                self.attacks = {
                    'FGSM': torchattacks.FGSM(model, eps=8/255),
                    'PGD': torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10),
                    'CW': torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01),
                }
                print(f"✅ {len(self.attacks)} adversarial attacks initialized")
            except Exception as e:
                print(f"⚠️ TorchAttacks initialization failed: {e}")
                self.attacks = {}
        else:
            # Implement basic attacks manually
            self.attacks = {
                'FGSM': self._fgsm_attack,
                'PGD': self._pgd_attack
            }
            print(f"✅ {len(self.attacks)} basic attacks implemented")
    
    def _fgsm_attack(self, data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module, eps: float = 0.031) -> torch.Tensor:
        """Basic FGSM implementation"""
        data = data.to(self.device).requires_grad_(True)
        target = target.to(self.device)
        
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        model.zero_grad()
        loss.backward()
        
        data_grad = data.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_data = data + eps * sign_data_grad
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        return perturbed_data.detach()
    
    def _pgd_attack(self, data: torch.Tensor, target: torch.Tensor, model: torch.nn.Module, 
                   eps: float = 0.031, alpha: float = 0.008, num_iter: int = 10) -> torch.Tensor:
        """Basic PGD implementation"""
        data = data.to(self.device)
        target = target.to(self.device)
        
        adv_data = data.clone().detach()
        
        for i in range(num_iter):
            adv_data.requires_grad_(True)
            output = model(adv_data)
            loss = torch.nn.functional.cross_entropy(output, target)
            
            model.zero_grad()
            loss.backward()
            
            adv_data = adv_data.detach() + alpha * adv_data.grad.sign()
            delta = torch.clamp(adv_data - data, min=-eps, max=eps)
            adv_data = torch.clamp(data + delta, min=0, max=1).detach()
        
        return adv_data
    
    def analyze_model_robustness(self, model: torch.nn.Module, 
                                dataloader: torch.utils.data.DataLoader,
                                attack_types: List[str] = None) -> Dict[str, Dict]:
        """Comprehensive robustness analysis using CUDA acceleration
        
        Args:
            model: Target neural network model
            dataloader: Data loader for test samples
            attack_types: List of attack types to evaluate
            
        Returns:
            Dictionary containing robustness analysis results
        """
        if attack_types is None:
            attack_types = list(self.attacks.keys())
        
        model = model.to(self.device)
        model.eval()
        
        results = {}
        layer_activations = {}
        
        # Hook for capturing layer activations
        def get_activation(name):
            def hook(model, input, output):
                layer_activations[name] = output.detach()
            return hook
        
        # Register hooks for convolutional and linear layers
        hooks = []
        for name, layer in model.named_modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(layer.register_forward_hook(get_activation(name)))
        
        for attack_name in attack_types:
            if attack_name not in self.attacks:
                continue
                
            print(f"🔍 Analyzing {attack_name} attack...")
            attack = self.attacks[attack_name]
            attack_results = {
                'success_rate': 0.0,
                'perturbation_norms': [],
                'confidence_drops': [],
                'layer_vulnerabilities': {},
                'gradient_magnitudes': []
            }
            
            total_samples = 0
            successful_attacks = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx >= 5:  # Limit for demonstration
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                
                # Generate adversarial examples
                if callable(attack):
                    # Custom attack function
                    if attack_name == 'FGSM':
                        adv_data = attack(data, target, model)
                    elif attack_name == 'PGD':
                        adv_data = attack(data, target, model)
                    else:
                        continue
                else:
                    # TorchAttacks object
                    try:
                        adv_data = attack(data, target)
                    except Exception as e:
                        print(f"⚠️ Attack {attack_name} failed: {e}")
                        continue
                
                # Original predictions
                with torch.no_grad():
                    orig_output = model(data)
                    orig_pred = orig_output.argmax(dim=1)
                    orig_confidence = torch.nn.functional.softmax(orig_output, dim=1).max(dim=1)[0]
                
                # Adversarial predictions
                with torch.no_grad():
                    adv_output = model(adv_data)
                    adv_pred = adv_output.argmax(dim=1)
                    adv_confidence = torch.nn.functional.softmax(adv_output, dim=1).max(dim=1)[0]
                
                # Calculate metrics
                perturbation = (adv_data - data).view(data.size(0), -1)
                perturbation_norms = torch.norm(perturbation, dim=1, p=2)
                
                confidence_drop = orig_confidence - adv_confidence
                
                # Success rate calculation
                attack_success = (orig_pred != adv_pred)
                successful_attacks += attack_success.sum().item()
                total_samples += data.size(0)
                
                # Store results using GPU acceleration if available
                if self.use_cuda and CUDA_AVAILABLE:
                    attack_results['perturbation_norms'].extend(
                        cp.asarray(perturbation_norms.detach().cpu().numpy())
                    )
                    attack_results['confidence_drops'].extend(
                        cp.asarray(confidence_drop.detach().cpu().numpy())
                    )
                else:
                    attack_results['perturbation_norms'].extend(
                        perturbation_norms.detach().cpu().numpy()
                    )
                    attack_results['confidence_drops'].extend(
                        confidence_drop.detach().cpu().numpy()
                    )
                
                # Gradient analysis
                data.requires_grad_(True)
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                
                grad_norms = torch.norm(data.grad.view(data.size(0), -1), dim=1)
                if self.use_cuda and CUDA_AVAILABLE:
                    attack_results['gradient_magnitudes'].extend(
                        cp.asarray(grad_norms.detach().cpu().numpy())
                    )
                else:
                    attack_results['gradient_magnitudes'].extend(
                        grad_norms.detach().cpu().numpy()
                    )
                
                # Layer vulnerability analysis
                for layer_name, activation in layer_activations.items():
                    if layer_name not in attack_results['layer_vulnerabilities']:
                        attack_results['layer_vulnerabilities'][layer_name] = []
                    
                    # Calculate activation statistics
                    activation_stats = {
                        'mean': torch.mean(activation).item(),
                        'std': torch.std(activation).item(),
                        'max': torch.max(activation).item(),
                        'sparsity': (activation == 0).float().mean().item()
                    }
                    attack_results['layer_vulnerabilities'][layer_name].append(activation_stats)
                
                layer_activations.clear()
            
            attack_results['success_rate'] = successful_attacks / total_samples if total_samples > 0 else 0.0
            results[attack_name] = attack_results
            print(f"✅ {attack_name}: {attack_results['success_rate']:.2%} misclassification rate")
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        return results
    
    def generate_robustness_dashboard(self, robustness_results: Dict[str, Dict]) -> go.Figure:
        """Generate comprehensive adversarial robustness dashboard
        
        Args:
            robustness_results: Results from robustness analysis
            
        Returns:
            Plotly figure with interactive dashboard
        """
        # Create subplot structure
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                '🎯 Attack Success Rates', '📏 Perturbation Distribution', '📉 Confidence Drop Analysis',
                '🌡️ Layer Vulnerability Heatmap', '⚡ Gradient Magnitudes', '📈 Robustness Timeline',
                '🔄 Attack Comparison Matrix', '🎲 Statistical Analysis', '🛡️ Defense Recommendations'
            ),
            specs=[
                [{"type": "bar"}, {"type": "histogram"}, {"type": "box"}],
                [{"type": "heatmap"}, {"type": "histogram"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "violin"}, {"type": "table"}]
            ]
        )
        
        attack_names = list(robustness_results.keys())
        if not attack_names:
            attack_names = ['No Attacks']
            success_rates = [0.0]
        else:
            success_rates = [robustness_results[attack]['success_rate'] for attack in attack_names]
        
        # Attack success rates bar chart
        colors = ['red' if rate > 0.5 else 'orange' if rate > 0.2 else 'green' for rate in success_rates]
        fig.add_trace(
            go.Bar(
                x=attack_names,
                y=success_rates,
                marker_color=colors,
                text=[f"{rate:.1%}" for rate in success_rates],
                textposition='auto',
                showlegend=False,
                hovertemplate='Attack: %{x}<br>Misclassification rate: %{y:.1%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Perturbation magnitude distribution
        all_perturbations = []
        for attack in attack_names:
            if attack in robustness_results and robustness_results[attack]['perturbation_norms']:
                if self.use_cuda and CUDA_AVAILABLE:
                    norms = robustness_results[attack]['perturbation_norms']
                    if isinstance(norms[0], cp.ndarray):
                        all_perturbations.extend([cp.asnumpy(norm) if hasattr(norm, 'asnumpy') else norm for norm in norms])
                    else:
                        all_perturbations.extend(norms)
                else:
                    all_perturbations.extend(robustness_results[attack]['perturbation_norms'])
        
        if all_perturbations:
            fig.add_trace(
                go.Histogram(
                    x=all_perturbations,
                    nbinsx=50,
                    showlegend=False,
                    marker_color='lightblue',
                    opacity=0.7,
                    hovertemplate='Perturbation: %{x:.4f}<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Confidence drop box plot
        for attack in attack_names:
            if attack in robustness_results:
                confidence_drops = robustness_results[attack]['confidence_drops']
                if confidence_drops:
                    if self.use_cuda and CUDA_AVAILABLE and hasattr(confidence_drops[0], 'asnumpy'):
                        drops = [cp.asnumpy(drop) if hasattr(drop, 'asnumpy') else drop for drop in confidence_drops]
                    else:
                        drops = confidence_drops
                    
                    fig.add_trace(
                        go.Box(
                            y=drops,
                            name=attack,
                            showlegend=False,
                            hovertemplate=f'{attack}<br>Confidence Drop: %{{y:.3f}}<extra></extra>'
                        ),
                        row=1, col=3
                    )
        
        # Layer vulnerability heatmap
        if attack_names and attack_names[0] in robustness_results:
            first_attack = robustness_results[attack_names[0]]
            if 'layer_vulnerabilities' in first_attack and first_attack['layer_vulnerabilities']:
                layer_names = list(first_attack['layer_vulnerabilities'].keys())[:10]  # Limit display
                vulnerability_matrix = []
                
                for attack in attack_names:
                    if attack not in robustness_results:
                        continue
                    attack_vulnerabilities = []
                    for layer in layer_names:
                        if layer in robustness_results[attack]['layer_vulnerabilities']:
                            layer_stats = robustness_results[attack]['layer_vulnerabilities'][layer]
                            if layer_stats:
                                avg_std = np.mean([stats['std'] for stats in layer_stats])
                                attack_vulnerabilities.append(avg_std)
                            else:
                                attack_vulnerabilities.append(0.0)
                        else:
                            attack_vulnerabilities.append(0.0)
                    vulnerability_matrix.append(attack_vulnerabilities)
                
                fig.add_trace(
                    go.Heatmap(
                        z=vulnerability_matrix,
                        x=[name.split('.')[-1][:8] for name in layer_names],  # Shorten layer names
                        y=attack_names,
                        colorscale='YlOrRd',
                        showscale=True,
                        colorbar=dict(title="Vulnerability", x=0.65),
                        hovertemplate='Attack: %{y}<br>Layer: %{x}<br>Vulnerability: %{z:.3f}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Gradient magnitude distribution
        all_gradients = []
        for attack in attack_names:
            if attack in robustness_results and robustness_results[attack]['gradient_magnitudes']:
                if self.use_cuda and CUDA_AVAILABLE:
                    grads = robustness_results[attack]['gradient_magnitudes']
                    if hasattr(grads[0], 'asnumpy'):
                        all_gradients.extend([cp.asnumpy(grad) if hasattr(grad, 'asnumpy') else grad for grad in grads])
                    else:
                        all_gradients.extend(grads)
                else:
                    all_gradients.extend(robustness_results[attack]['gradient_magnitudes'])
        
        if all_gradients:
            fig.add_trace(
                go.Histogram(
                    x=all_gradients,
                    nbinsx=50,
                    showlegend=False,
                    marker_color='lightgreen',
                    opacity=0.7,
                    hovertemplate='Gradient Magnitude: %{x:.4f}<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Robustness score timeline (simulated)
        timeline_x = list(range(len(attack_names)))
        robustness_scores = [1.0 - rate for rate in success_rates]
        
        fig.add_trace(
            go.Scatter(
                x=timeline_x,
                y=robustness_scores,
                mode='lines+markers',
                marker=dict(size=10, color=colors),
                showlegend=False,
                line=dict(width=3),
                hovertemplate='Attack: %{x}<br>Robustness: %{y:.1%}<extra></extra>'
            ),
            row=2, col=3
        )
        
        # Attack comparison matrix
        comparison_matrix = np.eye(len(attack_names))
        for i, attack1 in enumerate(attack_names):
            for j, attack2 in enumerate(attack_names):
                if i != j and attack1 in robustness_results and attack2 in robustness_results:
                    rate1 = robustness_results[attack1]['success_rate']
                    rate2 = robustness_results[attack2]['success_rate']
                    comparison_matrix[i][j] = 1.0 - abs(rate1 - rate2)
        
        fig.add_trace(
            go.Heatmap(
                z=comparison_matrix,
                x=attack_names,
                y=attack_names,
                colorscale='Blues',
                showscale=False,
                hovertemplate='Attack 1: %{y}<br>Attack 2: %{x}<br>Similarity: %{z:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Statistical analysis violin plot
        for attack in attack_names:
            if attack in robustness_results and robustness_results[attack]['confidence_drops']:
                confidence_drops = robustness_results[attack]['confidence_drops']
                if self.use_cuda and CUDA_AVAILABLE and hasattr(confidence_drops[0], 'asnumpy'):
                    drops = [cp.asnumpy(drop) if hasattr(drop, 'asnumpy') else drop for drop in confidence_drops]
                else:
                    drops = confidence_drops
                
                fig.add_trace(
                    go.Violin(
                        y=drops,
                        name=attack,
                        showlegend=False,
                        box_visible=True,
                        meanline_visible=True
                    ),
                    row=3, col=2
                )
        
        # Defense recommendations table
        recommendations = []
        priorities = []
        for attack in attack_names:
            if attack in robustness_results:
                success_rate = robustness_results[attack]['success_rate']
                if success_rate > 0.7:
                    recommendations.append("CRITICAL: Implement adversarial training")
                    priorities.append("🔴 High")
                elif success_rate > 0.3:
                    recommendations.append("IMPORTANT: Add input preprocessing")
                    priorities.append("🟡 Medium")
                else:
                    recommendations.append("MONITOR: Continue surveillance")
                    priorities.append("🟢 Low")
            else:
                recommendations.append("No data available")
                priorities.append("🔵 Info")
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Attack Type', 'Success Rate', 'Priority', 'Recommendation'],
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=[
                        attack_names,
                        [f"{robustness_results.get(attack, {}).get('success_rate', 0):.1%}" for attack in attack_names],
                        priorities,
                        recommendations
                    ],
                    fill_color='lavender',
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=3, col=3
        )
        
        device_info = "CUDA-Accelerated" if self.use_cuda else "CPU Mode"
        fig.update_layout(
            title=f"🛡️ Neural Network Adversarial Robustness Dashboard ({device_info})",
            height=1200,
            showlegend=False,
            template='plotly_dark',
            font=dict(size=12)
        )
        
        return fig

def create_simple_model() -> torch.nn.Module:
    """Create a simple model for testing"""
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    )

def create_sample_dataloader() -> torch.utils.data.DataLoader:
    """Create sample data for testing"""
    # Create dummy MNIST-like data
    data = torch.randn(100, 1, 28, 28)
    targets = torch.randint(0, 10, (100,))
    dataset = torch.utils.data.TensorDataset(data, targets)
    return torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

def create_security_dashboard(attack_results: Optional[Dict[str, Dict]] = None, output_path: str = None) -> go.Figure:
    """
    Create a comprehensive security dashboard from attack results
    
    Args:
        attack_results: Results from security analysis
        output_path: Optional path to save the dashboard
        
    Returns:
        Plotly figure with security dashboard
    """
    # Initialize dashboard
    dashboard = CUDAAdversarialRobustnessDashboard()
    
    # Require explicit results to avoid silently fabricating data.
    if attack_results is None:
        raise ValueError(
            "attack_results is required. This function does not generate sample results.\n"
            "Run a real analysis pipeline to produce attack results, or call the demo helpers explicitly."
        )
    
    # Generate dashboard
    fig = dashboard.generate_robustness_dashboard(attack_results)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="🛡️ NeurInSpectre Security Dashboard",
            x=0.5,
            font=dict(size=20, color='darkblue')
        ),
        showlegend=True,
        height=1000,
        template='plotly_white',
        hovermode='closest'
    )
    
    # Save if path provided
    if output_path:
        fig.write_html(output_path)
        print(f"✅ Security dashboard saved to: {output_path}")
    
    return fig


def create_comprehensive_security_report(security_data: Dict[str, Any], 
                                        save_path: str = None) -> Dict[str, Any]:
    """
    Create comprehensive security report from security data
    
    Args:
        security_data: Security analysis data
        save_path: Optional path to save report
        
    Returns:
        Comprehensive security report
    """
    report = {
        'timestamp': time.time(),
        'executive_summary': {
            'total_attacks_analyzed': len(security_data.get('attacks', {})),
            'overall_security_score': 0.0,
            'critical_vulnerabilities': []
        },
        'detailed_analysis': security_data,
        'recommendations': [
            "Implement adversarial training to improve robustness",
            "Deploy real-time attack detection systems",
            "Regular security assessments and updates",
            "Enhanced input validation and sanitization"
        ]
    }
    
    # Calculate overall security score
    if 'attacks' in security_data:
        success_rates = [data.get('success_rate', 0) for data in security_data['attacks'].values()]
        if success_rates:
            report['executive_summary']['overall_security_score'] = 1.0 - np.mean(success_rates)
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"✅ Security report saved to: {save_path}")
    
    return report


def generate_threat_visualization(threat_data: Dict[str, Any], 
                                 output_path: str = None) -> go.Figure:
    """
    Generate threat visualization from threat data
    
    Args:
        threat_data: Threat analysis data
        output_path: Optional path to save visualization
        
    Returns:
        Plotly figure with threat visualization
    """
    # Create threat visualization
    fig = go.Figure()
    
    # Add threat level indicators
    if 'threat_levels' in threat_data:
        threat_levels = threat_data['threat_levels']
        
        # Create threat level bar chart
        fig.add_trace(go.Bar(
            x=list(threat_levels.keys()),
            y=list(threat_levels.values()),
            marker_color=['red' if level > 0.8 else 'orange' if level > 0.5 else 'green' 
                         for level in threat_levels.values()],
            text=[f"{level:.1%}" for level in threat_levels.values()],
            textposition='auto',
            name='Threat Levels'
        ))
    
    # Add timeline if available
    if 'timeline' in threat_data:
        timeline = threat_data['timeline']
        fig.add_trace(go.Scatter(
            x=timeline.get('timestamps', []),
            y=timeline.get('threat_scores', []),
            mode='lines+markers',
            name='Threat Timeline',
            line=dict(color='red', width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title="🚨 Threat Visualization Dashboard",
        xaxis_title="Components",
        yaxis_title="Threat Level",
        showlegend=True,
        height=600,
        template='plotly_white'
    )
    
    # Save if path provided
    if output_path:
        fig.write_html(output_path)
        print(f"✅ Threat visualization saved to: {output_path}")
    
    return fig


def main():
    """Main function for testing the dashboard"""
    print("🛡️ NeurInSpectre CUDA Adversarial Robustness Dashboard")
    print("=" * 60)
    
    # Create simple model and data
    model = create_simple_model()
    dataloader = create_sample_dataloader()
    
    # Initialize dashboard
    dashboard = CUDAAdversarialRobustnessDashboard()
    dashboard.initialize_attacks(model)
    
    # Run robustness analysis
    print("\n🔍 Running robustness analysis...")
    results = dashboard.analyze_model_robustness(model, dataloader)
    
    # Generate dashboard
    print("\n📊 Generating dashboard...")
    fig = dashboard.generate_robustness_dashboard(results)
    
    # Save dashboard
    output_path = "adversarial_robustness_dashboard.html"
    fig.write_html(output_path)
    
    print(f"\n✅ Dashboard saved to: {output_path}")
    print("🌐 Open the file in your web browser to view the interactive dashboard")
    
    # Print summary
    print("\n📈 Analysis Summary:")
    for attack_name, attack_results in results.items():
        print(f"   • {attack_name}: {attack_results['success_rate']:.1%} misclassification rate")
    
    return fig


if __name__ == "__main__":
    main() 