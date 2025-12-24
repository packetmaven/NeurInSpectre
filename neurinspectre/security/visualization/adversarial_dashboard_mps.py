#!/usr/bin/env python3
"""
Apple Silicon MPS-Optimized Adversarial Robustness Dashboard
Comprehensive adversarial attack analysis and visualization using Apple's Metal Performance Shaders
"""

import torch
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MPSAdversarialRobustnessDashboard:
    """MPS-accelerated adversarial robustness analysis optimized for Apple Silicon"""
    
    def __init__(self, memory_fraction: float = 0.8):
        """Initialize MPS adversarial robustness dashboard
        
        Args:
            memory_fraction: Fraction of unified memory to use
        """
        if not torch.backends.mps.is_available():
            print("⚠️ MPS not available on this system. Falling back to CPU mode.")
            self.device = torch.device("cpu")
            self.use_mps = False
        else:
            try:
                self.device = torch.device("mps")
                
                # Configure MPS memory management for unified memory architecture
                torch.mps.set_per_process_memory_fraction(memory_fraction)
                torch.mps.empty_cache()
                
                self.use_mps = True
                print("✅ Apple MPS initialized for adversarial robustness analysis")
                print(f"🍎 Using {memory_fraction:.0%} of unified memory")
            except Exception as e:
                print(f"⚠️ MPS initialization failed: {e}. Falling back to CPU mode.")
                self.device = torch.device("cpu")
                self.use_mps = False
        
        # Enable channels-last memory format for better MPS performance
        self.memory_format = torch.channels_last if self.use_mps else torch.contiguous_format
        
        # MPS-optimized adversarial attack implementations
        self.attacks = {}
        
        print(f"🎯 Adversarial robustness analysis ready on {self.device}")
    
    def fgsm_attack_mps(self, model: torch.nn.Module, data: torch.Tensor, 
                       target: torch.Tensor, epsilon: float = 0.031) -> torch.Tensor:
        """MPS-optimized Fast Gradient Sign Method implementation
        
        Args:
            model: Target neural network model
            data: Input data tensor
            target: Target labels
            epsilon: Perturbation magnitude
            
        Returns:
            Adversarial examples tensor
        """
        data = data.to(self.device, memory_format=self.memory_format)
        target = target.to(self.device)
        data.requires_grad_(True)
        
        # Forward pass
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        # Backward pass to get gradients
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples using MPS-optimized operations
        data_grad = data.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_data = data + epsilon * sign_data_grad
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        return perturbed_data.detach()
    
    def pgd_attack_mps(self, model: torch.nn.Module, data: torch.Tensor, 
                      target: torch.Tensor, epsilon: float = 0.031, 
                      alpha: float = 0.008, num_iter: int = 10) -> torch.Tensor:
        """MPS-optimized Projected Gradient Descent implementation
        
        Args:
            model: Target neural network model
            data: Input data tensor
            target: Target labels
            epsilon: Maximum perturbation magnitude
            alpha: Step size
            num_iter: Number of iterations
            
        Returns:
            Adversarial examples tensor
        """
        data = data.to(self.device, memory_format=self.memory_format)
        target = target.to(self.device)
        
        # Initialize adversarial examples
        adv_data = data.clone().detach()
        
        for i in range(num_iter):
            adv_data.requires_grad_(True)
            
            # Forward pass
            output = model(adv_data)
            loss = torch.nn.functional.cross_entropy(output, target)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            adv_data = adv_data.detach() + alpha * adv_data.grad.sign()
            delta = torch.clamp(adv_data - data, min=-epsilon, max=epsilon)
            adv_data = torch.clamp(data + delta, min=0, max=1).detach()
        
        return adv_data
    
    def cw_attack_mps(self, model: torch.nn.Module, data: torch.Tensor, 
                     target: torch.Tensor, c: float = 1.0, kappa: float = 0.0,
                     num_iter: int = 100, lr: float = 0.01) -> torch.Tensor:
        """MPS-optimized Carlini & Wagner attack implementation
        
        Args:
            model: Target neural network model
            data: Input data tensor
            target: Target labels
            c: Confidence parameter
            kappa: Confidence margin
            num_iter: Number of iterations
            lr: Learning rate
            
        Returns:
            Adversarial examples tensor
        """
        data = data.to(self.device, memory_format=self.memory_format)
        target = target.to(self.device)
        
        # Initialize adversarial variable
        w = torch.zeros_like(data, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([w], lr=lr)
        
        for i in range(num_iter):
            # Convert w to adversarial examples
            adv_data = 0.5 * (torch.tanh(w) + 1.0)
            
            # Forward pass
            output = model(adv_data)
            
            # Calculate CW loss
            target_logits = torch.gather(output, 1, target.unsqueeze(1)).squeeze(1)
            max_other_logits = torch.max(
                output - 1000 * torch.nn.functional.one_hot(target, output.size(1)).float(), 
                dim=1
            )[0]
            
            loss_adv = torch.clamp(max_other_logits - target_logits + kappa, min=0)
            loss_dist = torch.norm((adv_data - data).view(data.size(0), -1), dim=1)
            loss = loss_dist + c * loss_adv
            
            # Backward pass
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        
        with torch.no_grad():
            adv_data = 0.5 * (torch.tanh(w) + 1.0)
        
        return adv_data.detach()
    
    def analyze_model_robustness_mps(self, model: torch.nn.Module, 
                                    dataloader: torch.utils.data.DataLoader,
                                    attack_types: List[str] = None) -> Dict[str, Dict]:
        """Comprehensive robustness analysis using MPS acceleration
        
        Args:
            model: Target neural network model
            dataloader: Data loader for test samples
            attack_types: List of attack types to evaluate
            
        Returns:
            Dictionary containing robustness analysis results
        """
        if attack_types is None:
            attack_types = ['FGSM', 'PGD', 'CW']
        
        model = model.to(self.device, memory_format=self.memory_format)
        model.eval()
        
        results = {}
        
        # Define attack methods
        attack_methods = {
            'FGSM': lambda m, d, t: self.fgsm_attack_mps(m, d, t),
            'PGD': lambda m, d, t: self.pgd_attack_mps(m, d, t),
            'CW': lambda m, d, t: self.cw_attack_mps(m, d, t)
        }
        
        for attack_name in attack_types:
            if attack_name not in attack_methods:
                continue
            
            print(f"🔍 Analyzing {attack_name} attack with MPS acceleration...")
            attack_method = attack_methods[attack_name]
            attack_results = {
                'success_rate': 0.0,
                'perturbation_norms': [],
                'confidence_drops': [],
                'layer_vulnerabilities': {},
                'gradient_magnitudes': [],
                'robustness_scores': []
            }
            
            total_samples = 0
            successful_attacks = 0
            
            # Hook for capturing layer activations
            layer_activations = {}
            def get_activation(name):
                def hook(model, input, output):
                    layer_activations[name] = output.detach()
                return hook
            
            # Register hooks for analysis
            hooks = []
            for name, layer in model.named_modules():
                if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                    hooks.append(layer.register_forward_hook(get_activation(name)))
            
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx >= 5:  # Limit for demonstration
                    break
                
                # Convert to MPS with optimal memory format
                data = data.to(self.device, memory_format=self.memory_format)
                target = target.to(self.device)
                
                # Original predictions
                with torch.no_grad():
                    orig_output = model(data)
                    orig_pred = orig_output.argmax(dim=1)
                    orig_confidence = torch.nn.functional.softmax(orig_output, dim=1).max(dim=1)[0]
                
                # Generate adversarial examples
                try:
                    adv_data = attack_method(model, data, target)
                except Exception as e:
                    print(f"⚠️ Attack {attack_name} failed: {e}")
                    continue
                
                # Adversarial predictions
                with torch.no_grad():
                    adv_output = model(adv_data)
                    adv_pred = adv_output.argmax(dim=1)
                    adv_confidence = torch.nn.functional.softmax(adv_output, dim=1).max(dim=1)[0]
                
                # Calculate metrics using MPS-optimized operations
                perturbation = (adv_data - data).view(data.size(0), -1)
                perturbation_norms = torch.norm(perturbation, dim=1, p=2)
                confidence_drop = orig_confidence - adv_confidence
                
                # Success rate calculation
                attack_success = (orig_pred != adv_pred)
                successful_attacks += attack_success.sum().item()
                total_samples += data.size(0)
                
                # Store results (transfer to CPU for storage)
                attack_results['perturbation_norms'].extend(perturbation_norms.detach().cpu().numpy())
                attack_results['confidence_drops'].extend(confidence_drop.detach().cpu().numpy())
                
                # Gradient analysis
                data_grad = data.clone().detach().requires_grad_(True)
                output = model(data_grad)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                
                grad_norms = torch.norm(data_grad.grad.view(data.size(0), -1), dim=1)
                attack_results['gradient_magnitudes'].extend(grad_norms.detach().cpu().numpy())
                
                # Calculate robustness score
                robustness_score = 1.0 - (perturbation_norms.mean() / confidence_drop.mean().clamp(min=1e-8))
                attack_results['robustness_scores'].append(robustness_score.detach().cpu().item())
                
                # Layer vulnerability analysis
                for layer_name, activation in layer_activations.items():
                    if layer_name not in attack_results['layer_vulnerabilities']:
                        attack_results['layer_vulnerabilities'][layer_name] = []
                    
                    # Calculate activation statistics using MPS operations
                    activation_stats = {
                        'mean': torch.mean(activation).detach().cpu().item(),
                        'std': torch.std(activation).detach().cpu().item(),
                        'max': torch.max(activation).detach().cpu().item(),
                        'sparsity': (activation == 0).float().mean().detach().cpu().item(),
                        'sensitivity': torch.std(activation.view(activation.size(0), -1), dim=1).mean().detach().cpu().item()
                    }
                    attack_results['layer_vulnerabilities'][layer_name].append(activation_stats)
                
                layer_activations.clear()
                
                # Memory management for MPS
                if batch_idx % 2 == 0:
                    self.optimize_memory_usage()
            
            attack_results['success_rate'] = successful_attacks / total_samples if total_samples > 0 else 0.0
            results[attack_name] = attack_results
            print(f"✅ {attack_name}: {attack_results['success_rate']:.2%} misclassification rate")
            
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        return results
    
    def generate_robustness_dashboard_mps(self, robustness_results: Dict[str, Dict]) -> go.Figure:
        """Generate comprehensive adversarial robustness dashboard using MPS results
        
        Args:
            robustness_results: Results from MPS robustness analysis
            
        Returns:
            Plotly figure with interactive dashboard
        """
        # Create comprehensive subplot layout
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                '🎯 Attack Success Rates', '📏 Perturbation Distribution', '📉 Confidence Analysis',
                '🌡️ Layer Vulnerability Map', '⚡ Gradient Magnitudes', '📈 Robustness Timeline',
                '🔄 Attack Effectiveness Matrix', '🎲 Sensitivity Analysis', '🛡️ Defense Metrics',
                '📊 Statistical Summary', '🚨 Threat Assessment', '📋 Recommendations'
            ),
            specs=[
                [{"type": "bar"}, {"type": "histogram"}, {"type": "box"}],
                [{"type": "heatmap"}, {"type": "histogram"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "indicator"}, {"type": "table"}]
            ]
        )
        
        attack_names = list(robustness_results.keys())
        if not attack_names:
            attack_names = ['No Attacks']
            success_rates = [0.0]
        else:
            success_rates = [robustness_results[attack]['success_rate'] for attack in attack_names]
        
        # Attack success rates with color coding
        colors = ['red' if rate > 0.7 else 'orange' if rate > 0.3 else 'green' for rate in success_rates]
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
            if attack in robustness_results:
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
        
        # Confidence drop analysis
        for attack in attack_names:
            if attack in robustness_results:
                confidence_drops = robustness_results[attack]['confidence_drops']
                if confidence_drops:
                    fig.add_trace(
                        go.Box(
                            y=confidence_drops,
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
                                avg_sensitivity = np.mean([stats.get('sensitivity', 0) for stats in layer_stats])
                                attack_vulnerabilities.append(avg_sensitivity)
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
                        colorbar=dict(title="Sensitivity", x=0.65),
                        hovertemplate='Attack: %{y}<br>Layer: %{x}<br>Sensitivity: %{z:.3f}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Gradient magnitude distribution
        all_gradients = []
        for attack in attack_names:
            if attack in robustness_results:
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
        
        # Robustness score timeline
        for attack in attack_names:
            if attack in robustness_results:
                robustness_scores = robustness_results[attack].get('robustness_scores', [])
                if robustness_scores:
                    fig.add_trace(
                        go.Scatter(
                            y=robustness_scores,
                            mode='lines+markers',
                            name=f"{attack} Robustness",
                            showlegend=False,
                            hovertemplate=f'{attack}<br>Batch: %{{x}}<br>Robustness: %{{y:.2f}}<extra></extra>'
                        ),
                        row=2, col=3
                    )
        
        # Overall robustness assessment
        overall_robustness = 1.0 - np.mean(success_rates) if success_rates else 0.5
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_robustness * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "🍎 MPS Robustness %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen" if overall_robustness > 0.7 else "orange" if overall_robustness > 0.3 else "red"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=4, col=2
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
                    recommendations.append("IMPORTANT: Add input validation")
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
            row=4, col=3
        )
        
        device_info = "Apple MPS-Accelerated" if self.use_mps else "CPU Mode"
        fig.update_layout(
            title=f"🛡️ Neural Network Adversarial Robustness Dashboard ({device_info})",
            height=1400,
            showlegend=False,
            template='plotly_dark',
            font=dict(size=12)
        )
        
        return fig
    
    def optimize_memory_usage(self):
        """Optimize memory usage for Apple Silicon unified memory"""
        if self.use_mps:
            # Clear MPS cache
            torch.mps.empty_cache()
            
            # Synchronize MPS operations
            torch.mps.synchronize()

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

def main():
    """Test the MPS adversarial robustness dashboard"""
    print("🍎 Testing Apple MPS Adversarial Robustness Dashboard")
    
    # Initialize dashboard
    dashboard = MPSAdversarialRobustnessDashboard()
    
    # Create simple model and data
    model = create_simple_model()
    dataloader = create_sample_dataloader()
    
    print(f"🤖 Created simple model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"📊 Created sample dataloader with {len(dataloader)} batches")
    
    # Analyze robustness
    results = dashboard.analyze_model_robustness_mps(model, dataloader, ['FGSM', 'PGD', 'CW'])
    print("✅ Robustness analysis completed")
    
    # Generate dashboard
    fig = dashboard.generate_robustness_dashboard_mps(results)
    
    # Save dashboard
    output_file = "adversarial_robustness_mps.html"
    fig.write_html(output_file)
    print(f"💾 Dashboard saved to {output_file}")
    
    # Print summary
    print("\n📊 Results Summary:")
    for attack_name, attack_results in results.items():
        print(f"   {attack_name}: {attack_results['success_rate']:.2%} misclassification rate")
    
    # Optimize memory
    dashboard.optimize_memory_usage()
    print("🍎 Memory optimization completed")

if __name__ == "__main__":
    main() 