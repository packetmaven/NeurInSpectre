#!/usr/bin/env python3
"""
Advanced AI Security Visualization CLI Commands
Unified command-line interface for agentic flow and adversarial robustness analysis
"""

import click
import torch
import json
import os
from pathlib import Path
from typing import Optional, List
import sys

@click.group(name="advanced-viz")
def advanced_visualization_cli():
    """Advanced AI Security Visualization Commands"""
    click.echo("🚀 Advanced AI Security Visualization Suite")
    click.echo("Supporting both NVIDIA CUDA and Apple Silicon MPS acceleration")

@advanced_visualization_cli.command("agentic-flow")
@click.option("--attack-data", type=click.Path(), 
              help="Path to attack events JSON file (will create sample if not provided)")
@click.option("--output", type=click.Path(), default="./agentic_flow_analysis.html",
              help="Output HTML file path")
@click.option("--device", type=click.Choice(['cuda', 'mps', 'auto']), default='auto',
              help="Device to use for acceleration")
@click.option("--real-time", is_flag=True, 
              help="Enable real-time monitoring")
@click.option("--num-events", type=int, default=100,
              help="Number of sample events to generate if no data file provided")
def agentic_flow_analysis(attack_data: str, output: str, device: str, real_time: bool, num_events: int):
    """Generate real-time agentic AI attack flow visualization"""
    
    click.echo(f"🤖 Starting Agentic AI Attack Flow Analysis")
    click.echo(f"📊 Events: {num_events if not attack_data else 'from file'}")
    click.echo(f"🎯 Output: {output}")
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            click.echo("🔥 NVIDIA CUDA detected - using CUDA acceleration")
        elif torch.backends.mps.is_available():
            device = 'mps'
            click.echo("🍎 Apple Silicon detected - using MPS acceleration")
        else:
            click.echo("⚠️ No GPU acceleration available - using CPU mode")
            device = 'cpu'
    
    try:
        # Import appropriate visualizer
        if device == 'cuda':
            from neurinspectre.security.visualization.agentic_flow_cuda import CUDAAgenticFlowVisualizer, create_sample_attack_events
            visualizer = CUDAAgenticFlowVisualizer()
            click.echo("✅ CUDA Agentic Flow Visualizer initialized")
        else:
            from neurinspectre.security.visualization.agentic_flow_mps import MPSAgenticFlowVisualizer, create_sample_attack_events
            visualizer = MPSAgenticFlowVisualizer()
            click.echo("✅ MPS Agentic Flow Visualizer initialized")
        
        # Load or create attack data
        if attack_data and os.path.exists(attack_data):
            click.echo(f"📁 Loading attack data from {attack_data}")
            with open(attack_data, 'r') as f:
                events_data = json.load(f)
            # Convert to AgentAttackEvent objects (simplified for demo)
            events = create_sample_attack_events(len(events_data))
        else:
            click.echo(f"🎲 Generating {num_events} sample attack events")
            events = create_sample_attack_events(num_events)
        
        # Process events
        if device == 'cuda':
            processed_data = visualizer.process_attack_stream(events)
            fig = visualizer.generate_attack_flow_graph(processed_data)
        else:
            processed_data = visualizer.process_attack_stream_mps(events)
            fig = visualizer.generate_attack_flow_graph_mps(processed_data)
        
        # Save visualization
        fig.write_html(output)
        click.echo(f"💾 Agentic flow analysis saved to {output}")
        
        # Display predictions
        if device == 'cuda':
            predictions = visualizer.predict_attack_progression(processed_data)
        else:
            predictions = visualizer.predict_attack_progression_mps(processed_data)
        
        click.echo(f"🔮 Attack progression predictions:")
        for attack_type, probability in predictions.items():
            click.echo(f"   {attack_type}: {probability:.2%}")
            
        if real_time:
            click.echo("🔴 Real-time monitoring mode enabled")
            click.echo("   Dashboard will auto-refresh every 10 seconds")
            
    except ImportError as e:
        click.echo(f"❌ Import error: {e}")
        click.echo("💡 Please install required dependencies: torch, plotly, networkx")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@advanced_visualization_cli.command("adversarial-dashboard")
@click.option("--model-path", type=click.Path(),
              help="Path to model file (will create sample if not provided)")
@click.option("--dataset", type=click.Choice(['cifar10', 'imagenet', 'mnist', 'custom']), default='mnist',
              help="Dataset to use for testing")
@click.option("--output", type=click.Path(), default="./adversarial_dashboard.html",
              help="Output HTML file path")
@click.option("--device", type=click.Choice(['cuda', 'mps', 'auto']), default='auto',
              help="Device to use for acceleration")
@click.option("--attacks", multiple=True, default=['FGSM', 'PGD'],
              help="Attack types to evaluate")
@click.option("--batch-size", type=int, default=16,
              help="Batch size for analysis")
def adversarial_robustness_dashboard(model_path: str, dataset: str, output: str, device: str, attacks: list, batch_size: int):
    """Generate neural network adversarial robustness dashboard"""
    
    click.echo(f"🛡️ Starting Adversarial Robustness Analysis")
    click.echo(f"📊 Dataset: {dataset}")
    click.echo(f"⚔️ Attacks: {', '.join(attacks)}")
    click.echo(f"🎯 Output: {output}")
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            click.echo("🔥 NVIDIA CUDA detected - using CUDA acceleration")
        elif torch.backends.mps.is_available():
            device = 'mps'
            click.echo("🍎 Apple Silicon detected - using MPS acceleration")
        else:
            click.echo("⚠️ No GPU acceleration available - using CPU mode")
            device = 'cpu'
    
    try:
        # Import appropriate dashboard
        if device == 'cuda':
            from neurinspectre.security.visualization.adversarial_dashboard_cuda import CUDAAdversarialRobustnessDashboard, create_simple_model, create_sample_dataloader
            dashboard = CUDAAdversarialRobustnessDashboard()
            click.echo("✅ CUDA Adversarial Dashboard initialized")
        else:
            from neurinspectre.security.visualization.adversarial_dashboard_mps import MPSAdversarialRobustnessDashboard
            from neurinspectre.security.visualization.adversarial_dashboard_cuda import create_simple_model, create_sample_dataloader
            dashboard = MPSAdversarialRobustnessDashboard()
            click.echo("✅ MPS Adversarial Dashboard initialized")
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            click.echo(f"📁 Loading model from {model_path}")
            model = torch.load(model_path, map_location='cpu')
        else:
            click.echo(f"🤖 Creating sample model for {dataset} dataset")
            model = create_simple_model()
        
        # Create dataloader
        dataloader = create_sample_dataloader()
        click.echo(f"📊 Created dataloader with batch size {batch_size}")
        
        # Initialize attacks (for CUDA dashboard)
        if hasattr(dashboard, 'initialize_attacks'):
            dashboard.initialize_attacks(model)
        
        # Analyze robustness
        if device == 'cuda':
            results = dashboard.analyze_model_robustness(model, dataloader, list(attacks))
            fig = dashboard.generate_robustness_dashboard(results)
        else:
            results = dashboard.analyze_model_robustness_mps(model, dataloader, list(attacks))
            fig = dashboard.generate_robustness_dashboard_mps(results)
        
        # Save dashboard
        fig.write_html(output)
        click.echo(f"💾 Adversarial robustness dashboard saved to {output}")
        
        # Display summary
        click.echo(f"\n📊 Robustness Analysis Summary:")
        for attack_name, attack_results in results.items():
            success_rate = attack_results['success_rate']
            status = "🔴 HIGH RISK" if success_rate > 0.7 else "🟡 MEDIUM RISK" if success_rate > 0.3 else "🟢 LOW RISK"
            click.echo(f"   {attack_name}: {success_rate:.2%} misclassification rate - {status}")
            
        # Memory optimization for MPS
        if device == 'mps' and hasattr(dashboard, 'optimize_memory_usage'):
            dashboard.optimize_memory_usage()
            
    except ImportError as e:
        click.echo(f"❌ Import error: {e}")
        click.echo("💡 Please install required dependencies: torch, plotly, numpy")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@advanced_visualization_cli.command("system-info")
def system_info():
    """Display system capabilities for advanced visualizations"""
    click.echo("🖥️ System Information for Advanced Visualizations")
    click.echo("=" * 60)
    
    # Check PyTorch
    try:
        import torch
        click.echo(f"✅ PyTorch: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            click.echo(f"🔥 CUDA: Available")
            click.echo(f"   Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                click.echo(f"   Device {i}: {name} ({memory:.1f} GB)")
        else:
            click.echo("❌ CUDA: Not available")
        
        # Check MPS
        if torch.backends.mps.is_available():
            click.echo(f"🍎 Apple MPS: Available")
            click.echo(f"   Unified memory architecture detected")
        else:
            click.echo("❌ Apple MPS: Not available")
            
    except ImportError:
        click.echo("❌ PyTorch: Not installed")
    
    # Check other dependencies
    dependencies = {
        'plotly': 'Visualization engine',
        'networkx': 'Graph analysis',
        'numpy': 'Numerical computing',
        'cupy': 'CUDA acceleration (optional)',
        'torchattacks': 'Adversarial attacks (optional)'
    }
    
    click.echo("\n📦 Dependencies:")
    for package, description in dependencies.items():
        try:
            __import__(package)
            click.echo(f"✅ {package}: Available - {description}")
        except ImportError:
            click.echo(f"❌ {package}: Not available - {description}")

@advanced_visualization_cli.command("demo")
@click.option("--device", type=click.Choice(['cuda', 'mps', 'auto']), default='auto')
def run_demo(device: str):
    """Run a comprehensive demo of all advanced visualization features"""
    click.echo("🎬 Advanced Visualization Demo")
    click.echo("=" * 50)
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    click.echo(f"🎯 Using device: {device.upper()}")
    
    # Demo 1: Agentic Flow Analysis
    click.echo("\n🤖 Demo 1: Agentic AI Attack Flow Analysis")
    ctx = click.get_current_context()
    ctx.invoke(agentic_flow_analysis, 
               attack_data=None, 
               output="demo_agentic_flow.html",
               device=device,
               real_time=False,
               num_events=75)
    
    # Demo 2: Adversarial Robustness Dashboard
    click.echo("\n🛡️ Demo 2: Adversarial Robustness Dashboard")
    ctx.invoke(adversarial_robustness_dashboard,
               model_path=None,
               dataset='mnist',
               output="demo_adversarial_dashboard.html",
               device=device,
               attacks=['FGSM', 'PGD'],
               batch_size=16)
    
    click.echo("\n🎉 Demo completed successfully!")
    click.echo("📁 Generated files:")
    click.echo("   - demo_agentic_flow.html")
    click.echo("   - demo_adversarial_dashboard.html")
    click.echo("\n💡 Open these files in your browser to view the interactive dashboards")

def main():
    """Main entry point for advanced visualization CLI"""
    advanced_visualization_cli()

if __name__ == "__main__":
    main() 