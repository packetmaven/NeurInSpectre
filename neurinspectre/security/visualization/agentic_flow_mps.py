#!/usr/bin/env python3
"""
Apple Silicon MPS-Optimized Agentic AI Attack Flow Visualizer
Real-time visualization of autonomous agent attack patterns using Apple's Metal Performance Shaders
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

@dataclass
class AgentAttackEvent:
    """Structured representation of autonomous agent attack events"""
    agent_id: str
    timestamp: datetime
    attack_type: str
    target_system: str
    confidence_score: float
    lateral_movement: bool
    persistence_indicators: List[str]
    ttl_seconds: int

class MPSAgenticFlowVisualizer:
    """Real-time agentic AI attack flow visualization optimized for Apple Silicon"""
    
    def __init__(self, memory_fraction: float = 0.8):
        """Initialize MPS-optimized agentic flow visualizer
        
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
                print(f"✅ Apple MPS initialized for agentic flow visualization")
                print(f"🍎 Using {memory_fraction:.0%} of unified memory")
            except Exception as e:
                print(f"⚠️ MPS initialization failed: {e}. Falling back to CPU mode.")
                self.device = torch.device("cpu")
                self.use_mps = False
        
        # Enable channels-last memory format for better MPS performance
        self.memory_format = torch.channels_last if self.use_mps else torch.contiguous_format
        
        # Initialize tracking tensors with MPS optimization
        if self.use_mps:
            self.agent_interactions = torch.zeros(1000, 1000, device=self.device, dtype=torch.float32)
            self.temporal_patterns = torch.zeros(1000, 24, device=self.device, dtype=torch.float32)
            self.attack_velocity = torch.zeros(1000, 10, device=self.device, dtype=torch.float32)
        else:
            self.agent_interactions = torch.zeros(1000, 1000, dtype=torch.float32)
            self.temporal_patterns = torch.zeros(1000, 24, dtype=torch.float32)
            self.attack_velocity = torch.zeros(1000, 10, dtype=torch.float32)
        
        # MPS-optimized neural network for pattern recognition  
        self.pattern_network = torch.nn.Sequential(
            torch.nn.Linear(256, 128),  # Input: flattened features [64x4=256]
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)  # 10 attack pattern classes
        ).to(self.device)
        
        # Event buffer for real-time processing
        self.event_buffer = []
        self.max_buffer_size = 10000
        
        print(f"🎯 Pattern recognition network ready on {self.device}")
    
    def process_attack_stream_mps(self, events: List[AgentAttackEvent]) -> Dict[str, torch.Tensor]:
        """Process streaming attack data with MPS acceleration
        
        Args:
            events: List of agent attack events
            
        Returns:
            Dictionary containing processed attack tensors
        """
        if not events:
            return self._get_empty_results_mps()
        
        # Convert events to MPS tensors
        agent_ids = torch.tensor([hash(event.agent_id) % 1000 for event in events], 
                                device=self.device, dtype=torch.long)
        timestamps = torch.tensor([event.timestamp.timestamp() for event in events], 
                                 device=self.device, dtype=torch.float32)
        attack_types = torch.tensor([hash(event.attack_type) % 10 for event in events], 
                                   device=self.device, dtype=torch.long)
        confidence_scores = torch.tensor([event.confidence_score for event in events], 
                                        device=self.device, dtype=torch.float32)
        
        # Compute temporal patterns using MPS-optimized operations
        time_bins = torch.floor((timestamps - timestamps.min()) / 3600) % 24
        
        # Update interaction matrix using scatter operations (MPS-optimized)
        if len(agent_ids) > 1:
            edge_indices = torch.stack([agent_ids[:-1], agent_ids[1:]], dim=0)
            edge_weights = torch.ones(len(agent_ids) - 1, device=self.device)
            self.agent_interactions.index_put_((edge_indices[0], edge_indices[1]), 
                                              edge_weights, accumulate=True)
        
        # Calculate attack velocity using sliding window convolution (MPS-optimized)
        window_size = 300  # 5-minute window
        velocity_kernel = torch.ones(window_size, device=self.device) / window_size
        
        # Pad confidence scores for convolution
        if len(confidence_scores) >= window_size:
            padded_scores = torch.nn.functional.pad(confidence_scores, (window_size//2, window_size//2))
            velocities = torch.nn.functional.conv1d(
                padded_scores.unsqueeze(0).unsqueeze(0),
                velocity_kernel.unsqueeze(0).unsqueeze(0),
                padding=0
            ).squeeze()
        else:
            # Fallback for small datasets
            velocities = confidence_scores.clone()
        
        # Detect anomalies using statistical methods (MPS-optimized)
        mean_velocity = torch.mean(velocities)
        std_velocity = torch.std(velocities)
        anomaly_scores = torch.abs(velocities - mean_velocity) / (std_velocity + 1e-8)
        
        # Pattern classification using neural network
        feature_matrix = torch.stack([
            agent_ids.float(),
            attack_types.float(),
            confidence_scores,
            velocities[:len(agent_ids)]
        ], dim=1)
        
        # Pad features to fixed size for neural network
        if feature_matrix.shape[0] < 64:
            padding = torch.zeros(64 - feature_matrix.shape[0], 4, device=self.device)
            feature_matrix = torch.cat([feature_matrix, padding], dim=0)
        else:
            feature_matrix = feature_matrix[:64]
        
        # Flatten for neural network input - should give 256 features (64x4)
        flattened_features = feature_matrix.flatten().unsqueeze(0)  # Shape: [1, 256]
        
        with torch.no_grad():
            pattern_predictions = torch.nn.functional.softmax(
                self.pattern_network(flattened_features), dim=1
            )
        
        return {
            'agent_interactions': self.agent_interactions,
            'temporal_patterns': self.temporal_patterns,
            'attack_velocity': velocities,
            'anomaly_scores': anomaly_scores,
            'pattern_predictions': pattern_predictions,
            'feature_matrix': feature_matrix
        }
    
    def _get_empty_results_mps(self) -> Dict[str, torch.Tensor]:
        """Return empty results structure for MPS"""
        return {
            'agent_interactions': torch.zeros(10, 10, device=self.device),
            'temporal_patterns': torch.zeros(10, 24, device=self.device),
            'attack_velocity': torch.tensor([], device=self.device),
            'anomaly_scores': torch.tensor([], device=self.device),
            'pattern_predictions': torch.zeros(1, 10, device=self.device),
            'feature_matrix': torch.zeros(1, 4, device=self.device)
        }
    
    def generate_attack_flow_graph_mps(self, processed_data: Dict[str, torch.Tensor]) -> go.Figure:
        """Generate interactive attack flow visualization using MPS-processed data
        
        Args:
            processed_data: Processed attack data from MPS operations
            
        Returns:
            Plotly figure with interactive attack flow graph
        """
        # Transfer data to CPU for visualization
        interactions_cpu = processed_data['agent_interactions'].cpu().numpy()
        velocities_cpu = processed_data['attack_velocity'].cpu().numpy()
        anomaly_scores_cpu = processed_data['anomaly_scores'].cpu().numpy()
        pattern_predictions_cpu = processed_data['pattern_predictions'].cpu().numpy()
        
        # Create NetworkX graph from interaction matrix
        G = nx.from_numpy_array(interactions_cpu, create_using=nx.DiGraph)
        
        # Handle empty graphs
        if G.number_of_nodes() == 0:
            G.add_node(0)
        
        # Calculate node positions using force-directed layout
        try:
            pos = nx.spring_layout(G, k=1/np.sqrt(max(G.number_of_nodes(), 1)), iterations=50)
        except:
            pos = {node: (np.random.random(), np.random.random()) for node in G.nodes()}
        
        if not pos:  # Handle empty graph
            pos = {0: (0, 0)}
        
        # Extract node and edge information
        node_x = [pos.get(node, (0, 0))[0] for node in G.nodes()]
        node_y = [pos.get(node, (0, 0))[1] for node in G.nodes()]
        node_colors = [anomaly_scores_cpu[i] if i < len(anomaly_scores_cpu) else 0 for i in G.nodes()]
        
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos.get(edge[0], (0, 0))
            x1, y1 = pos.get(edge[1], (0, 0))
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create comprehensive visualization
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('🕸️ Attack Flow Network', '🕐 Temporal Heatmap', '🧠 Pattern Classification',
                           '⚡ Velocity Timeline', '📊 Anomaly Distribution', '🔗 Agent Interaction Matrix'),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}, {"type": "heatmap"}]]
        )
        
        # Attack flow network
        if edge_x:
            fig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=2, color='rgba(125,125,125,0.5)'),
                    hoverinfo='none',
                    showlegend=False,
                    name='Attack Connections'
                ),
                row=1, col=1
            )
        
        if node_x:
            fig.add_trace(
                go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=node_colors,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Anomaly Score", x=0.3)
                    ),
                    text=[f"A{i}" for i in G.nodes()],
                    textposition="middle center",
                    hovertemplate='Agent %{text}<br>Anomaly: %{marker.color:.2f}<extra></extra>',
                    showlegend=False,
                    name='AI Agents'
                ),
                row=1, col=1
            )
        
        # Temporal patterns heatmap
        temporal_data = processed_data['temporal_patterns'][:50].cpu().numpy()
        if temporal_data.size > 0:
            fig.add_trace(
                go.Heatmap(
                    z=temporal_data,
                    colorscale='YlOrRd',
                    showscale=False,
                    hovertemplate='Hour: %{x}<br>Agent: %{y}<br>Activity: %{z:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Pattern classification bar chart
        pattern_labels = ['Reconnaissance', 'Initial Access', 'Execution', 'Persistence', 
                         'Privilege Escalation', 'Defense Evasion', 'Credential Access',
                         'Discovery', 'Lateral Movement', 'Exfiltration']
        
        fig.add_trace(
            go.Bar(
                x=pattern_labels,
                y=pattern_predictions_cpu.flatten()[:10],
                showlegend=False,
                marker_color='lightblue',
                hovertemplate='Pattern: %{x}<br>Probability: %{y:.2%}<extra></extra>'
            ),
            row=1, col=3
        )
        
        # Velocity timeline
        if len(velocities_cpu) > 0:
            fig.add_trace(
                go.Scatter(
                    y=velocities_cpu[:100],
                    mode='lines+markers',
                    name='Attack Velocity',
                    showlegend=False,
                    line=dict(color='orange'),
                    hovertemplate='Time: %{x}<br>Velocity: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Anomaly distribution
        if len(anomaly_scores_cpu) > 0:
            fig.add_trace(
                go.Histogram(
                    x=anomaly_scores_cpu,
                    nbinsx=30,
                    showlegend=False,
                    marker_color='red',
                    opacity=0.7,
                    hovertemplate='Score: %{x:.2f}<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Agent interaction matrix
        fig.add_trace(
            go.Heatmap(
                z=interactions_cpu[:20, :20],
                colorscale='Blues',
                showscale=False,
                hovertemplate='From Agent: %{y}<br>To Agent: %{x}<br>Interactions: %{z}<extra></extra>'
            ),
            row=2, col=3
        )
        
        device_info = "Apple MPS-Accelerated" if self.use_mps else "CPU Mode"
        fig.update_layout(
            title=f"🤖 Real-Time Agentic AI Attack Flow Analysis ({device_info})",
            height=800,
            showlegend=False,
            template='plotly_dark',
            font=dict(size=12)
        )
        
        # Update axes
        fig.update_xaxes(title_text="X Position", row=1, col=1)
        fig.update_yaxes(title_text="Y Position", row=1, col=1)
        fig.update_xaxes(title_text="Hour", row=1, col=2)
        fig.update_yaxes(title_text="Agent", row=1, col=2)
        fig.update_xaxes(title_text="Attack Pattern", row=1, col=3)
        fig.update_yaxes(title_text="Probability", row=1, col=3)
        fig.update_xaxes(title_text="Time Step", row=2, col=1)
        fig.update_yaxes(title_text="Velocity", row=2, col=1)
        fig.update_xaxes(title_text="Anomaly Score", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_xaxes(title_text="Target Agent", row=2, col=3)
        fig.update_yaxes(title_text="Source Agent", row=2, col=3)
        
        return fig
    
    def predict_attack_progression_mps(self, current_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Predict next attack steps using MPS-accelerated neural networks
        
        Args:
            current_state: Current attack state tensors
            
        Returns:
            Dictionary of predicted attack probabilities
        """
        with torch.no_grad():
            # Extract recent features
            recent_velocities = current_state['attack_velocity'][-10:] if len(current_state['attack_velocity']) > 10 else current_state['attack_velocity']
            recent_anomalies = current_state['anomaly_scores'][-10:] if len(current_state['anomaly_scores']) > 10 else current_state['anomaly_scores']
            
            # Handle empty tensors
            if len(recent_velocities) == 0:
                recent_velocities = torch.tensor([0.5], device=self.device)
            if len(recent_anomalies) == 0:
                recent_anomalies = torch.tensor([0.1], device=self.device)
            
            # Pad if necessary
            if len(recent_velocities) < 10:
                padding = torch.zeros(10 - len(recent_velocities), device=self.device)
                recent_velocities = torch.cat([padding, recent_velocities])
            if len(recent_anomalies) < 10:
                padding = torch.zeros(10 - len(recent_anomalies), device=self.device)
                recent_anomalies = torch.cat([padding, recent_anomalies])
            
            # Create feature vector (need to pad to 256 features to match network)
            features = torch.cat([recent_velocities, recent_anomalies]).unsqueeze(0)
            
            # Pad to network input size (256)
            if features.shape[1] < 256:
                padding_size = 256 - features.shape[1]
                features = torch.nn.functional.pad(features, (0, padding_size))
            elif features.shape[1] > 256:
                features = features[:, :256]
            
            # Predict using pattern network
            predictions = torch.nn.functional.softmax(
                self.pattern_network(features), dim=1
            )
            
            attack_types = ['lateral_movement', 'persistence', 'exfiltration', 
                           'c2_communication', 'privilege_escalation', 'defense_evasion',
                           'credential_access', 'discovery', 'collection', 'impact']
            
            return {attack_type: float(pred) for attack_type, pred in 
                   zip(attack_types, predictions.cpu().numpy().flatten())}
    
    def optimize_memory_usage(self):
        """Optimize memory usage for Apple Silicon unified memory"""
        if self.use_mps:
            # Clear MPS cache
            torch.mps.empty_cache()
            
            # Synchronize MPS operations
            torch.mps.synchronize()
            
            print("🍎 MPS memory optimized")

def create_sample_attack_events(num_events: int = 50) -> List[AgentAttackEvent]:
    """Create sample attack events for testing"""
    events = []
    base_time = datetime.now()
    
    agent_ids = ['agent_alpha', 'agent_beta', 'agent_gamma', 'agent_delta', 'agent_epsilon']
    attack_types = ['reconnaissance', 'initial_access', 'execution', 'persistence', 
                   'privilege_escalation', 'defense_evasion', 'credential_access']
    target_systems = ['web_server', 'database', 'file_server', 'domain_controller', 
                     'workstation', 'mobile_device', 'iot_device']
    
    for i in range(num_events):
        event = AgentAttackEvent(
            agent_id=np.random.choice(agent_ids),
            timestamp=base_time + timedelta(minutes=np.random.randint(0, 180)),
            attack_type=np.random.choice(attack_types),
            target_system=np.random.choice(target_systems),
            confidence_score=np.random.uniform(0.1, 1.0),
            lateral_movement=np.random.choice([True, False]),
            persistence_indicators=['registry_key', 'scheduled_task', 'service', 'startup_folder'],
            ttl_seconds=np.random.randint(60, 7200)
        )
        events.append(event)
    
    return events

def main():
    """Test the MPS agentic flow visualizer"""
    print("🍎 Testing Apple MPS Agentic Flow Visualizer")
    
    # Initialize visualizer
    visualizer = MPSAgenticFlowVisualizer()
    
    # Create sample events
    events = create_sample_attack_events(150)
    print(f"📊 Created {len(events)} sample attack events")
    
    # Process events
    processed_data = visualizer.process_attack_stream_mps(events)
    print(f"✅ Processed attack stream with {len(processed_data)} result tensors")
    
    # Generate visualization
    fig = visualizer.generate_attack_flow_graph_mps(processed_data)
    
    # Save visualization
    output_file = "agentic_flow_analysis_mps.html"
    fig.write_html(output_file)
    print(f"💾 Visualization saved to {output_file}")
    
    # Predict attack progression
    predictions = visualizer.predict_attack_progression_mps(processed_data)
    print(f"🔮 Attack progression predictions:")
    for attack_type, probability in predictions.items():
        print(f"   {attack_type}: {probability:.2%}")
    
    # Optimize memory
    visualizer.optimize_memory_usage()
    print(f"🎯 Memory optimization completed")

if __name__ == "__main__":
    main() 