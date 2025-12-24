#!/usr/bin/env python3
"""
CUDA-Accelerated Agentic AI Attack Flow Visualizer
Real-time visualization of autonomous agent attack patterns using NVIDIA GPU acceleration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

try:
    import cupy as cp
    import cudf
    import cuml
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    import numpy as cp  # Fallback to numpy
    
import torch

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

class CUDAAgenticFlowVisualizer:
    """Real-time agentic AI attack flow visualization with NVIDIA CUDA acceleration"""
    
    def __init__(self, device_id: int = 0, memory_pool_size: int = 2**30):
        """Initialize CUDA-accelerated agentic flow visualizer
        
        Args:
            device_id: CUDA device ID
            memory_pool_size: GPU memory pool size in bytes
        """
        if not CUDA_AVAILABLE:
            print("⚠️ CUDA libraries not available. Falling back to CPU mode.")
            self.use_cuda = False
            self.device = torch.device("cpu")
        else:
            try:
                self.device = cp.cuda.Device(device_id)
                self.memory_pool = cp.get_default_memory_pool()
                self.memory_pool.set_limit(size=memory_pool_size)
                
                # Initialize CUDA streams for parallel processing
                self.computation_stream = cp.cuda.Stream(non_blocking=True)
                self.visualization_stream = cp.cuda.Stream(non_blocking=True)
                self.use_cuda = True
                print(f"✅ CUDA device {device_id} initialized for agentic flow visualization")
            except Exception as e:
                print(f"⚠️ CUDA initialization failed: {e}. Falling back to CPU mode.")
                self.use_cuda = False
                self.device = torch.device("cpu")
        
        # Agent behavior tracking matrices
        if self.use_cuda:
            self.agent_interactions = cp.zeros((1000, 1000), dtype=cp.float32)
            self.temporal_patterns = cp.zeros((1000, 24), dtype=cp.float32)
            self.attack_velocity = cp.zeros((1000, 10), dtype=cp.float32)
        else:
            self.agent_interactions = np.zeros((1000, 1000), dtype=np.float32)
            self.temporal_patterns = np.zeros((1000, 24), dtype=np.float32)
            self.attack_velocity = np.zeros((1000, 10), dtype=np.float32)
        
        # Real-time event buffer
        self.event_buffer = []
        self.max_buffer_size = 10000
        
        # Initialize cuML for clustering and anomaly detection (CUDA only)
        if self.use_cuda and CUDA_AVAILABLE:
            try:
                with self.device:
                    self.dbscan = cuml.DBSCAN(eps=0.5, min_samples=5)
                    self.isolation_forest = cuml.IsolationForest(contamination=0.1)
                print("✅ cuML models initialized")
            except Exception as e:
                print(f"⚠️ cuML initialization failed: {e}")
                self.dbscan = None
                self.isolation_forest = None
        else:
            self.dbscan = None
            self.isolation_forest = None
    
    def process_attack_stream(self, events: List[AgentAttackEvent]) -> Dict[str, np.ndarray]:
        """Process streaming attack data with CUDA acceleration
        
        Args:
            events: List of agent attack events
            
        Returns:
            Dictionary containing processed attack matrices
        """
        if not events:
            return self._get_empty_results()
        
        if self.use_cuda:
            return self._process_attack_stream_cuda(events)
        else:
            return self._process_attack_stream_cpu(events)
    
    def _process_attack_stream_cuda(self, events: List[AgentAttackEvent]) -> Dict[str, np.ndarray]:
        """CUDA-accelerated attack stream processing"""
        with self.device, self.computation_stream:
            # Convert events to GPU arrays
            agent_ids = cp.array([hash(event.agent_id) % 1000 for event in events])
            timestamps = cp.array([event.timestamp.timestamp() for event in events])
            attack_types = cp.array([hash(event.attack_type) % 10 for event in events])
            confidence_scores = cp.array([event.confidence_score for event in events])
            
            # Compute temporal attack patterns using CUDA kernels
            time_bins = cp.floor((timestamps - timestamps.min()) / 3600) % 24
            
            # Update interaction matrix using advanced indexing
            if len(agent_ids) > 1:
                cp.add.at(self.agent_interactions, (agent_ids[:-1], agent_ids[1:]), 1.0)
            
            # Calculate attack velocity using sliding window
            velocities = self._calculate_velocity_cuda(timestamps, confidence_scores)
            
            # Detect anomalous attack patterns
            feature_matrix = cp.column_stack([agent_ids, attack_types, confidence_scores, velocities])
            
            # Use cuML if available, otherwise fallback
            if self.isolation_forest is not None:
                try:
                    anomaly_scores = self.isolation_forest.decision_function(feature_matrix)
                    cluster_labels = self.dbscan.fit_predict(feature_matrix)
                except Exception as e:
                    print(f"⚠️ cuML processing failed: {e}")
                    anomaly_scores = cp.abs(confidence_scores - cp.mean(confidence_scores))
                    cluster_labels = cp.zeros_like(agent_ids)
            else:
                anomaly_scores = cp.abs(confidence_scores - cp.mean(confidence_scores))
                cluster_labels = cp.zeros_like(agent_ids)
            
            return {
                'agent_interactions': cp.asnumpy(self.agent_interactions),
                'temporal_patterns': cp.asnumpy(self.temporal_patterns),
                'attack_velocity': cp.asnumpy(velocities),
                'anomaly_scores': cp.asnumpy(anomaly_scores),
                'cluster_labels': cp.asnumpy(cluster_labels),
                'feature_matrix': cp.asnumpy(feature_matrix)
            }
    
    def _process_attack_stream_cpu(self, events: List[AgentAttackEvent]) -> Dict[str, np.ndarray]:
        """CPU fallback for attack stream processing"""
        agent_ids = np.array([hash(event.agent_id) % 1000 for event in events])
        timestamps = np.array([event.timestamp.timestamp() for event in events])
        attack_types = np.array([hash(event.attack_type) % 10 for event in events])
        confidence_scores = np.array([event.confidence_score for event in events])
        
        # Update interaction matrix
        if len(agent_ids) > 1:
            for i in range(len(agent_ids) - 1):
                self.agent_interactions[agent_ids[i], agent_ids[i+1]] += 1.0
        
        # Calculate attack velocity
        velocities = self._calculate_velocity_cpu(timestamps, confidence_scores)
        
        # Simple anomaly detection
        anomaly_scores = np.abs(confidence_scores - np.mean(confidence_scores))
        cluster_labels = np.zeros_like(agent_ids)
        
        feature_matrix = np.column_stack([agent_ids, attack_types, confidence_scores, velocities])
        
        return {
            'agent_interactions': self.agent_interactions,
            'temporal_patterns': self.temporal_patterns,
            'attack_velocity': velocities,
            'anomaly_scores': anomaly_scores,
            'cluster_labels': cluster_labels,
            'feature_matrix': feature_matrix
        }
    
    def _calculate_velocity_cuda(self, timestamps, confidence_scores):
        """CUDA-optimized velocity calculation"""
        window_size = 300  # 5-minute window
        velocities = cp.zeros_like(confidence_scores)
        
        for i in range(len(timestamps)):
            # Find points within window
            window_mask = cp.abs(timestamps - timestamps[i]) <= window_size
            if cp.sum(window_mask) > 0:
                velocities[i] = cp.mean(confidence_scores[window_mask])
        
        return velocities
    
    def _calculate_velocity_cpu(self, timestamps, confidence_scores):
        """CPU fallback velocity calculation"""
        window_size = 300  # 5-minute window
        velocities = np.zeros_like(confidence_scores)
        
        for i in range(len(timestamps)):
            window_mask = np.abs(timestamps - timestamps[i]) <= window_size
            if np.sum(window_mask) > 0:
                velocities[i] = np.mean(confidence_scores[window_mask])
        
        return velocities
    
    def _get_empty_results(self) -> Dict[str, np.ndarray]:
        """Return empty results structure"""
        return {
            'agent_interactions': np.zeros((10, 10)),
            'temporal_patterns': np.zeros((10, 24)),
            'attack_velocity': np.array([]),
            'anomaly_scores': np.array([]),
            'cluster_labels': np.array([]),
            'feature_matrix': np.zeros((1, 4))
        }
    
    def generate_attack_flow_graph(self, processed_data: Dict[str, np.ndarray]) -> go.Figure:
        """Generate interactive attack flow visualization
        
        Args:
            processed_data: Processed attack data from CUDA operations
            
        Returns:
            Plotly figure with interactive attack flow graph
        """
        # Transfer data to CPU for visualization (already done in processing)
        interactions_cpu = processed_data['agent_interactions']
        velocities_cpu = processed_data['attack_velocity']
        anomaly_scores_cpu = processed_data['anomaly_scores']
        
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
        
        # Extract node and edge information
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_colors = [anomaly_scores_cpu[i] if i < len(anomaly_scores_cpu) else 0 for i in G.nodes()]
        
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = G[edge[0]][edge[1]].get('weight', 1)
            edge_weights.append(weight)
        
        # Create interactive plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('🕸️ Attack Flow Network', '🕐 Temporal Patterns', 
                           '⚡ Velocity Analysis', '📊 Anomaly Distribution'),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "scatter3d"}, {"type": "histogram"}]]
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
                        colorbar=dict(title="Anomaly Score", x=0.45)
                    ),
                    text=[f"Agent {i}" for i in G.nodes()],
                    textposition="middle center",
                    hovertemplate='Agent %{text}<br>Anomaly: %{marker.color:.2f}<extra></extra>',
                    showlegend=False,
                    name='AI Agents'
                ),
                row=1, col=1
            )
        
        # Temporal patterns heatmap
        temporal_data = processed_data['temporal_patterns'][:50]
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
        
        # 3D velocity analysis
        if len(velocities_cpu) > 2:
            fig.add_trace(
                go.Scatter3d(
                    x=velocities_cpu[:100],
                    y=anomaly_scores_cpu[:100] if len(anomaly_scores_cpu) > 0 else [0],
                    z=np.arange(len(velocities_cpu[:100])),
                    mode='markers',
                    marker=dict(
                        size=5, 
                        color=anomaly_scores_cpu[:100] if len(anomaly_scores_cpu) > 0 else [0], 
                        colorscale='Plasma'
                    ),
                    showlegend=False,
                    hovertemplate='Velocity: %{x:.2f}<br>Anomaly: %{y:.2f}<br>Time: %{z}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Anomaly distribution histogram
        if len(anomaly_scores_cpu) > 0:
            fig.add_trace(
                go.Histogram(
                    x=anomaly_scores_cpu,
                    nbinsx=30,
                    showlegend=False,
                    marker_color='red',
                    opacity=0.7,
                    hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
        
        device_info = "CUDA-Accelerated" if self.use_cuda else "CPU Mode"
        fig.update_layout(
            title=f"🤖 Real-Time Agentic AI Attack Flow Analysis ({device_info})",
            height=800,
            showlegend=False,
            template='plotly_dark',
            font=dict(size=12)
        )
        
        # Update subplot titles
        fig.update_xaxes(title_text="X Position", row=1, col=1)
        fig.update_yaxes(title_text="Y Position", row=1, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
        fig.update_yaxes(title_text="Agent ID", row=1, col=2)
        fig.update_xaxes(title_text="Velocity", row=2, col=1)
        fig.update_yaxes(title_text="Anomaly Score", row=2, col=1)
        fig.update_xaxes(title_text="Anomaly Score", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        return fig
    
    def predict_attack_progression(self, current_state: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Predict next attack steps using CUDA-accelerated ML models
        
        Args:
            current_state: Current attack state matrices
            
        Returns:
            Dictionary of predicted attack probabilities
        """
        if self.use_cuda:
            return self._predict_attack_progression_cuda(current_state)
        else:
            return self._predict_attack_progression_cpu(current_state)
    
    def _predict_attack_progression_cuda(self, current_state: Dict[str, np.ndarray]) -> Dict[str, float]:
        """CUDA-accelerated attack progression prediction"""
        with self.device, self.computation_stream:
            # Convert to GPU
            velocities = cp.asarray(current_state['attack_velocity'][-100:]) if len(current_state['attack_velocity']) > 0 else cp.array([0.5])
            anomalies = cp.asarray(current_state['anomaly_scores'][-100:]) if len(current_state['anomaly_scores']) > 0 else cp.array([0.1])
            
            # Simple neural network prediction using CuPy
            features = cp.column_stack([velocities[:len(anomalies)], anomalies[:len(velocities)]])
            if features.size == 0:
                features = cp.array([[0.5, 0.1]])
            
            weights = cp.random.randn(features.shape[1], 5) * 0.1
            predictions = cp.exp(cp.dot(features.mean(axis=0), weights))
            predictions = predictions / cp.sum(predictions)  # Softmax
            
            attack_types = ['lateral_movement', 'persistence', 'exfiltration', 'c2_communication', 'privilege_escalation']
            
            return {attack_type: float(pred) for attack_type, pred in zip(attack_types, cp.asnumpy(predictions))}
    
    def _predict_attack_progression_cpu(self, current_state: Dict[str, np.ndarray]) -> Dict[str, float]:
        """CPU fallback attack progression prediction"""
        velocities = current_state['attack_velocity'][-10:] if len(current_state['attack_velocity']) > 0 else np.array([0.5])
        anomalies = current_state['anomaly_scores'][-10:] if len(current_state['anomaly_scores']) > 0 else np.array([0.1])
        
        # Simple prediction based on current state
        base_probs = [0.2, 0.15, 0.25, 0.3, 0.1]  # Base probabilities
        velocity_factor = np.mean(velocities) if len(velocities) > 0 else 0.5
        anomaly_factor = np.mean(anomalies) if len(anomalies) > 0 else 0.1
        
        predictions = np.array(base_probs) * (1 + velocity_factor + anomaly_factor)
        predictions = predictions / np.sum(predictions)  # Normalize
        
        attack_types = ['lateral_movement', 'persistence', 'exfiltration', 'c2_communication', 'privilege_escalation']
        
        return {attack_type: float(pred) for attack_type, pred in zip(attack_types, predictions)}

def create_sample_attack_events(num_events: int = 50) -> List[AgentAttackEvent]:
    """Create sample attack events for testing"""
    events = []
    base_time = datetime.now()
    
    agent_ids = ['agent_001', 'agent_002', 'agent_003', 'agent_004', 'agent_005']
    attack_types = ['reconnaissance', 'initial_access', 'execution', 'persistence', 'privilege_escalation']
    target_systems = ['web_server', 'database', 'file_server', 'domain_controller', 'workstation']
    
    for i in range(num_events):
        event = AgentAttackEvent(
            agent_id=np.random.choice(agent_ids),
            timestamp=base_time + timedelta(minutes=np.random.randint(0, 120)),
            attack_type=np.random.choice(attack_types),
            target_system=np.random.choice(target_systems),
            confidence_score=np.random.uniform(0.1, 1.0),
            lateral_movement=np.random.choice([True, False]),
            persistence_indicators=['registry_key', 'scheduled_task', 'service'],
            ttl_seconds=np.random.randint(60, 3600)
        )
        events.append(event)
    
    return events

def main():
    """Test the CUDA agentic flow visualizer"""
    print("🚀 Testing CUDA Agentic Flow Visualizer")
    
    # Initialize visualizer
    visualizer = CUDAAgenticFlowVisualizer()
    
    # Create sample events
    events = create_sample_attack_events(100)
    print(f"📊 Created {len(events)} sample attack events")
    
    # Process events
    processed_data = visualizer.process_attack_stream(events)
    print(f"✅ Processed attack stream with {len(processed_data)} result matrices")
    
    # Generate visualization
    fig = visualizer.generate_attack_flow_graph(processed_data)
    
    # Save visualization
    output_file = "agentic_flow_analysis_cuda.html"
    fig.write_html(output_file)
    print(f"💾 Visualization saved to {output_file}")
    
    # Predict attack progression
    predictions = visualizer.predict_attack_progression(processed_data)
    print(f"🔮 Attack progression predictions:")
    for attack_type, probability in predictions.items():
        print(f"   {attack_type}: {probability:.2%}")

if __name__ == "__main__":
    main() 