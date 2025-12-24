"""
ConcreTizer: Model Inversion Attack for 3D/Voxel Reconstruction
Based on research in model inversion and membership inference (2023-2025)

Key Papers:
- "Model Inversion Attacks that Exploit Confidence Information" (Fredrikson et al. 2015)
- "The Secret Sharer: Evaluating and Testing Unintended Memorization" (Carlini et al. 2019)
- "Plug & Play Attacks: Towards Robust and Flexible Model Inversion Attacks" (Struppek et al. 2022)
- 3D reconstruction from neural network activations

Attack Methodology:
1. Systematic Query Strategy: Grid-based or adaptive sampling
2. Voxel Occupancy Inference: Binary classification per voxel
3. Iterative Refinement: Confidence-based boundary sharpening
4. Information Leakage Maximization: Query optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)


@dataclass
class ConcreTizerConfig:
    """Configuration for ConcreTizer attack"""
    voxel_resolution: int = 32  # Grid resolution (32x32x32)
    max_queries: int = 1000  # Maximum number of model queries
    query_strategy: str = 'adaptive'  # 'grid', 'adaptive', 'confidence'
    confidence_threshold: float = 0.7  # Threshold for voxel occupancy
    refinement_iterations: int = 3  # Number of boundary refinement passes
    batch_size: int = 64  # Queries per batch
    temperature: float = 1.0  # Softmax temperature for confidence
    privacy_budget: Optional[int] = None  # Optional hard query budget constraint
    device: str = 'cpu'
    verbose: bool = True


@dataclass
class ConcreTizerResult:
    """Results from ConcreTizer attack"""
    reconstructed_voxels: np.ndarray  # [resolution, resolution, resolution]
    voxel_confidences: np.ndarray  # Confidence scores per voxel
    num_queries_used: int
    reconstruction_quality: float  # Estimated quality (0-1)
    information_leakage_score: float  # Amount of private info extracted
    query_efficiency: float  # Quality per query
    metadata: Dict[str, Any]


class ConcreTizerAttack:
    """
    ConcreTizer: Model Inversion Attack for Voxel/3D Reconstruction
    
    Systematically queries a target model to reconstruct private training data
    by exploiting prediction confidence and activation patterns.
    
    Attack assumes:
    - Black-box or gray-box access (query + confidence scores)
    - Target model trained on spatial/voxel data
    - Confidence information available from model outputs
    """
    
    def __init__(self, 
                 target_model: Callable[[torch.Tensor], torch.Tensor],
                 config: ConcreTizerConfig):
        """
        Args:
            target_model: Function that takes input and returns logits or probabilities
            config: Attack configuration
        """
        self.target_model = target_model
        self.config = config
        self.device = torch.device(config.device)
        self.query_count = 0
        self.query_history = []
        
    def _initialize_voxel_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize voxel grid and confidence matrix
        
        Returns:
            voxels: [resolution, resolution, resolution] - binary occupancy
            confidences: [resolution, resolution, resolution] - confidence scores
        """
        res = self.config.voxel_resolution
        voxels = torch.zeros(res, res, res, dtype=torch.float32, device=self.device)
        confidences = torch.zeros(res, res, res, dtype=torch.float32, device=self.device)
        return voxels, confidences
    
    def _generate_grid_queries(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate systematic grid-based queries
        Probes each voxel in sequence
        """
        res = self.config.voxel_resolution
        total_voxels = res ** 3

        remaining = max(0, int(self.config.max_queries) - int(self.query_count))
        if remaining <= 0:
            return []
        max_voxels = min(total_voxels, remaining)

        batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, max_voxels, self.config.batch_size):
            end = min(start + self.config.batch_size, max_voxels)
            batch_indices: List[List[int]] = []
            batch_queries: List[torch.Tensor] = []

            for flat_idx in range(start, end):
                x = flat_idx // (res ** 2)
                y = (flat_idx % (res ** 2)) // res
                z = flat_idx % res
                batch_indices.append([x, y, z])
                batch_queries.append(self._encode_voxel_query(x, y, z, res))

            if batch_queries:
                idx_tensor = torch.tensor(batch_indices, device=self.device, dtype=torch.long)
                batches.append((idx_tensor, torch.stack(batch_queries)))

        return batches
    
    def _generate_adaptive_queries(self, 
                                   current_voxels: torch.Tensor,
                                   current_confidences: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate adaptive queries focusing on uncertain regions
        Targets boundaries and low-confidence areas
        """
        res = self.config.voxel_resolution
        
        # Find uncertain voxels (confidence near 0.5)
        uncertainty = torch.abs(current_confidences - 0.5)
        uncertain_mask = uncertainty < 0.3  # Focus on uncertain regions
        
        # Also target boundaries (high gradient in voxel space)
        dx = torch.diff(current_voxels, dim=0, prepend=current_voxels[:1])
        dy = torch.diff(current_voxels, dim=1, prepend=current_voxels[:, :1])
        dz = torch.diff(current_voxels, dim=2, prepend=current_voxels[:, :, :1])
        gradient_magnitude = torch.sqrt(dx**2 + dy**2 + dz**2)
        boundary_mask = gradient_magnitude > 0.1
        
        # Combine masks
        priority_mask = uncertain_mask | boundary_mask
        priority_indices = torch.nonzero(priority_mask, as_tuple=False)
        
        remaining = max(0, int(self.config.max_queries) - int(self.query_count))
        if remaining <= 0:
            return []

        # Generate queries for priority regions
        queries: List[Tuple[torch.Tensor, torch.Tensor]] = []
        num_priority = min(int(priority_indices.shape[0]), remaining)
        
        for i in range(0, num_priority, self.config.batch_size):
            batch_indices = priority_indices[i:i+self.config.batch_size]
            batch_queries = []
            
            for idx in batch_indices:
                x, y, z = idx[0].item(), idx[1].item(), idx[2].item()
                # Create query encoding voxel neighborhood
                query = self._encode_voxel_query(x, y, z, res)
                batch_queries.append(query)
            
            if batch_queries:
                queries.append((batch_indices.to(self.device, dtype=torch.long), torch.stack(batch_queries)))
        
        return queries
    
    def _encode_voxel_query(self, x: int, y: int, z: int, res: int) -> torch.Tensor:
        """
        Encode a voxel query as input to target model
        Uses 3D coordinate encoding
        """
        # Normalized coordinates
        coords = torch.tensor([x / res, y / res, z / res], device=self.device)
        
        # Add positional encoding (sinusoidal)
        freqs = torch.tensor([1, 2, 4, 8, 16], device=self.device)
        encoded = []
        for freq in freqs:
            encoded.append(torch.sin(2 * np.pi * freq * coords))
            encoded.append(torch.cos(2 * np.pi * freq * coords))
        
        query = torch.cat([coords] + encoded, dim=0)
        return query
    
    def _query_model(self, queries: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Query target model and extract confidence scores
        
        Args:
            queries: [batch_size, feature_dim]
        
        Returns:
            predictions: [batch_size] - predicted classes
            confidences: [batch_size] - confidence scores (max probability)
        """
        self.query_count += len(queries)
        
        # Check privacy budget
        if self.config.privacy_budget is not None:
            if int(self.query_count) > int(self.config.privacy_budget):
                logger.warning(f"Query budget exceeded: {self.query_count}/{self.config.privacy_budget}")
                return None, None
        
        # Query model
        with torch.no_grad():
            outputs = self.target_model(queries)
            
            # Apply temperature scaling
            if self.config.temperature != 1.0:
                outputs = outputs / self.config.temperature
            
            # Get probabilities
            probs = F.softmax(outputs, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
        
        # Store query history
        self.query_history.append({
            'queries': queries.cpu().numpy(),
            'predictions': predictions.cpu().numpy(),
            'confidences': confidences.cpu().numpy()
        })
        
        return predictions, confidences
    
    def _update_voxel_grid(self,
                          voxels: torch.Tensor,
                          confidences: torch.Tensor,
                          query_results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        """
        Update voxel grid based on query results
        
        Args:
            voxels: Current voxel occupancy grid
            confidences: Current confidence scores
            query_results: List of (indices, predictions, confidence_scores)
        """
        for indices, predictions, conf_scores in query_results:
            for i, (idx, pred, conf) in enumerate(zip(indices, predictions, conf_scores)):
                x, y, z = idx[0].item(), idx[1].item(), idx[2].item()
                
                # Update voxel based on confidence
                if conf > self.config.confidence_threshold:
                    voxels[x, y, z] = 1.0  # Occupied
                else:
                    voxels[x, y, z] = 0.0  # Empty
                
                # Store confidence
                confidences[x, y, z] = conf
    
    def _refine_boundaries(self,
                          voxels: torch.Tensor,
                          confidences: torch.Tensor) -> None:
        """
        Refine voxel boundaries using morphological operations
        and confidence-based smoothing
        """
        # Find boundary voxels
        kernel = torch.ones(3, 3, 3, device=self.device)
        kernel[1, 1, 1] = 0  # Exclude center
        
        # Convolve to find neighbors
        voxels_expanded = voxels.unsqueeze(0).unsqueeze(0)
        neighbor_count = F.conv3d(
            voxels_expanded,
            kernel.unsqueeze(0).unsqueeze(0),
            padding=1
        ).squeeze()
        
        # Refine boundaries: if majority of neighbors agree, update voxel
        boundary_mask = (neighbor_count > 13) & (voxels == 0)  # More than half neighbors occupied
        voxels[boundary_mask] = 1.0
        
        boundary_mask = (neighbor_count < 13) & (voxels == 1)  # Less than half neighbors occupied
        voxels[boundary_mask] = 0.0
        
        # Smooth low-confidence regions
        low_conf_mask = confidences < 0.6
        smoothed = F.avg_pool3d(
            voxels.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1
        ).squeeze()
        
        voxels[low_conf_mask] = (smoothed[low_conf_mask] > 0.5).float()
    
    def _compute_reconstruction_quality(self,
                                       voxels: torch.Tensor,
                                       confidences: torch.Tensor) -> float:
        """
        Estimate reconstruction quality based on confidence distribution
        and voxel coherence
        """
        # High-confidence voxel ratio
        high_conf_ratio = (confidences > self.config.confidence_threshold).float().mean()
        
        # Voxel coherence (smoothness)
        dx = torch.diff(voxels, dim=0)
        dy = torch.diff(voxels, dim=1)
        dz = torch.diff(voxels, dim=2)
        smoothness = 1.0 - (torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)) / 3.0
        
        # Combined quality score
        quality = 0.6 * high_conf_ratio + 0.4 * smoothness
        return float(quality.item())
    
    def _compute_information_leakage(self) -> float:
        """
        Compute information leakage score based on query history
        Measures how much private information was extracted
        """
        if not self.query_history:
            return 0.0
        
        # Aggregate all confidences
        all_confidences = np.concatenate([h['confidences'] for h in self.query_history])
        
        # High confidence means more information leaked
        # Use entropy as inverse measure of information gain
        mean_confidence = np.mean(all_confidences)
        confidence_std = np.std(all_confidences)
        
        # Information leakage: high mean confidence + low std = more leakage
        leakage_score = mean_confidence * (1.0 - confidence_std / (mean_confidence + 1e-6))
        return float(np.clip(leakage_score, 0, 1))
    
    def attack(self, 
               initial_seed: Optional[torch.Tensor] = None) -> ConcreTizerResult:
        """
        Execute ConcreTizer model inversion attack
        
        Args:
            initial_seed: Optional seed data to guide reconstruction
        
        Returns:
            ConcreTizerResult with reconstructed voxels and metrics
        """
        logger.info("🔴 Starting ConcreTizer model inversion attack...")
        logger.info(f"   Voxel resolution: {self.config.voxel_resolution}³")
        logger.info(f"   Max queries: {self.config.max_queries}")
        logger.info(f"   Query strategy: {self.config.query_strategy}")
        
        # Initialize voxel grid
        voxels, confidences = self._initialize_voxel_grid()
        
        # Main attack loop
        for iteration in range(self.config.refinement_iterations):
            logger.info(f"\n📊 Iteration {iteration + 1}/{self.config.refinement_iterations}")
            
            # Generate queries based on strategy
            if iteration == 0 or self.config.query_strategy == 'grid':
                queries = self._generate_grid_queries()
            else:
                queries = self._generate_adaptive_queries(voxels, confidences)
            
            if not queries:
                logger.warning("   No more queries to generate")
                break
            
            # Execute queries
            query_results = []
            for batch_idx, (indices, query_batch) in enumerate(queries):
                if self.query_count >= self.config.max_queries:
                    logger.info(f"   Reached query limit: {self.config.max_queries}")
                    break
                
                predictions, conf_scores = self._query_model(query_batch)
                
                if predictions is None:
                    break
                
                query_results.append((indices, predictions, conf_scores))
                
                if self.config.verbose and batch_idx % 10 == 0:
                    logger.info(f"   Queries: {self.query_count}/{self.config.max_queries}")
            
            # Update voxel grid
            self._update_voxel_grid(voxels, confidences, query_results)
            
            # Refine boundaries
            if iteration < self.config.refinement_iterations - 1:
                self._refine_boundaries(voxels, confidences)
            
            # Compute current quality
            quality = self._compute_reconstruction_quality(voxels, confidences)
            logger.info(f"   Reconstruction quality: {quality:.3f}")
        
        # Compute final metrics
        final_quality = self._compute_reconstruction_quality(voxels, confidences)
        information_leakage = self._compute_information_leakage()
        query_efficiency = final_quality / (self.query_count + 1e-6)
        
        result = ConcreTizerResult(
            reconstructed_voxels=voxels.cpu().numpy(),
            voxel_confidences=confidences.cpu().numpy(),
            num_queries_used=self.query_count,
            reconstruction_quality=final_quality,
            information_leakage_score=information_leakage,
            query_efficiency=query_efficiency,
            metadata={
                'config': self.config.__dict__,
                'query_strategy': self.config.query_strategy,
                'total_voxels': self.config.voxel_resolution ** 3,
                'occupied_voxels': int(voxels.sum().item())
            }
        )
        
        logger.info("\n✅ Attack complete:")
        logger.info(f"   Queries used: {self.query_count}")
        logger.info(f"   Reconstruction quality: {final_quality:.3f}")
        logger.info(f"   Information leakage: {information_leakage:.3f}")
        logger.info(f"   Query efficiency: {query_efficiency:.6f}")
        logger.info(f"   Occupied voxels: {int(voxels.sum().item())}/{self.config.voxel_resolution**3}")
        
        return result


def create_simple_voxel_classifier(input_dim: int = 33,  # 3 coords + 30 encoding
                                   num_classes: int = 10,
                                   hidden_dim: int = 256) -> nn.Module:
    """
    Create a simple voxel classifier for testing ConcreTizer attack
    """
    class SimpleVoxelClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, num_classes)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            logits = self.fc3(x)
            return logits
    
    return SimpleVoxelClassifier()

