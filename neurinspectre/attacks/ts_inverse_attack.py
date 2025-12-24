"""
TS-Inverse: Gradient Inversion Attack for Time-Series Models
Based on research in federated learning gradient inversion (2023-2025)

Key Papers:
- "Inverting Gradients: How easy is it to break privacy in federated learning?" (Geiping et al. 2020)
- "See through Gradients: Image Batch Recovery via GradInversion" (Yin et al. 2021)
- "R-GAP: Recursive Gradient Attack on Privacy" (Zhu & Blaschko 2021)
- Time-series adaptations from federated forecasting literature

Attack Methodology:
1. Label Inference: Analytical recovery from final layer gradients
2. Temporal Pattern Matching: FFT-based frequency domain reconstruction
3. Iterative Optimization: LBFGS/Adam to match gradient statistics
4. Sequence Constraints: Temporal smoothness and causality
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TSInverseConfig:
    """Configuration for TS-Inverse attack"""
    max_iterations: int = 1000
    learning_rate: float = 0.1
    optimizer: str = 'LBFGS'  # 'LBFGS', 'Adam', 'SGD'
    reconstruction_loss: str = 'cosine'  # 'cosine', 'l2', 'hybrid'
    temporal_reg_weight: float = 0.01  # Temporal smoothness
    frequency_reg_weight: float = 0.001  # Frequency domain matching
    total_variation_weight: float = 0.01  # TV regularization
    init_strategy: str = 'random'  # 'random', 'zeros', 'noise'
    batch_size: int = 1
    sequence_length: int = 100
    num_features: int = 64
    device: str = 'cpu'
    verbose: bool = True


@dataclass
class TSInverseResult:
    """Results from TS-Inverse attack"""
    reconstructed_data: np.ndarray
    reconstruction_loss: float
    iterations_used: int
    success: bool
    label_inference_accuracy: float
    temporal_coherence: float
    frequency_match_score: float
    metadata: Dict[str, Any]


class TSInverseAttack:
    """
    TS-Inverse: Gradient Inversion Attack for Time-Series Federated Learning
    
    Reconstructs private training data from leaked gradients by:
    1. Inferring labels from final layer gradients (analytical)
    2. Reconstructing time-series via gradient matching optimization
    3. Applying temporal and frequency domain constraints
    """
    
    def __init__(self, model: nn.Module, config: TSInverseConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()
        
    def infer_labels(self, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Analytical label inference from final layer gradients
        Based on: Geiping et al. 2020
        
        For classification: softmax(logits) - one_hot(label) = gradient
        Solve for label by finding argmin of gradient magnitude
        """
        # Find final layer gradient (typically fc.weight or output.weight)
        final_layer_keys = [k for k in gradients.keys() if 'fc' in k or 'output' in k or 'classifier' in k]
        
        if not final_layer_keys:
            logger.warning("Could not find final layer gradients; defaulting inferred labels to 0 (deterministic fallback).")
            return torch.zeros((self.config.batch_size,), dtype=torch.long, device=self.device)
        
        final_grad = gradients[final_layer_keys[0]]
        
        # For each sample in batch, infer label from gradient pattern
        if final_grad.dim() == 2:  # [output_dim, input_dim]
            # Gradient at class i is proportional to -(one_hot[i] - softmax[i])
            # The true label has largest negative gradient
            grad_norms = torch.norm(final_grad, dim=1)
            label = int(torch.argmax(grad_norms).item())
            inferred_labels = torch.full((self.config.batch_size,), label, dtype=torch.long, device=self.device)
        else:
            # Fallback for other gradient shapes
            inferred_labels = torch.zeros(self.config.batch_size, dtype=torch.long, device=self.device)
        
        return inferred_labels
    
    def compute_temporal_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Temporal smoothness loss: penalize high-frequency changes
        Encourages realistic time-series patterns
        """
        # First-order differences
        diff1 = torch.diff(x, dim=-1)
        # Second-order differences (acceleration)
        diff2 = torch.diff(diff1, dim=-1)
        
        # L2 norm of differences
        temporal_loss = torch.mean(diff1 ** 2) + 0.5 * torch.mean(diff2 ** 2)
        return temporal_loss
    
    def compute_frequency_loss(self, x: torch.Tensor, target_fft: torch.Tensor) -> torch.Tensor:
        """
        Frequency domain matching loss
        Matches power spectral density of reconstruction to gradient statistics
        """
        # Compute FFT of reconstruction
        x_fft = torch.fft.rfft(x, dim=-1)
        x_psd = torch.abs(x_fft) ** 2

        # Match *shape* of the power spectral density (normalize to distributions).
        x_psd = x_psd / (torch.sum(x_psd, dim=-1, keepdim=True) + 1e-10)
        target_sum = torch.sum(target_fft)
        target = target_fft / (target_sum + 1e-10)
        # Expand target to match x_psd shape to avoid implicit broadcasting warnings.
        target = target.reshape(1, 1, -1)
        if target.shape[-1] != x_psd.shape[-1]:
            # Best-effort truncate/pad to match.
            n = int(x_psd.shape[-1])
            if target.shape[-1] > n:
                target = target[..., :n]
            else:
                pad = torch.zeros((1, 1, n - int(target.shape[-1])), device=target.device, dtype=target.dtype)
                target = torch.cat([target, pad], dim=-1)
        target = target.expand_as(x_psd)
        frequency_loss = F.mse_loss(x_psd, target)
        return frequency_loss
    
    def compute_gradient_loss(self, 
                            dummy_grads: Dict[str, torch.Tensor],
                            target_grads: Dict[str, torch.Tensor],
                            loss_type: str = 'cosine') -> torch.Tensor:
        """
        Gradient matching loss: minimize distance between dummy and target gradients
        
        loss_type:
            - 'cosine': 1 - cosine_similarity (most common)
            - 'l2': mean squared error
            - 'hybrid': combination of both
        """
        total_loss = torch.tensor(0.0, device=self.device)
        n_layers = 0
        
        for key in target_grads.keys():
            if key not in dummy_grads:
                continue
            
            tg = target_grads[key].flatten()
            dg = dummy_grads[key].flatten()
            
            if loss_type == 'cosine':
                # Cosine similarity loss (most effective for gradient matching)
                cos_sim = F.cosine_similarity(tg.unsqueeze(0), dg.unsqueeze(0))
                layer_loss = (1.0 - cos_sim).mean()
            elif loss_type == 'l2':
                # L2 distance
                layer_loss = F.mse_loss(dg, tg)
            elif loss_type == 'hybrid':
                # Combination
                cos_sim = F.cosine_similarity(tg.unsqueeze(0), dg.unsqueeze(0))
                l2_loss = F.mse_loss(dg, tg)
                layer_loss = 0.5 * (1.0 - cos_sim).mean() + 0.5 * l2_loss
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            total_loss += layer_loss
            n_layers += 1
        
        return total_loss / max(n_layers, 1)
    
    def initialize_dummy_data(self) -> torch.Tensor:
        """Initialize dummy data for optimization"""
        if self.config.init_strategy == 'random':
            # Random normal initialization
            dummy_data = torch.randn(
                self.config.batch_size,
                self.config.sequence_length,
                self.config.num_features,
                device=self.device,
                requires_grad=True
            )
        elif self.config.init_strategy == 'zeros':
            dummy_data = torch.zeros(
                self.config.batch_size,
                self.config.sequence_length,
                self.config.num_features,
                device=self.device,
                requires_grad=True
            )
        elif self.config.init_strategy == 'noise':
            # Small noise around zero
            dummy_data = 0.01 * torch.randn(
                self.config.batch_size,
                self.config.sequence_length,
                self.config.num_features,
                device=self.device,
                requires_grad=True
            )
        else:
            raise ValueError(f"Unknown init strategy: {self.config.init_strategy}")
        
        return dummy_data
    
    def compute_dummy_gradients(self, 
                               dummy_data: torch.Tensor,
                               dummy_labels: torch.Tensor,
                               criterion: nn.Module) -> Dict[str, torch.Tensor]:
        """Compute gradients of dummy data through model"""
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(dummy_data)
        loss = criterion(outputs, dummy_labels)
        
        # Backward pass
        dummy_grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        
        # Convert to dictionary
        dummy_grad_dict = {}
        for (name, param), grad in zip(self.model.named_parameters(), dummy_grads):
            if grad is not None:
                dummy_grad_dict[name] = grad
        
        return dummy_grad_dict
    
    def attack(self, 
               leaked_gradients: Dict[str, torch.Tensor],
               ground_truth_labels: Optional[torch.Tensor] = None) -> TSInverseResult:
        """
        Execute TS-Inverse gradient inversion attack
        
        Args:
            leaked_gradients: Dictionary of layer_name -> gradient tensor
            ground_truth_labels: Optional true labels (for evaluation)
        
        Returns:
            TSInverseResult with reconstructed data and metrics
        """
        logger.info("🔴 Starting TS-Inverse gradient inversion attack...")

        # Robustness: sanitize non-finite values in leaked gradients so spectral stats and losses don't explode.
        safe_grads: Dict[str, torch.Tensor] = {}
        for k, g in leaked_gradients.items():
            if not isinstance(g, torch.Tensor):
                g = torch.as_tensor(g)
            g = g.to(self.device)
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            safe_grads[k] = g
        leaked_gradients = safe_grads
        
        # Step 1: Infer labels from gradients
        inferred_labels = self.infer_labels(leaked_gradients)
        logger.info(f"   Inferred labels: {inferred_labels.cpu().numpy()}")
        
        # Compute label inference accuracy if ground truth available
        label_accuracy = 0.0
        if ground_truth_labels is not None:
            correct = (inferred_labels == ground_truth_labels).sum().item()
            label_accuracy = correct / len(ground_truth_labels)
            logger.info(f"   Label inference accuracy: {label_accuracy:.2%}")
        
        # Step 2: Initialize dummy data
        dummy_data = self.initialize_dummy_data()
        
        # Step 3: Compute target frequency statistics
        # Extract frequency characteristics from gradient statistics
        grad_values = torch.cat([g.flatten() for g in leaked_gradients.values()])
        grad_values = torch.nan_to_num(grad_values, nan=0.0, posinf=0.0, neginf=0.0)
        target_fft = torch.abs(torch.fft.rfft(grad_values)) ** 2
        target_fft = target_fft[:dummy_data.shape[-1]//2 + 1]  # Match length
        
        # Step 4: Setup optimizer
        criterion = nn.CrossEntropyLoss()
        
        if self.config.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS([dummy_data], lr=self.config.learning_rate, max_iter=20)
        elif self.config.optimizer == 'Adam':
            optimizer = torch.optim.Adam([dummy_data], lr=self.config.learning_rate)
        elif self.config.optimizer == 'SGD':
            optimizer = torch.optim.SGD([dummy_data], lr=self.config.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Step 5: Optimization loop
        best_loss = float('inf')
        # Always have a defined best_data fallback to avoid downstream None crashes.
        best_data = dummy_data.detach().clone()
        
        for iteration in range(self.config.max_iterations):
            def closure():
                optimizer.zero_grad()
                
                # Compute gradients of dummy data
                dummy_grads = self.compute_dummy_gradients(dummy_data, inferred_labels, criterion)
                
                # Gradient matching loss
                grad_loss = self.compute_gradient_loss(dummy_grads, leaked_gradients, self.config.reconstruction_loss)
                
                # Temporal regularization
                temporal_loss = self.compute_temporal_loss(dummy_data)
                
                # Frequency regularization
                frequency_loss = self.compute_frequency_loss(dummy_data, target_fft)
                
                # Total variation (smoothness)
                tv_loss = torch.mean(torch.abs(torch.diff(dummy_data, dim=-1)))
                
                # Total loss
                total_loss = (grad_loss + 
                            self.config.temporal_reg_weight * temporal_loss +
                            self.config.frequency_reg_weight * frequency_loss +
                            self.config.total_variation_weight * tv_loss)
                
                total_loss.backward()
                return total_loss
            
            loss = optimizer.step(closure)
            
            # Track best reconstruction
            if isinstance(loss, torch.Tensor):
                loss_value = loss.item()
            else:
                loss_value = loss
            
            # Ignore non-finite losses; keep the best finite reconstruction found so far.
            if not np.isfinite(float(loss_value)):
                continue

            if loss_value < best_loss:
                best_loss = loss_value
                best_data = dummy_data.detach().clone()
            
            # Logging
            if self.config.verbose and (iteration % 100 == 0 or iteration == self.config.max_iterations - 1):
                logger.info(f"   Iteration {iteration}/{self.config.max_iterations}: Loss = {loss_value:.6f}")
        
        # Step 6: Compute final metrics
        # Temporal coherence: measure smoothness
        temporal_coherence = float(1.0 / (1.0 + torch.mean(torch.diff(best_data, dim=-1) ** 2).item()))
        
        # Frequency match score
        best_fft = torch.abs(torch.fft.rfft(best_data.flatten())) ** 2
        a = best_fft[:len(target_fft)]
        b = target_fft
        a = a / (torch.sum(a) + 1e-10)
        b = b / (torch.sum(b) + 1e-10)
        mse = F.mse_loss(a, b)
        freq_match = float(torch.clamp(1.0 - mse, 0.0, 1.0).item())
        
        result = TSInverseResult(
            reconstructed_data=best_data.cpu().numpy(),
            reconstruction_loss=best_loss,
            iterations_used=self.config.max_iterations,
            success=best_loss < 0.1,  # Threshold for success
            label_inference_accuracy=label_accuracy,
            temporal_coherence=temporal_coherence,
            frequency_match_score=freq_match,
            metadata={
                'config': self.config.__dict__,
                'optimizer': self.config.optimizer,
                'reconstruction_loss_type': self.config.reconstruction_loss
            }
        )
        
        logger.info(f"✅ Attack complete: Loss={best_loss:.6f}, Success={result.success}")
        logger.info(f"   Temporal Coherence: {temporal_coherence:.4f}")
        logger.info(f"   Frequency Match: {freq_match:.4f}")
        
        return result


def create_simple_time_series_model(input_dim: int = 64, 
                                    sequence_length: int = 100,
                                    num_classes: int = 10,
                                    hidden_dim: int = 128) -> nn.Module:
    """
    Create a simple time-series classification model for testing
    """
    class SimpleTimeSeriesModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x):
            # x: [batch, seq_len, features]
            lstm_out, _ = self.lstm(x)
            # Take last timestep
            last_out = lstm_out[:, -1, :]
            logits = self.fc(last_out)
            return logits
    
    return SimpleTimeSeriesModel()

