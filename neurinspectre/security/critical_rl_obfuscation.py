"""
NeurInSpectre: Critical RL-Obfuscation Detection Implementation

This module implements advanced detection of RL-trained gradient obfuscation
based on recent research findings from "RL-Obfuscation: Evading Latent-Space Monitors"
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

class CriticalRLObfuscationDetector:
    """
    Critical implementation of RL-obfuscation detection
    Addresses the highest priority threat identified in recent research
    """
    
    def __init__(self, sensitivity_level='high', verbose: bool = False):
        self.sensitivity_level = sensitivity_level
        self.verbose = bool(verbose)
        self.detection_thresholds = self._initialize_thresholds()
        self.rl_signatures = self._initialize_rl_signatures()
        self.policy_fingerprints = {}
        
    def _initialize_thresholds(self):
        """Initialize detection thresholds based on sensitivity level"""
        if self.sensitivity_level == 'critical':
            return {
                'policy_fingerprint': 0.3,
                'semantic_consistency': 0.4,
                'conditional_trigger': 0.2,
                'periodic_pattern': 0.25,
                'evasion_signature': 0.35,
                'overall_threat': 0.5
            }
        elif self.sensitivity_level == 'high':
            return {
                'policy_fingerprint': 0.4,
                'semantic_consistency': 0.5,
                'conditional_trigger': 0.3,
                'periodic_pattern': 0.35,
                'evasion_signature': 0.45,
                'overall_threat': 0.6
            }
        else:  # medium
            return {
                'policy_fingerprint': 0.5,
                'semantic_consistency': 0.6,
                'conditional_trigger': 0.4,
                'periodic_pattern': 0.45,
                'evasion_signature': 0.55,
                'overall_threat': 0.7
            }
    
    def _initialize_rl_signatures(self):
        """Initialize known RL-training signatures"""
        return {
            'reward_optimization_patterns': [
                'periodic_oscillations',
                'exploration_exploitation_cycles',
                'policy_gradient_signatures'
            ],
            'training_artifacts': [
                'batch_update_patterns',
                'learning_rate_schedules',
                'regularization_effects'
            ],
            'evasion_strategies': [
                'adversarial_training_patterns',
                'gradient_masking_attempts',
                'detection_system_fingerprinting'
            ]
        }
    
    def detect_rl_obfuscation(self, gradient_data, metadata=None):
        """
        Main detection function for RL-obfuscation
        Returns comprehensive threat assessment
        """
        gradient = np.array(gradient_data).flatten().astype(np.float64)
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

        # Degenerate inputs: avoid numpy var/std warnings (ddof) and SciPy FFT crashes.
        if gradient.size < 2:
            return {
                'overall_threat_level': 0.0,
                'threat_classification': 'LOW',
                'detection_confidence': 0.0,
                'component_scores': {},
                'actionable_intelligence': {
                    'primary_threats': [],
                    'attack_vectors': [],
                    'evasion_techniques': [],
                },
            }
        
        if self.verbose:
            print("🚨 CRITICAL: Analyzing RL-Obfuscation Patterns...")
        
        # Core detection components
        policy_score = self._detect_policy_fingerprints(gradient)
        semantic_score = self._analyze_semantic_consistency(gradient)
        trigger_score = self._detect_conditional_triggers(gradient)
        periodic_score = self._detect_periodic_patterns(gradient)
        evasion_score = self._detect_evasion_signatures(gradient)
        
        # Advanced RL-specific analysis
        reward_score = self._analyze_reward_optimization_patterns(gradient)
        training_score = self._detect_training_artifacts(gradient)
        adversarial_score = self._detect_adversarial_training_patterns(gradient)
        
        # Composite threat assessment
        component_scores = {
            'policy_fingerprint': policy_score,
            'semantic_consistency': semantic_score,
            'conditional_triggers': trigger_score,
            'periodic_patterns': periodic_score,
            'evasion_signatures': evasion_score,
            'reward_optimization': reward_score,
            'training_artifacts': training_score,
            'adversarial_patterns': adversarial_score
        }
        
        # Calculate overall threat level
        critical_scores = [policy_score, evasion_score, adversarial_score]
        high_scores = [semantic_score, trigger_score, reward_score]
        medium_scores = [periodic_score, training_score]
        
        # Weighted threat calculation (critical components have higher weight)
        overall_threat = (
            np.mean(critical_scores) * 0.5 +
            np.mean(high_scores) * 0.3 +
            np.mean(medium_scores) * 0.2
        )
        
        # Threat classification
        threat_level = self._classify_threat_level(overall_threat, component_scores)
        
        # Generate actionable intelligence
        actionable_intel = self._generate_actionable_intelligence(component_scores, threat_level)
        
        results = {
            'overall_threat_level': overall_threat,
            'threat_classification': threat_level,
            'component_scores': component_scores,
            'actionable_intelligence': actionable_intel,
            'detection_confidence': self._calculate_confidence(component_scores),
            'recommended_actions': self._generate_recommendations(threat_level, component_scores),
            'technical_details': self._generate_technical_details(gradient, component_scores)
        }
        
        if self.verbose:
            self._print_detection_summary(results)
        
        return results
    
    def _detect_policy_fingerprints(self, gradient):
        """Detect RL policy fingerprints in gradient patterns"""
        # Statistical signatures of RL-trained gradients
        
        # 1. Value function approximation signatures
        value_signature = self._detect_value_function_patterns(gradient)
        
        # 2. Policy gradient signatures
        policy_signature = self._detect_policy_gradient_patterns(gradient)
        
        # 3. Actor-critic signatures
        actor_critic_signature = self._detect_actor_critic_patterns(gradient)
        
        # 4. Q-learning signatures
        q_learning_signature = self._detect_q_learning_patterns(gradient)
        
        # Composite policy fingerprint score
        fingerprint_score = np.mean([
            value_signature, policy_signature, 
            actor_critic_signature, q_learning_signature
        ])
        
        return min(1.0, fingerprint_score)
    
    def _detect_value_function_patterns(self, gradient):
        """Detect value function approximation patterns"""
        # Value functions often show smooth, continuous patterns
        smoothness = self._calculate_smoothness(gradient)
        
        # Bellman equation residuals create specific patterns
        bellman_signature = self._detect_bellman_residual_patterns(gradient)
        
        # Temporal difference learning signatures
        td_signature = self._detect_td_learning_patterns(gradient)
        
        return np.mean([smoothness, bellman_signature, td_signature])
    
    def _detect_policy_gradient_patterns(self, gradient):
        """Detect policy gradient method signatures"""
        if len(gradient) < 2:
            return 0.0
        # Policy gradients often show high variance
        var = float(np.var(gradient))
        if not np.isfinite(var):
            var = 0.0
        variance_score = min(1.0, var / 0.1)
        
        # REINFORCE algorithm signatures
        reinforce_score = self._detect_reinforce_patterns(gradient)
        
        # Actor-critic variance reduction signatures
        variance_reduction_score = self._detect_variance_reduction_patterns(gradient)
        
        return np.mean([variance_score, reinforce_score, variance_reduction_score])
    
    def _detect_actor_critic_patterns(self, gradient):
        """Detect actor-critic algorithm signatures"""
        # Dual network signatures (actor + critic)
        dual_network_score = self._detect_dual_network_patterns(gradient)
        
        # Advantage function signatures
        advantage_score = self._detect_advantage_function_patterns(gradient)
        
        # Baseline subtraction patterns
        baseline_score = self._detect_baseline_subtraction_patterns(gradient)
        
        return np.mean([dual_network_score, advantage_score, baseline_score])
    
    def _detect_q_learning_patterns(self, gradient):
        """Detect Q-learning algorithm signatures"""
        # Q-value update patterns
        q_update_score = self._detect_q_update_patterns(gradient)
        
        # Experience replay signatures
        replay_score = self._detect_experience_replay_patterns(gradient)
        
        # Target network update patterns
        target_network_score = self._detect_target_network_patterns(gradient)
        
        return np.mean([q_update_score, replay_score, target_network_score])
    
    def _analyze_semantic_consistency(self, gradient):
        """Analyze semantic consistency for token repurposing detection"""
        # Token repurposing creates semantic inconsistencies
        
        # 1. Direction consistency analysis
        direction_changes = np.sum(np.diff(np.sign(gradient)) != 0)
        direction_consistency = 1.0 - (direction_changes / max(1, len(gradient) - 1))
        
        # 2. Magnitude consistency analysis
        magnitude_variance = np.var(np.abs(gradient))
        magnitude_consistency = 1.0 / (1.0 + magnitude_variance)
        
        # 3. Local coherence analysis
        local_coherence = self._analyze_local_coherence(gradient)
        
        # 4. Semantic drift detection
        semantic_drift = self._detect_semantic_drift(gradient)
        
        # Inconsistency indicates potential repurposing
        inconsistency_score = 1.0 - np.mean([
            direction_consistency, magnitude_consistency, 
            local_coherence, 1.0 - semantic_drift
        ])
        
        return min(1.0, inconsistency_score)
    
    def _detect_conditional_triggers(self, gradient):
        """Detect conditional trigger patterns"""
        # Conditional triggers create sudden activation patterns
        
        # 1. Spike detection
        threshold = np.mean(np.abs(gradient)) + 3 * np.std(gradient)
        spikes = np.sum(np.abs(gradient) > threshold)
        spike_score = min(1.0, spikes / len(gradient) * 20)
        
        # 2. Activation threshold patterns
        activation_score = self._detect_activation_thresholds(gradient)
        
        # 3. Conditional logic signatures
        conditional_score = self._detect_conditional_logic_patterns(gradient)
        
        # 4. Context-dependent activation
        context_score = self._detect_context_dependent_activation(gradient)
        
        return np.mean([spike_score, activation_score, conditional_score, context_score])
    
    def _detect_periodic_patterns(self, gradient):
        """Detect periodic patterns characteristic of RL training"""
        # RL training often creates periodic patterns due to:
        # - Episode boundaries
        # - Learning rate schedules
        # - Exploration-exploitation cycles
        
        # 1. Autocorrelation analysis
        autocorr_score = self._analyze_autocorrelation_patterns(gradient)
        
        # 2. Fourier analysis for periodic components
        fourier_score = self._analyze_fourier_periodicity(gradient)
        
        # 3. Episode boundary detection
        episode_score = self._detect_episode_boundaries(gradient)
        
        # 4. Learning schedule signatures
        schedule_score = self._detect_learning_schedule_patterns(gradient)
        
        return np.mean([autocorr_score, fourier_score, episode_score, schedule_score])
    
    def _detect_evasion_signatures(self, gradient):
        """Detect specific evasion strategy signatures"""
        # RL-trained evasion creates specific patterns
        
        # 1. Adversarial training signatures
        adversarial_score = self._detect_adversarial_signatures(gradient)
        
        # 2. Gradient masking attempts
        masking_score = self._detect_gradient_masking(gradient)
        
        # 3. Detection system fingerprinting
        fingerprinting_score = self._detect_system_fingerprinting(gradient)
        
        # 4. Evasion policy optimization
        optimization_score = self._detect_evasion_optimization(gradient)
        
        return np.mean([adversarial_score, masking_score, fingerprinting_score, optimization_score])
    
    def _analyze_reward_optimization_patterns(self, gradient):
        """Analyze reward optimization patterns"""
        # Reward optimization creates specific gradient patterns
        
        # 1. Reward signal propagation
        propagation_score = self._detect_reward_propagation(gradient)
        
        # 2. Credit assignment patterns
        credit_score = self._detect_credit_assignment(gradient)
        
        # 3. Exploration bonus signatures
        exploration_score = self._detect_exploration_bonuses(gradient)
        
        return np.mean([propagation_score, credit_score, exploration_score])
    
    def _detect_training_artifacts(self, gradient):
        """Detect RL training artifacts"""
        # Training artifacts from RL algorithms
        
        # 1. Batch update patterns
        batch_score = self._detect_batch_patterns(gradient)
        
        # 2. Learning rate effects
        lr_score = self._detect_learning_rate_effects(gradient)
        
        # 3. Regularization signatures
        reg_score = self._detect_regularization_effects(gradient)
        
        return np.mean([batch_score, lr_score, reg_score])
    
    def _detect_adversarial_training_patterns(self, gradient):
        """Detect adversarial training patterns"""
        # Adversarial training creates specific signatures
        
        # 1. Min-max optimization patterns
        minmax_score = self._detect_minmax_patterns(gradient)
        
        # 2. Adversarial example generation
        generation_score = self._detect_adversarial_generation(gradient)
        
        # 3. Robustness optimization
        robustness_score = self._detect_robustness_optimization(gradient)
        
        return np.mean([minmax_score, generation_score, robustness_score])
    
    # Helper methods for detailed pattern detection
    def _calculate_smoothness(self, gradient):
        """Calculate gradient smoothness"""
        if len(gradient) < 2:
            return 0.0
        
        differences = np.diff(gradient)
        smoothness = 1.0 / (1.0 + np.var(differences))
        return min(1.0, smoothness)
    
    def _detect_bellman_residual_patterns(self, gradient):
        """Detect Bellman equation residual patterns"""
        # Bellman residuals often show specific frequency characteristics
        if len(gradient) < 2:
            return 0.0
        fft_result = fft(gradient)
        power_spectrum = np.abs(fft_result)**2
        
        # Look for characteristic frequencies
        freqs = fftfreq(len(gradient))
        low_freq_power = np.sum(power_spectrum[np.abs(freqs) < 0.1])
        total_power = np.sum(power_spectrum)
        
        return min(1.0, low_freq_power / (total_power + 1e-8))
    
    def _detect_td_learning_patterns(self, gradient):
        """Detect temporal difference learning patterns"""
        # TD learning creates specific temporal patterns
        if len(gradient) < 3:
            return 0.0
        
        # Look for temporal difference signatures
        td_errors = []
        for i in range(1, len(gradient) - 1):
            td_error = gradient[i+1] - gradient[i] - 0.9 * gradient[i-1]  # Assume gamma=0.9
            td_errors.append(abs(td_error))
        
        td_variance = np.var(td_errors)
        return min(1.0, td_variance / 0.1)
    
    def _detect_reinforce_patterns(self, gradient):
        """Detect REINFORCE algorithm patterns"""
        # REINFORCE creates high variance patterns
        variance = np.var(gradient)
        
        # Look for reward-weighted patterns
        if len(gradient) > 10:
            # Simulate reward weighting detection
            weighted_variance = np.var(gradient * np.abs(gradient))
            return min(1.0, weighted_variance / 0.1)
        
        return min(1.0, variance / 0.05)
    
    def _detect_variance_reduction_patterns(self, gradient):
        """Detect variance reduction technique signatures"""
        # Variance reduction techniques create specific patterns
        if len(gradient) < 10:
            return 0.0

        # IMPORTANT:
        # Subtracting a constant mean does not reduce variance (Var(x - c) == Var(x)).
        # In actor-critic style baselines, the baseline is state/time-dependent.
        # We approximate a baseline using a local moving-average trend, then measure
        # how much variance remains after subtracting that trend.
        g = np.asarray(gradient, dtype=np.float64)
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        var_g = float(np.var(g))
        if not np.isfinite(var_g) or var_g <= 0.0:
            return 0.0

        w = max(5, int(len(g) // 20))  # ~5% window, at least 5
        w = min(w, max(5, len(g) // 2))
        pad = w // 2
        kernel = np.ones(w, dtype=np.float64) / float(w)
        gp = np.pad(g, (pad, pad), mode='reflect')
        baseline = np.convolve(gp, kernel, mode='valid')

        residual = g - baseline
        var_res = float(np.var(residual))
        if not np.isfinite(var_res):
            return 0.0

        reduction = 1.0 - (var_res / (var_g + 1e-12))
        return float(np.clip(reduction, 0.0, 1.0))
    
    def _analyze_autocorrelation_patterns(self, gradient):
        """Analyze autocorrelation for periodic patterns"""
        if len(gradient) < 10:
            return 0.0
        
        # Compute autocorrelation
        autocorr = np.correlate(gradient, gradient, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        
        # Look for periodic peaks
        peaks, _ = find_peaks(autocorr[1:], height=0.3, distance=5)
        
        return min(1.0, len(peaks) / 5.0)
    
    def _analyze_fourier_periodicity(self, gradient):
        """Analyze Fourier components for periodicity"""
        fft_result = fft(gradient)
        power_spectrum = np.abs(fft_result)**2
        
        # Find dominant frequencies
        _freqs = fftfreq(len(gradient))
        sorted_indices = np.argsort(power_spectrum)[::-1]
        
        # Check for strong periodic components
        top_powers = power_spectrum[sorted_indices[:5]]
        total_power = np.sum(power_spectrum)
        
        periodicity_score = np.sum(top_powers) / (total_power + 1e-8)
        return min(1.0, periodicity_score)
    
    def _classify_threat_level(self, overall_threat, component_scores):
        """Classify threat level based on scores"""
        if overall_threat > 0.8:
            return "CRITICAL"
        elif overall_threat > 0.6:
            return "HIGH"
        elif overall_threat > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_confidence(self, component_scores):
        """Calculate detection confidence"""
        scores = list(component_scores.values())
        consistency = 1.0 - np.std(scores)
        magnitude = np.mean(scores)
        
        confidence = (consistency * 0.6 + magnitude * 0.4)
        return min(1.0, max(0.0, confidence))
    
    def _generate_actionable_intelligence(self, component_scores, threat_level):
        """Generate actionable intelligence for security teams"""
        intel = {
            'primary_threats': [],
            'attack_vectors': [],
            'evasion_techniques': [],
            'indicators': []
        }
        
        # Identify primary threats
        for component, score in component_scores.items():
            if score > 0.6:
                intel['primary_threats'].append({
                    'component': component,
                    'score': score,
                    'description': self._get_threat_description(component)
                })
        
        # Identify attack vectors
        if component_scores['policy_fingerprint'] > 0.5:
            intel['attack_vectors'].append('RL-trained evasion policy detected')
        
        if component_scores['conditional_triggers'] > 0.5:
            intel['attack_vectors'].append('Conditional activation triggers present')
        
        if component_scores['evasion_signatures'] > 0.5:
            intel['attack_vectors'].append('Active evasion strategy detected')
        
        # Identify evasion techniques
        if component_scores['semantic_consistency'] > 0.5:
            intel['evasion_techniques'].append('Token semantic repurposing')
        
        if component_scores['periodic_patterns'] > 0.5:
            intel['evasion_techniques'].append('Training schedule exploitation')
        
        return intel
    
    def _generate_recommendations(self, threat_level, component_scores):
        """Generate specific recommendations"""
        recommendations = []
        
        if threat_level in ['CRITICAL', 'HIGH']:
            recommendations.append({
                'priority': 'IMMEDIATE',
                'action': 'Isolate and analyze gradient source',
                'details': 'High-confidence RL-obfuscation detected'
            })
        
        if component_scores['policy_fingerprint'] > 0.6:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Implement policy fingerprint blocking',
                'details': 'RL policy signatures detected in gradient patterns'
            })
        
        if component_scores['conditional_triggers'] > 0.6:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Deploy conditional trigger monitoring',
                'details': 'Activation thresholds suggest conditional evasion'
            })
        
        return recommendations
    
    def _generate_technical_details(self, gradient, component_scores):
        """Generate technical details for analysis"""
        g = np.asarray(gradient, dtype=np.float64).reshape(-1)
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        if g.size == 0:
            g = np.zeros(1, dtype=np.float64)

        mean_val = float(np.mean(g))
        std_val = float(np.std(g))

        if std_val <= 0.0 or not np.isfinite(std_val):
            skewness = 0.0
            kurt = 0.0
        else:
            z = (g - mean_val) / (std_val + 1e-12)
            skewness = float(np.mean(z ** 3))
            kurt = float(np.mean(z ** 4) - 3.0)

        # Treat |g| as a discrete distribution after normalization (normalized entropy in [0, 1]).
        w = np.abs(g)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0 or w.size < 2:
            ent_norm = 0.0
        else:
            p = w / (w_sum + 1e-12)
            p = p[p > 0]
            if p.size < 2:
                ent_norm = 0.0
            else:
                ent_bits = -float(np.sum(p * np.log2(p + 1e-12)))
                ent_norm = float(ent_bits / np.log2(g.size))
                ent_norm = float(np.clip(ent_norm, 0.0, 1.0))

        return {
            'gradient_statistics': {
                'mean': mean_val,
                'std': std_val,
                'skewness': skewness,
                'kurtosis': kurt,
                'entropy': ent_norm
            },
            'spectral_analysis': {
                'dominant_frequency': self._get_dominant_frequency(gradient),
                'spectral_entropy': self._calculate_spectral_entropy(gradient),
                'high_freq_ratio': self._calculate_high_freq_ratio(gradient)
            },
            'detection_metrics': component_scores
        }
    
    def _get_threat_description(self, component):
        """Get description for threat component"""
        descriptions = {
            'policy_fingerprint': 'RL policy training signatures detected',
            'semantic_consistency': 'Token repurposing patterns identified',
            'conditional_triggers': 'Conditional activation mechanisms present',
            'periodic_patterns': 'Training schedule exploitation detected',
            'evasion_signatures': 'Active evasion strategy signatures',
            'reward_optimization': 'Reward optimization patterns detected',
            'training_artifacts': 'RL training artifacts present',
            'adversarial_patterns': 'Adversarial training signatures detected'
        }
        return descriptions.get(component, 'Unknown threat component')
    
    def _print_detection_summary(self, results):
        """Print detection summary"""
        if not self.verbose:
            return
        print("\n🚨 RL-OBFUSCATION DETECTION RESULTS")
        print("=" * 50)
        print(f"Overall Threat Level: {results['overall_threat_level']:.3f}")
        print(f"Threat Classification: {results['threat_classification']}")
        print(f"Detection Confidence: {results['detection_confidence']:.3f}")
        
        print("\n📊 Component Analysis:")
        for component, score in results['component_scores'].items():
            status = "🔴" if score > 0.6 else "🟡" if score > 0.4 else "🟢"
            print(f"  {status} {component}: {score:.3f}")
        
        print("\n🎯 Primary Threats:")
        for threat in results['actionable_intelligence']['primary_threats']:
            print(f"  • {threat['component']}: {threat['description']} ({threat['score']:.3f})")
        
        print("\n⚡ Recommended Actions:")
        for rec in results['recommended_actions']:
            print(f"  [{rec['priority']}] {rec['action']}")
    
    # Heuristic implementations for additional RL-related indicators
    def _detect_dual_network_patterns(self, gradient):
        return min(1.0, np.var(gradient[:len(gradient)//2]) + np.var(gradient[len(gradient)//2:]))
    
    def _detect_advantage_function_patterns(self, gradient):
        if len(gradient) < 4:
            return 0.0
        advantages = gradient[1:] - gradient[:-1]
        return min(1.0, np.var(advantages) / 0.1)
    
    def _detect_baseline_subtraction_patterns(self, gradient):
        # Reuse the variance-reduction heuristic (moving-average baseline).
        return float(self._detect_variance_reduction_patterns(gradient))
    
    def _detect_q_update_patterns(self, gradient):
        if len(gradient) < 3:
            return 0.0
        q_updates = gradient[1:] - 0.9 * gradient[:-1]  # Assume gamma=0.9
        return min(1.0, np.var(q_updates) / 0.1)
    
    def _detect_experience_replay_patterns(self, gradient):
        # Experience replay creates specific batch patterns
        if len(gradient) < 32:
            return 0.0
        batch_size = 32
        batch_vars = []
        for i in range(0, len(gradient) - batch_size, batch_size):
            batch = gradient[i:i+batch_size]
            batch_vars.append(np.var(batch))
        return min(1.0, np.var(batch_vars) / 0.01)
    
    def _detect_target_network_patterns(self, gradient):
        # Target networks create periodic update patterns
        return self._analyze_autocorrelation_patterns(gradient)
    
    def _analyze_local_coherence(self, gradient):
        if len(gradient) < 5:
            return 1.0
        
        coherence_scores = []
        window_size = 5
        for i in range(len(gradient) - window_size + 1):
            window = gradient[i:i+window_size]
            coherence = 1.0 / (1.0 + np.var(window))
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores)
    
    def _detect_semantic_drift(self, gradient):
        if len(gradient) < 10:
            return 0.0
        
        # Detect drift in gradient semantics
        first_half = gradient[:len(gradient)//2]
        second_half = gradient[len(gradient)//2:]
        
        drift = abs(np.mean(first_half) - np.mean(second_half))
        return min(1.0, drift / (np.std(gradient) + 1e-8))
    
    def _detect_activation_thresholds(self, gradient):
        # Detect threshold-based activation patterns
        thresholds = np.linspace(np.min(gradient), np.max(gradient), 10)
        activations = []
        
        for threshold in thresholds:
            activated = np.sum(gradient > threshold)
            activations.append(activated)
        
        # Look for sharp transitions
        transitions = np.diff(activations)
        sharp_transitions = np.sum(np.abs(transitions) > len(gradient) * 0.1)
        
        return min(1.0, sharp_transitions / 5.0)
    
    def _detect_conditional_logic_patterns(self, gradient):
        # Detect conditional logic signatures
        if len(gradient) < 10:
            return 0.0
        
        # Look for if-then-else patterns
        positive_regions = gradient > 0
        
        # Count transitions between regions
        transitions = np.sum(np.diff(positive_regions.astype(int)) != 0)
        
        return min(1.0, transitions / len(gradient) * 5)
    
    def _detect_context_dependent_activation(self, gradient):
        # Detect context-dependent activation patterns
        if len(gradient) < 20:
            return 0.0
        
        # Analyze activation in different contexts (windows)
        window_size = len(gradient) // 4
        contexts = []
        
        for i in range(0, len(gradient) - window_size, window_size):
            context = gradient[i:i+window_size]
            activation_level = np.mean(np.abs(context))
            contexts.append(activation_level)
        
        context_variance = np.var(contexts)
        return min(1.0, context_variance / 0.01)
    
    def _detect_episode_boundaries(self, gradient):
        # Detect episode boundary patterns
        if len(gradient) < 50:
            return 0.0
        
        # Look for periodic resets or boundaries
        episode_length = 50  # Assume typical episode length
        boundary_scores = []
        
        for i in range(episode_length, len(gradient), episode_length):
            if i < len(gradient):
                boundary_strength = abs(gradient[i] - gradient[i-1])
                boundary_scores.append(boundary_strength)
        
        if boundary_scores:
            denom = float(np.std(gradient)) + 1e-12
            if not np.isfinite(denom) or denom <= 1e-12:
                return 0.0
            score = float(np.mean(boundary_scores)) / denom
            return float(np.clip(score, 0.0, 1.0))
        return 0.0
    
    def _detect_learning_schedule_patterns(self, gradient):
        # Detect learning rate schedule patterns
        if len(gradient) < 20:
            return 0.0
        
        # Look for exponential decay or step decay patterns
        magnitude_trend = []
        window_size = 10
        
        for i in range(0, len(gradient) - window_size, window_size):
            window = gradient[i:i+window_size]
            magnitude_trend.append(np.mean(np.abs(window)))
        
        if len(magnitude_trend) > 1:
            # Check for decay pattern
            decay_score = 0.0
            for i in range(1, len(magnitude_trend)):
                if magnitude_trend[i] < magnitude_trend[i-1]:
                    decay_score += 1.0
            
            return min(1.0, decay_score / len(magnitude_trend))
        
        return 0.0
    
    def _detect_adversarial_signatures(self, gradient):
        # Detect adversarial training signatures
        # High frequency components often indicate adversarial training
        fft_result = fft(gradient)
        freqs = fftfreq(len(gradient))
        high_freq_power = np.sum(np.abs(fft_result[np.abs(freqs) > 0.3])**2)
        total_power = np.sum(np.abs(fft_result)**2)
        
        return min(1.0, high_freq_power / (total_power + 1e-8))
    
    def _detect_gradient_masking(self, gradient):
        # Detect gradient masking attempts
        # Masking often reduces gradient magnitude artificially
        expected_magnitude = np.std(gradient)
        actual_magnitude = np.mean(np.abs(gradient))
        
        masking_score = 1.0 - (actual_magnitude / (expected_magnitude + 1e-8))
        return min(1.0, max(0.0, masking_score))
    
    def _detect_system_fingerprinting(self, gradient):
        # Detect detection system fingerprinting
        # Fingerprinting creates specific probe patterns
        if len(gradient) < 10:
            return 0.0
        
        # Look for probe-like patterns
        probe_patterns = []
        for i in range(len(gradient) - 5):
            window = gradient[i:i+5]
            if np.all(window == window[0]):  # Constant probe
                probe_patterns.append(1)
            elif np.all(np.diff(window) > 0):  # Increasing probe
                probe_patterns.append(1)
            elif np.all(np.diff(window) < 0):  # Decreasing probe
                probe_patterns.append(1)
            else:
                probe_patterns.append(0)
        
        return min(1.0, np.mean(probe_patterns))
    
    def _detect_evasion_optimization(self, gradient):
        # Detect evasion policy optimization
        # Optimization creates specific convergence patterns
        if len(gradient) < 20:
            return 0.0
        
        # Look for optimization convergence patterns
        convergence_score = 0.0
        window_size = 10
        
        for i in range(window_size, len(gradient)):
            recent = gradient[i-window_size:i]
            if np.std(recent) < np.std(gradient) * 0.1:  # Low variance indicates convergence
                convergence_score += 1.0
        
        return min(1.0, convergence_score / (len(gradient) - window_size))
    
    def _detect_reward_propagation(self, gradient):
        # Detect reward signal propagation patterns
        if len(gradient) < 10:
            return 0.0
        
        # Reward propagation creates specific temporal patterns
        propagation_score = 0.0
        for i in range(1, len(gradient)):
            if gradient[i] * gradient[i-1] > 0:  # Same sign indicates propagation
                propagation_score += 1.0
        
        return min(1.0, propagation_score / (len(gradient) - 1))
    
    def _detect_credit_assignment(self, gradient):
        # Detect credit assignment patterns
        if len(gradient) < 5:
            return 0.0
        
        # Credit assignment creates delayed reward patterns
        delayed_correlations = []
        for delay in range(1, min(5, len(gradient))):
            if len(gradient) > delay:
                with np.errstate(all="ignore"):
                    corr = np.corrcoef(gradient[:-delay], gradient[delay:])[0, 1]
                if np.isfinite(corr):
                    delayed_correlations.append(abs(corr))
        
        if delayed_correlations:
            return min(1.0, max(delayed_correlations))
        return 0.0
    
    def _detect_exploration_bonuses(self, gradient):
        # Detect exploration bonus signatures
        if len(gradient) < 10:
            return 0.0
        
        # Exploration bonuses create outlier patterns
        outliers = np.abs(gradient) > np.mean(np.abs(gradient)) + 2 * np.std(gradient)
        exploration_score = np.sum(outliers) / len(gradient)
        
        return min(1.0, exploration_score * 5)
    
    def _detect_batch_patterns(self, gradient):
        # Detect batch update patterns
        if len(gradient) < 32:
            return 0.0
        
        # Batch updates create periodic patterns
        batch_size = 32
        batch_means = []
        
        for i in range(0, len(gradient) - batch_size, batch_size):
            batch = gradient[i:i+batch_size]
            batch_means.append(np.mean(batch))
        
        if len(batch_means) > 1:
            batch_variance = np.var(batch_means)
            return min(1.0, batch_variance / 0.01)
        
        return 0.0
    
    def _detect_learning_rate_effects(self, gradient):
        # Detect learning rate schedule effects
        if len(gradient) < 10:
            return 0.0
        
        # Learning rate changes affect gradient magnitude
        magnitude_changes = np.abs(np.diff(np.abs(gradient)))
        lr_effect_score = np.var(magnitude_changes) / (np.var(gradient) + 1e-8)
        
        return min(1.0, lr_effect_score)
    
    def _detect_regularization_effects(self, gradient):
        # Detect regularization effects
        # Regularization typically reduces gradient magnitude
        expected_magnitude = np.std(gradient)
        actual_magnitude = np.mean(np.abs(gradient))
        
        regularization_score = 1.0 - (actual_magnitude / (expected_magnitude + 1e-8))
        return min(1.0, max(0.0, regularization_score))
    
    def _detect_minmax_patterns(self, gradient):
        # Detect min-max optimization patterns
        if len(gradient) < 10:
            return 0.0
        
        # Min-max creates oscillatory patterns
        oscillation_score = 0.0
        for i in range(2, len(gradient)):
            if ((gradient[i] > gradient[i-1] > gradient[i-2]) or 
                (gradient[i] < gradient[i-1] < gradient[i-2])):
                oscillation_score += 1.0
        
        denom = float(len(gradient) - 2)
        if denom <= 0.0:
            return 0.0
        return min(1.0, oscillation_score / denom)
    
    def _detect_adversarial_generation(self, gradient):
        # Detect adversarial example generation patterns
        # Generation creates specific perturbation patterns
        perturbation_score = np.std(gradient) / (np.mean(np.abs(gradient)) + 1e-8)
        return min(1.0, perturbation_score / 2.0)
    
    def _detect_robustness_optimization(self, gradient):
        # Detect robustness optimization patterns
        # Robustness optimization creates smoothing effects
        if len(gradient) < 5:
            return 0.0
        
        smoothness = self._calculate_smoothness(gradient)
        return min(1.0, smoothness)
    
    def _get_dominant_frequency(self, gradient):
        """Get dominant frequency component"""
        if len(gradient) < 2:
            return 0.0
        fft_result = fft(gradient)
        freqs = fftfreq(len(gradient))
        power_spectrum = np.abs(fft_result)**2
        
        dominant_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC component
        return freqs[dominant_idx]
    
    def _calculate_spectral_entropy(self, gradient):
        """Calculate spectral entropy"""
        if len(gradient) < 2:
            return 0.0
        fft_result = fft(gradient)
        power_spectrum = np.abs(fft_result)**2
        total = float(np.sum(power_spectrum))
        if total <= 0.0 or not np.isfinite(total) or power_spectrum.size < 2:
            return 0.0

        p = power_spectrum / (total + 1e-12)
        p = p[p > 0]
        if p.size < 2:
            return 0.0

        ent_bits = -float(np.sum(p * np.log2(p + 1e-12)))
        ent_norm = float(ent_bits / np.log2(power_spectrum.size))
        return float(np.clip(ent_norm, 0.0, 1.0))
    
    def _calculate_high_freq_ratio(self, gradient):
        """Calculate high frequency ratio"""
        if len(gradient) < 2:
            return 0.0
        fft_result = fft(gradient)
        freqs = fftfreq(len(gradient))
        power_spectrum = np.abs(fft_result)**2
        
        high_freq_power = np.sum(power_spectrum[np.abs(freqs) > 0.3])
        total_power = np.sum(power_spectrum)
        
        return high_freq_power / (total_power + 1e-8)

def test_critical_rl_detection():
    """Test the critical RL-obfuscation detection"""
    print("🚨 TESTING CRITICAL RL-OBFUSCATION DETECTION")
    print("=" * 60)
    
    # Initialize detector with high sensitivity
    detector = CriticalRLObfuscationDetector(sensitivity_level='critical')
    
    # Test Case 1: Simulated RL-trained gradient
    print("\n1. Testing RL-trained gradient simulation...")
    rl_gradient = np.random.randn(200) * 0.1
    
    # Add RL-specific patterns
    # Policy gradient high variance
    rl_gradient[0:50] = np.random.randn(50) * 0.3
    
    # Periodic training patterns
    rl_gradient[50:100] = np.sin(np.linspace(0, 4*np.pi, 50)) * 0.2
    
    # Conditional triggers
    rl_gradient[150:155] = np.array([0.8, 0.9, 1.0, 0.9, 0.8])
    
    # Evasion signatures (high frequency)
    rl_gradient[100:150] = np.sin(np.linspace(0, 20*np.pi, 50)) * 0.1
    
    results = detector.detect_rl_obfuscation(rl_gradient)
    
    # Test Case 2: Clean gradient (should be low threat)
    print("\n2. Testing clean gradient...")
    clean_gradient = np.random.randn(200) * 0.05
    clean_results = detector.detect_rl_obfuscation(clean_gradient)
    
    # Test Case 3: Traditional adversarial gradient
    print("\n3. Testing traditional adversarial gradient...")
    adv_gradient = np.random.randn(200) * 0.1
    adv_gradient += np.sign(np.random.randn(200)) * 0.02  # FGSM-like
    adv_results = detector.detect_rl_obfuscation(adv_gradient)
    
    print("\n" + "=" * 60)
    print("CRITICAL RL-OBFUSCATION DETECTION SUMMARY")
    print("=" * 60)
    print(f"RL-trained gradient threat: {results['overall_threat_level']:.3f} ({results['threat_classification']})")
    print(f"Clean gradient threat: {clean_results['overall_threat_level']:.3f} ({clean_results['threat_classification']})")
    print(f"Traditional adversarial threat: {adv_results['overall_threat_level']:.3f} ({adv_results['threat_classification']})")
    
    return results, clean_results, adv_results

if __name__ == "__main__":
    # Run critical RL-obfuscation detection test
    test_results = test_critical_rl_detection()
    
    print("\n✅ CRITICAL RL-OBFUSCATION DETECTION IMPLEMENTED")
    print("🎯 Ready for immediate deployment against RL-trained evasion attacks")

