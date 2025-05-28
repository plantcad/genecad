import numpy as np
import numpy.typing as npt
from numba import njit
from typing import Protocol
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle
from sklearn.metrics import f1_score, accuracy_score

logger = logging.getLogger(__name__)


class Optimizer(ABC):
    """Abstract base class for optimizers."""
    
    @abstractmethod
    def update(self, params: dict[str, np.ndarray], gradients: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Update parameters using gradients."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset optimizer state."""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def update(self, params: dict[str, np.ndarray], gradients: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Update parameters using SGD."""
        updated_params = {}
        for name, param in params.items():
            if name in gradients:
                updated_params[name] = param - self.learning_rate * gradients[name]
            else:
                updated_params[name] = param
        return updated_params
    
    def reset(self):
        """SGD has no state to reset."""
        pass


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step
    
    def update(self, params: dict[str, np.ndarray], gradients: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Update parameters using Adam."""
        self.t += 1
        updated_params = {}
        
        for name, param in params.items():
            if name in gradients:
                grad = gradients[name]
                
                # Initialize moments if needed
                if name not in self.m:
                    self.m[name] = np.zeros_like(param)
                    self.v[name] = np.zeros_like(param)
                
                # Update biased first moment estimate
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                updated_params[name] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            else:
                updated_params[name] = param
        
        return updated_params
    
    def reset(self):
        """Reset Adam optimizer state."""
        self.m = {}
        self.v = {}
        self.t = 0


@dataclass
class TrainingSequence:
    """Training sequence data for CRF."""
    emissions: np.ndarray  # Shape (T, N) - emission probabilities
    true_path: np.ndarray  # Shape (T,) - true state sequence
    
    def __post_init__(self):
        """Validate the training sequence data."""
        if len(self.emissions.shape) != 2:
            raise ValueError(f"emissions must be 2D, got shape {self.emissions.shape}")
        if len(self.true_path.shape) != 1:
            raise ValueError(f"true_path must be 1D, got shape {self.true_path.shape}")
        if self.emissions.shape[0] != len(self.true_path):
            raise ValueError(f"Length mismatch: emissions {self.emissions.shape[0]} vs true_path {len(self.true_path)}")


class FeatureSet(ABC):
    """Abstract base class for feature extraction and storage."""
    
    def __init__(self, emissions: np.ndarray):
        self.emissions = emissions
        self.T, self.N = emissions.shape
        
    @property
    @abstractmethod
    def emission_dim(self) -> int:
        """Number of emission feature dimensions."""
        pass
    
    @property
    @abstractmethod
    def transition_dim(self) -> int:
        """Number of transition feature dimensions."""
        pass
    
    @abstractmethod
    def get_emission_potentials(self, emission_weights: np.ndarray) -> np.ndarray:
        """Get emission feature potentials for all positions and states as (T, N) array."""
        pass
    
    @abstractmethod
    def get_transition_potentials(self, transition_weights: np.ndarray) -> np.ndarray:
        """Get transition feature potentials for all positions as (T, N, N) array."""
        pass
    
    @abstractmethod
    def compute_emission_gradients(self, marginals: np.ndarray, true_path: np.ndarray) -> np.ndarray:
        """Compute emission feature gradients given marginals and true path."""
        pass
    
    @abstractmethod
    def compute_transition_gradients(self, expected_transitions: np.ndarray, true_path: np.ndarray) -> np.ndarray:
        """Compute transition feature gradients given expected transitions and true path."""
        pass


@njit
def _compute_genic_content_features(emissions: np.ndarray, num_samples: int = 512, stride: int = 32, intergenic_idx: int = 0) -> np.ndarray:
    """Compute genic content features for genomic context analysis.
    
    Quantifies how much of the surrounding context (upstream/downstream) is not intergenic.
    Uses strided sampling to cover a larger receptive field efficiently.
    
    Parameters
    ----------
    emissions : np.ndarray
        Emission probabilities of shape (T, N)
    num_samples : int
        Number of sampling points in each direction (default: 512)
    stride : int
        Stride for sampling positions within the window (default: 32)
        Actual receptive field will be num_samples * stride bp
    intergenic_idx : int
        Index of intergenic state (default: 0)
        
    Returns
    -------
    np.ndarray
        Genic content features of shape (T, 2) where:
        - result[t, 0] = upstream genic content (fraction non-intergenic)
        - result[t, 1] = downstream genic content (fraction non-intergenic)
    """
    T, N = emissions.shape
    result = np.zeros((T, 2))
    
    for t in range(T):
        # Upstream genic content with strided sampling
        upstream_genic = 0.0
        upstream_count = 0
        
        for i in range(num_samples):
            pos = t - (i + 1) * stride  # Sample at stride intervals going backwards
            if pos >= 0:
                # Sum all non-intergenic probabilities at this position
                genic_prob = 0.0
                for state in range(N):
                    if state != intergenic_idx:
                        genic_prob += emissions[pos, state]
                upstream_genic += genic_prob
                upstream_count += 1
        
        if upstream_count > 0:
            result[t, 0] = upstream_genic / upstream_count
        else:
            result[t, 0] = 0.0
        
        # Downstream genic content with strided sampling
        downstream_genic = 0.0
        downstream_count = 0
        
        for i in range(num_samples):
            pos = t + (i + 1) * stride  # Sample at stride intervals going forwards
            if pos < T:
                # Sum all non-intergenic probabilities at this position
                genic_prob = 0.0
                for state in range(N):
                    if state != intergenic_idx:
                        genic_prob += emissions[pos, state]
                downstream_genic += genic_prob
                downstream_count += 1
        
        if downstream_count > 0:
            result[t, 1] = downstream_genic / downstream_count
        else:
            result[t, 1] = 0.0
    
    return result


@njit
def _compute_transition_potentials_genomic(T: int, N: int, transition_weights: np.ndarray) -> np.ndarray:
    """Compute transition potentials for genomic features (numba-optimized)."""
    features = np.zeros((T, N, N, 1))
    
    for t in range(T):
        pos_feature = t / T  # Normalized position
        for i in range(N):
            for j in range(N):
                features[t, i, j, 0] = pos_feature
    
    # Compute potentials as weighted sum
    potentials = np.zeros((T, N, N))
    for t in range(T):
        for i in range(N):
            for j in range(N):
                potentials[t, i, j] = features[t, i, j, 0] * transition_weights[i, j, 0]
    
    return potentials


@njit
def _compute_emission_gradients_genomic(marginals: np.ndarray, true_path: np.ndarray, genic_features: np.ndarray) -> np.ndarray:
    """Compute emission gradients for genomic features (numba-optimized)."""
    T, N = marginals.shape
    grad = np.zeros(2)  # upstream + downstream features
    
    # Expected features (weighted by marginals)
    expected = np.zeros(2)
    for t in range(T):
        for state in range(N):
            for feat in range(2):
                expected[feat] += marginals[t, state] * genic_features[t, feat]
    
    # Observed features (from true path)
    observed = np.zeros(2)
    for t in range(len(true_path)):
        for feat in range(2):
            observed[feat] += genic_features[t, feat]
    
    # Gradient is expected - observed
    grad[0] = expected[0] - observed[0]
    grad[1] = expected[1] - observed[1]
    
    return grad


@njit
def _compute_transition_gradients_genomic(expected_transitions: np.ndarray, true_path: np.ndarray, T: int, N: int) -> np.ndarray:
    """Compute transition gradients for genomic features (numba-optimized)."""
    grad = np.zeros((N, N, 1))
    
    # Expected: weight by expected transitions and average position
    avg_pos = 0.5  # Simple approximation
    for i in range(N):
        for j in range(N):
            grad[i, j, 0] += expected_transitions[i, j] * avg_pos
    
    # Observed: subtract actual transition positions
    for t in range(1, len(true_path)):
        prev_state = true_path[t-1]
        curr_state = true_path[t]
        pos_feature = (t-1) / T
        grad[prev_state, curr_state, 0] -= pos_feature
    
    return grad


class GenomicFeatureSet(FeatureSet):
    """Genomic feature set using genic content analysis."""
    
    def __init__(self, emissions: np.ndarray, num_samples: int = 512, stride: int = 32, intergenic_idx: int = 0):
        super().__init__(emissions)
        self.num_samples = num_samples
        self.stride = stride
        self.intergenic_idx = intergenic_idx
        
        # Precompute genic content features
        self.genic_features = _compute_genic_content_features(emissions, num_samples, stride, intergenic_idx)
    
    @property
    def emission_dim(self) -> int:
        """Number of emission feature dimensions: upstream + downstream genic content."""
        return 2
    
    @property
    def transition_dim(self) -> int:
        """Number of transition feature dimensions: simple position-based feature."""
        return 1
    
    def get_emission_potentials(self, emission_weights: np.ndarray) -> np.ndarray:
        """Get emission potentials using genic content features."""
        # Simple linear combination of genic content features
        potentials_1d = np.dot(self.genic_features, emission_weights[:2])  # Use first 2 weights for upstream/downstream
        # Broadcast to (T, N) shape - same potential for all states at each position
        return potentials_1d[:, None] * np.ones((1, self.N))
    
    def get_transition_potentials(self, transition_weights: np.ndarray) -> np.ndarray:
        """Get transition potentials using simple position-based features."""
        return _compute_transition_potentials_genomic(self.T, self.N, transition_weights)
    
    def compute_emission_gradients(self, marginals: np.ndarray, true_path: np.ndarray) -> np.ndarray:
        """Compute emission feature gradients."""
        return _compute_emission_gradients_genomic(marginals, true_path, self.genic_features)
    
    def compute_transition_gradients(self, expected_transitions: np.ndarray, true_path: np.ndarray) -> np.ndarray:
        """Compute transition feature gradients."""
        return _compute_transition_gradients_genomic(expected_transitions, true_path, self.T, self.N)


def _compute_features_worker(args: tuple) -> FeatureSet:
    """Worker function for computing features in parallel."""
    seq, feature_set_class, feature_kwargs = args
    return feature_set_class(seq.emissions, **feature_kwargs)


class CRF:
    """Feature-dependent CRF with learned transitions and emission features."""
    
    def __init__(self, num_states: int, optimizer: Optimizer | None = None, base_transitions: np.ndarray | None = None, seed: int = 42):
        """Initialize the CRF.
        
        Parameters
        ----------
        num_states : int
            Number of states in the CRF
        optimizer : Optimizer, optional
            Optimizer for parameter updates (default: SGD with lr=0.01)
        base_transitions : np.ndarray, optional
            Base transition matrix of shape (num_states, num_states).
            If None, initializes to zeros (default: None)
        seed : int
            Random seed for weight initialization (default: 42)
        """
        self.num_states = num_states
        self.optimizer = optimizer if optimizer is not None else SGD(learning_rate=0.01)
        self.rng = np.random.RandomState(seed)
        
        # Weights will be initialized when we see the first FeatureSet
        self.emission_weights = None
        self.transition_weights = None
        
        # Initialize base transitions
        if base_transitions is not None:
            if base_transitions.shape != (num_states, num_states):
                raise ValueError(f"base_transitions must have shape ({num_states}, {num_states}), got {base_transitions.shape}")
            self.base_transitions = base_transitions.copy()
        else:
            self.base_transitions = np.zeros((num_states, num_states))
        
        logger.info(f"Initialized CRF with {num_states} states, optimizer={type(self.optimizer).__name__}, seed={seed}")
    
    def _initialize_weights(self, feature_set: FeatureSet):
        """Initialize weights based on feature dimensions from FeatureSet."""
        if self.emission_weights is None:
            emission_dim = feature_set.emission_dim
            transition_dim = feature_set.transition_dim
            
            self.emission_weights = self.rng.normal(0, 0.1, emission_dim)
            self.transition_weights = self.rng.normal(0, 0.1, (self.num_states, self.num_states, transition_dim))
            
            logger.info(f"Initialized weights: emission_dim={emission_dim}, transition_dim={transition_dim}")
    
    def decode(self, emissions: np.ndarray, features: FeatureSet) -> np.ndarray:
        """Decode the most likely state sequence using Viterbi algorithm."""
        self._initialize_weights(features)
        
        T, N = emissions.shape
        
        # Get feature potentials
        emission_potentials = features.get_emission_potentials(self.emission_weights)
        transition_potentials = features.get_transition_potentials(self.transition_weights)
        
        # Combine with base transitions - broadcast base_transitions to (T-1, N, N)
        log_transitions = self.base_transitions[None, :, :] + transition_potentials[:-1]
        log_emissions = np.log(emissions + 1e-10)
        
        return _viterbi_decode(log_emissions, log_transitions, emission_potentials)
    
    def compute_loss(self, sequence: TrainingSequence, features: FeatureSet) -> float:
        """Compute negative log-likelihood loss."""
        self._initialize_weights(features)
        
        T = len(sequence.true_path)
        
        # Get potentials
        emission_potentials = features.get_emission_potentials(self.emission_weights)
        transition_potentials = features.get_transition_potentials(self.transition_weights)
        
        # Combine with base transitions - broadcast base_transitions to (T-1, N, N)
        log_transitions = self.base_transitions[None, :, :] + transition_potentials[:-1]
        log_emissions = np.log(sequence.emissions + 1e-10)
        
        # Forward algorithm for partition function
        _, log_Z = _forward_algorithm(log_emissions, log_transitions, emission_potentials)
        
        # Path score
        path_score = _compute_path_score(sequence.true_path, log_emissions, log_transitions, emission_potentials)
        
        return (log_Z - path_score) / T
    
    def fit(self, sequences: list[TrainingSequence], feature_set_class, num_epochs: int = 100, 
            verbose: bool = True, num_workers: int | None = None, chunk_size: int | None = None, **feature_kwargs):
        """Train the CRF on sequences.
        
        Parameters
        ----------
        sequences : list[TrainingSequence]
            Training sequences
        feature_set_class : type
            FeatureSet class to use for feature computation
        num_epochs : int
            Number of training epochs (default: 100)
        verbose : bool
            Whether to print training progress (default: True)
        num_workers : int, optional
            Number of workers for parallel feature computation (default: None, uses min(cpu_count, len(sequences)))
        chunk_size : int, optional
            Maximum length for training sequences. If provided, longer sequences will be split into chunks (default: None)
        **feature_kwargs
            Additional keyword arguments passed to feature_set_class
        """
        logger.info(f"Starting CRF training for {num_epochs} epochs")
        
        # Chunk sequences if requested
        if chunk_size is not None:
            chunked_sequences = []
            for seq in sequences:
                if len(seq.true_path) <= chunk_size:
                    chunked_sequences.append(seq)
                else:
                    # Split into chunks
                    for start in range(0, len(seq.true_path), chunk_size):
                        end = min(start + chunk_size, len(seq.true_path))
                        chunked_sequences.append(TrainingSequence(
                            emissions=seq.emissions[start:end],
                            true_path=seq.true_path[start:end]
                        ))
            logger.info(f"Chunked {len(sequences)} sequences into {len(chunked_sequences)} chunks (max length: {chunk_size})")
            sequences = chunked_sequences
        
        # Pre-compute features with multiprocessing if requested
        if num_workers is None:
            num_workers = min(cpu_count(), len(sequences))
        
        if num_workers > 1 and len(sequences) > 1:
            logger.info(f"Computing features using {num_workers} workers")
            with Pool(num_workers) as pool:
                args = [(seq, feature_set_class, feature_kwargs) for seq in sequences]
                precomputed_features = pool.map(_compute_features_worker, args)
        else:
            logger.info("Computing features sequentially")
            precomputed_features = [feature_set_class(seq.emissions, **feature_kwargs) for seq in sequences]
        
        # Initialize weights from first feature set
        self._initialize_weights(precomputed_features[0])
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            # Initialize gradient accumulators
            emission_grad = np.zeros_like(self.emission_weights)
            transition_grad = np.zeros_like(self.transition_weights)
            base_transition_grad = np.zeros_like(self.base_transitions)
            
            # Collect predictions for metrics computation
            all_true_labels = []
            all_predictions = []
            
            # Process each sequence
            args_list = [(seq, features, self.emission_weights, self.transition_weights, self.base_transitions, self.num_states) 
                        for seq, features in zip(sequences, precomputed_features)]
            
            if num_workers > 1 and len(sequences) > 1:
                # Parallel gradient computation
                with Pool(num_workers) as pool:
                    results = pool.map(_compute_gradients_worker, args_list)
            else:
                # Sequential processing using same worker function
                results = [_compute_gradients_worker(args) for args in args_list]
            
            # Accumulate results and collect predictions for metrics
            total_tokens = 0
            for i, (emission_grad_seq, transition_grad_seq, base_transition_grad_seq, loss_seq) in enumerate(results):
                emission_grad += emission_grad_seq
                transition_grad += transition_grad_seq
                base_transition_grad += base_transition_grad_seq
                total_loss += loss_seq
                
                # Decode sequence for metrics computation
                seq = sequences[i]
                features = precomputed_features[i]
                predicted_path = self.decode(seq.emissions, features)
                
                all_true_labels.extend(seq.true_path)
                all_predictions.extend(predicted_path)
            
            # Count total tokens across all sequences
            for seq in sequences:
                total_tokens += len(seq.true_path)
            
            # Normalize gradients by total number of tokens to prevent scaling issues
            emission_grad /= total_tokens
            transition_grad /= total_tokens
            base_transition_grad /= total_tokens
            
            # Update parameters using optimizer
            params = {
                'emission_weights': self.emission_weights,
                'transition_weights': self.transition_weights,
                'base_transitions': self.base_transitions
            }
            gradients = {
                'emission_weights': emission_grad,
                'transition_weights': transition_grad,
                'base_transitions': base_transition_grad
            }
            
            updated_params = self.optimizer.update(params, gradients)
            self.emission_weights = updated_params['emission_weights']
            self.transition_weights = updated_params['transition_weights']
            self.base_transitions = updated_params['base_transitions']
            
            if verbose:
                avg_loss = total_loss / len(sequences)
                
                # Compute training metrics
                all_true_labels = np.array(all_true_labels)
                all_predictions = np.array(all_predictions)
                
                train_accuracy = accuracy_score(all_true_labels, all_predictions)
                train_f1_macro = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)
                train_f1_weighted = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
                
                # Compute gradient norm
                grad_norm = np.sqrt(
                    np.sum(emission_grad**2) + 
                    np.sum(transition_grad**2) + 
                    np.sum(base_transition_grad**2)
                )
                
                # Weight distribution summary
                emission_stats = f"emission: mean={np.mean(self.emission_weights):.4f}, std={np.std(self.emission_weights):.4f}, range=[{np.min(self.emission_weights):.4f}, {np.max(self.emission_weights):.4f}]"
                transition_stats = f"transition: mean={np.mean(self.transition_weights):.4f}, std={np.std(self.transition_weights):.4f}, range=[{np.min(self.transition_weights):.4f}, {np.max(self.transition_weights):.4f}]"
                base_stats = f"base: mean={np.mean(self.base_transitions):.4f}, std={np.std(self.base_transitions):.4f}, range=[{np.min(self.base_transitions):.4f}, {np.max(self.base_transitions):.4f}]"
                
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Grad Norm: {grad_norm:.4f}")
                logger.info(f"  Training Metrics - Accuracy: {train_accuracy:.4f}, F1 (macro): {train_f1_macro:.4f}, F1 (weighted): {train_f1_weighted:.4f}")
                logger.info(f"  Weights - {emission_stats}")
                logger.info(f"  Weights - {transition_stats}")
                logger.info(f"  Weights - {base_stats}")
        
        logger.info("Training completed")
    
    def print_weight_info(self, prefix: str = ""):
        """Print information about the model weights.
        
        Parameters
        ----------
        prefix : str
            Prefix to add to log messages (e.g., "Before training: " or "After training: ")
        """
        # Count total parameters
        emission_params = self.emission_weights.size
        transition_params = self.transition_weights.size
        base_transition_params = self.base_transitions.size
        total_params = emission_params + transition_params + base_transition_params
        
        logger.info(f"{prefix}Model parameter summary:")
        logger.info(f"{prefix}  Total parameters: {total_params:,}")
        logger.info(f"{prefix}  Emission parameters: {emission_params:,}")
        logger.info(f"{prefix}  Transition parameters: {transition_params:,}")
        logger.info(f"{prefix}  Base transition parameters: {base_transition_params:,}")
        
        # Emission weight details
        emission_stats = {
            'shape': self.emission_weights.shape,
            'mean': float(np.mean(self.emission_weights)),
            'std': float(np.std(self.emission_weights)),
            'norm': float(np.linalg.norm(self.emission_weights))
        }
        logger.info(f"{prefix}Emission weights: shape={emission_stats['shape']}, "
                   f"mean={emission_stats['mean']:.6f}, std={emission_stats['std']:.6f}, "
                   f"norm={emission_stats['norm']:.6f}")
        
        # Transition weight details
        transition_stats = {
            'shape': self.transition_weights.shape,
            'mean': float(np.mean(self.transition_weights)),
            'std': float(np.std(self.transition_weights)),
            'norm': float(np.linalg.norm(self.transition_weights))
        }
        logger.info(f"{prefix}Transition weights: shape={transition_stats['shape']}, "
                   f"mean={transition_stats['mean']:.6f}, std={transition_stats['std']:.6f}, "
                   f"norm={transition_stats['norm']:.6f}")
        
        # Base transition details
        base_stats = {
            'shape': self.base_transitions.shape,
            'mean': float(np.mean(self.base_transitions)),
            'std': float(np.std(self.base_transitions)),
            'norm': float(np.linalg.norm(self.base_transitions))
        }
        logger.info(f"{prefix}Base transitions: shape={base_stats['shape']}, "
                   f"mean={base_stats['mean']:.6f}, std={base_stats['std']:.6f}, "
                   f"norm={base_stats['norm']:.6f}")


def _compute_observed_transitions(true_path: np.ndarray, num_states: int) -> np.ndarray:
    """Compute observed transition counts."""
    observed = np.zeros((num_states, num_states))
    for t in range(1, len(true_path)):
        observed[true_path[t-1], true_path[t]] += 1.0
    return observed


# Numba-optimized functions
@njit
def _viterbi_decode(log_emissions: np.ndarray, log_transitions: np.ndarray, emission_potentials: np.ndarray) -> np.ndarray:
    """Viterbi algorithm (numba-optimized)."""
    T, N = log_emissions.shape
    
    # Initialize
    viterbi_prob = np.zeros((T, N))
    backpointer = np.zeros((T, N), dtype=np.int64)
    
    # Base case
    viterbi_prob[0] = log_emissions[0] + emission_potentials[0]
    
    # Recursive case
    for t in range(1, T):
        for j in range(N):
            probs = viterbi_prob[t-1] + log_transitions[t-1, :, j]
            backpointer[t, j] = np.argmax(probs)
            viterbi_prob[t, j] = probs[backpointer[t, j]] + log_emissions[t, j] + emission_potentials[t, j]
    
    # Backtrack
    path = np.zeros(T, dtype=np.int64)
    path[T-1] = np.argmax(viterbi_prob[T-1])
    for t in range(T-2, -1, -1):
        path[t] = backpointer[t+1, path[t+1]]
    
    return path


@njit
def _forward_algorithm(log_emissions: np.ndarray, log_transitions: np.ndarray, emission_potentials: np.ndarray) -> tuple[np.ndarray, float]:
    """Forward algorithm (numba-optimized)."""
    T, N = log_emissions.shape
    alpha = np.zeros((T, N))
    
    # Base case
    alpha[0] = log_emissions[0] + emission_potentials[0]
    
    # Recursive case
    for t in range(1, T):
        for j in range(N):
            max_val = -np.inf
            for i in range(N):
                val = alpha[t-1, i] + log_transitions[t-1, i, j]
                if val > max_val:
                    max_val = val
            
            sum_exp = 0.0
            for i in range(N):
                val = alpha[t-1, i] + log_transitions[t-1, i, j]
                sum_exp += np.exp(val - max_val)
            
            alpha[t, j] = max_val + np.log(sum_exp) + log_emissions[t, j] + emission_potentials[t, j]
    
    # Partition function
    max_alpha = np.max(alpha[T-1])
    log_Z = max_alpha + np.log(np.sum(np.exp(alpha[T-1] - max_alpha)))
    
    return alpha, log_Z


@njit
def _backward_algorithm(log_emissions: np.ndarray, log_transitions: np.ndarray, emission_potentials: np.ndarray) -> np.ndarray:
    """Backward algorithm (numba-optimized)."""
    T, N = log_emissions.shape
    beta = np.zeros((T, N))
    
    # Recursive case
    for t in range(T-2, -1, -1):
        for i in range(N):
            max_val = -np.inf
            for j in range(N):
                val = beta[t+1, j] + log_emissions[t+1, j] + emission_potentials[t+1, j] + log_transitions[t, i, j]
                if val > max_val:
                    max_val = val
            
            sum_exp = 0.0
            for j in range(N):
                val = beta[t+1, j] + log_emissions[t+1, j] + emission_potentials[t+1, j] + log_transitions[t, i, j]
                sum_exp += np.exp(val - max_val)
            
            beta[t, i] = max_val + np.log(sum_exp)
    
    return beta


@njit
def _compute_marginals(alpha: np.ndarray, beta: np.ndarray, log_Z: float) -> np.ndarray:
    """Compute marginals (numba-optimized)."""
    T, N = alpha.shape
    marginals = np.zeros((T, N))
    
    for t in range(T):
        for i in range(N):
            marginals[t, i] = np.exp(alpha[t, i] + beta[t, i] - log_Z)
    
    return marginals


@njit
def _compute_path_score(true_path: np.ndarray, log_emissions: np.ndarray, log_transitions: np.ndarray, emission_potentials: np.ndarray) -> float:
    """Compute path score (numba-optimized)."""
    T = len(true_path)
    score = 0.0
    
    # Initial state
    score += log_emissions[0, true_path[0]] + emission_potentials[0, true_path[0]]
    
    # Transitions and emissions
    for t in range(1, T):
        prev_state = true_path[t-1]
        curr_state = true_path[t]
        score += log_transitions[t-1, prev_state, curr_state]
        score += log_emissions[t, curr_state] + emission_potentials[t, curr_state]
    
    return score


@njit
def _compute_expected_transitions(alpha: np.ndarray, beta: np.ndarray, log_Z: float,
                                log_emissions: np.ndarray, log_transitions: np.ndarray, 
                                emission_potentials: np.ndarray) -> np.ndarray:
    """Compute expected transitions (numba-optimized)."""
    T, N = alpha.shape
    expected = np.zeros((N, N))
    
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                log_prob = (alpha[t-1, i] + log_transitions[t-1, i, j] + 
                           log_emissions[t, j] + emission_potentials[t, j] + 
                           beta[t, j] - log_Z)
                expected[i, j] += np.exp(log_prob)
    
    return expected


def _compute_gradients_worker(args: tuple) -> tuple:
    """Worker function for computing gradients in parallel."""
    seq, features, emission_weights, transition_weights, base_transitions, num_states = args
    
    # Get potentials
    emission_potentials = features.get_emission_potentials(emission_weights)
    transition_potentials = features.get_transition_potentials(transition_weights)
    
    # Combine with base transitions
    log_transitions = base_transitions[None, :, :] + transition_potentials[:-1]
    log_emissions = np.log(seq.emissions + 1e-10)
    
    # Forward-backward
    alpha, log_Z = _forward_algorithm(log_emissions, log_transitions, emission_potentials)
    beta = _backward_algorithm(log_emissions, log_transitions, emission_potentials)
    marginals = _compute_marginals(alpha, beta, log_Z)
    
    # Expected transitions
    expected_transitions = _compute_expected_transitions(alpha, beta, log_Z, log_emissions, log_transitions, emission_potentials)
    
    # Compute gradients
    emission_grad = features.compute_emission_gradients(marginals, seq.true_path)
    transition_grad = features.compute_transition_gradients(expected_transitions, seq.true_path)
    
    # Base transition gradients
    observed_transitions = _compute_observed_transitions(seq.true_path, num_states)
    base_transition_grad = expected_transitions - observed_transitions
    
    # Compute loss
    path_score = _compute_path_score(seq.true_path, log_emissions, log_transitions, emission_potentials)
    loss = (log_Z - path_score) / len(seq.true_path)
    
    return emission_grad, transition_grad, base_transition_grad, loss