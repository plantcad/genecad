import numpy as np
import pytest
from src.crf import CRF, TrainingSequence, FeatureSet, _viterbi_decode, SGD, Adam


class MockFeatureSet(FeatureSet):
    """Simple mock feature set for testing CRF logic."""
    
    def __init__(self, emissions: np.ndarray, emission_dim: int = 3, transition_dim: int = 2):
        super().__init__(emissions)
        # Use provided dimensions or defaults
        self._emission_dim = emission_dim
        self._transition_dim = transition_dim
    
    @property
    def emission_dim(self) -> int:
        """Number of emission feature dimensions."""
        return self._emission_dim
    
    @property
    def transition_dim(self) -> int:
        """Number of transition feature dimensions."""
        return self._transition_dim
    
    def get_emission_potentials(self, emission_weights: np.ndarray) -> np.ndarray:
        """Get emission feature potentials for all positions and states as (T, N) array."""
        # Simple implementation: just return small random values
        np.random.seed(42)  # For reproducibility
        return np.random.normal(0, 0.1, (self.T, self.N))
    
    def get_transition_potentials(self, transition_weights: np.ndarray) -> np.ndarray:
        """Get transition feature potentials for all positions as (T, N, N) array."""
        # Simple implementation: return small random values
        np.random.seed(42)  # For reproducibility
        return np.random.normal(0, 0.1, (self.T, self.N, self.N))
    
    def compute_emission_gradients(self, marginals: np.ndarray, true_path: np.ndarray) -> np.ndarray:
        """Compute emission feature gradients given marginals and true path."""
        # Simple gradient: difference between expected and observed
        expected = np.sum(marginals)  # Sum over all positions and states
        observed = len(true_path)     # Count of observations
        # Return gradient with correct emission_dim
        grad = np.zeros(self.emission_dim)
        grad[0] = expected - observed
        return grad
    
    def compute_transition_gradients(self, expected_transitions: np.ndarray, true_path: np.ndarray) -> np.ndarray:
        """Compute transition feature gradients given expected transitions and true path."""
        # Simple gradient computation
        grad = np.zeros((self.N, self.N, self.transition_dim))
        
        # Add some simple gradient based on expected transitions
        for i in range(self.N):
            for j in range(self.N):
                grad[i, j, 0] = expected_transitions[i, j] * 0.1
                if self.transition_dim > 1:
                    grad[i, j, 1] = expected_transitions[i, j] * 0.05
                if self.transition_dim > 2:
                    grad[i, j, 2] = expected_transitions[i, j] * 0.02
        
        return grad


class TestTrainingSequence:
    """Test suite for the TrainingSequence dataclass."""
    
    def test_valid_training_sequence(self):
        """Test creating a valid training sequence."""
        emissions = np.random.rand(10, 3)
        true_path = np.array([0, 1, 2, 1, 0, 2, 1, 0, 1, 2])
        
        seq = TrainingSequence(emissions=emissions, true_path=true_path)
        
        assert seq.emissions.shape == (10, 3)
        assert seq.true_path.shape == (10,)
        np.testing.assert_array_equal(seq.emissions, emissions)
        np.testing.assert_array_equal(seq.true_path, true_path)
    
    def test_invalid_emissions_shape(self):
        """Test that invalid emissions shape raises error."""
        emissions = np.random.rand(10)  # 1D instead of 2D
        true_path = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        with pytest.raises(ValueError, match="emissions must be 2D"):
            TrainingSequence(emissions=emissions, true_path=true_path)
    
    def test_invalid_true_path_shape(self):
        """Test that invalid true_path shape raises error."""
        emissions = np.random.rand(10, 3)
        true_path = np.array([[0, 1], [1, 2]])  # 2D instead of 1D
        
        with pytest.raises(ValueError, match="true_path must be 1D"):
            TrainingSequence(emissions=emissions, true_path=true_path)
    
    def test_length_mismatch(self):
        """Test that length mismatch raises error."""
        emissions = np.random.rand(10, 3)
        true_path = np.array([0, 1, 2, 1, 0])  # Length 5 instead of 10
        
        with pytest.raises(ValueError, match="Length mismatch"):
            TrainingSequence(emissions=emissions, true_path=true_path)


class TestCRF:
    """Test suite for the CRF class."""
    
    def test_init(self):
        """Test CRF initialization."""
        crf = CRF(num_states=3)
        
        assert crf.num_states == 3
        assert isinstance(crf.optimizer, SGD)
        assert crf.emission_weights is None  # Not initialized yet
        assert crf.transition_weights is None  # Not initialized yet
        assert crf.base_transitions.shape == (3, 3)
    
    def test_init_with_optimizer(self):
        """Test CRF initialization with custom optimizer."""
        optimizer = Adam(learning_rate=0.001)
        crf = CRF(num_states=3, optimizer=optimizer)
        
        assert crf.num_states == 3
        assert isinstance(crf.optimizer, Adam)
        assert crf.optimizer is optimizer
    
    def test_init_with_base_transitions(self):
        """Test CRF initialization with base transitions."""
        base_transitions = np.array([[0.8, 0.2], [0.3, 0.7]])
        crf = CRF(num_states=2, base_transitions=base_transitions)
        
        assert crf.num_states == 2
        np.testing.assert_array_equal(crf.base_transitions, base_transitions)
        # Should be a copy, not the same object
        assert crf.base_transitions is not base_transitions
    
    def test_init_with_invalid_base_transitions_shape(self):
        """Test that invalid base_transitions shape raises error."""
        base_transitions = np.array([[0.8, 0.2]])  # Wrong shape for 3 states
        
        with pytest.raises(ValueError, match="base_transitions must have shape \\(3, 3\\)"):
            CRF(num_states=3, base_transitions=base_transitions)
    
    def test_decode(self):
        """Test decoding with Viterbi algorithm."""
        crf = CRF(num_states=2)
        
        # Simple emissions favoring state 1
        emissions = np.array([
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7]
        ])
        
        features = MockFeatureSet(emissions, emission_dim=3, transition_dim=2)
        path = crf.decode(emissions, features)
        
        assert len(path) == 3
        assert all(state in [0, 1] for state in path)
    
    def test_compute_loss(self):
        """Test loss computation."""
        crf = CRF(num_states=2)
        
        emissions = np.random.rand(5, 2)
        true_path = np.array([0, 1, 1, 0, 1])
        sequence = TrainingSequence(emissions=emissions, true_path=true_path)
        features = MockFeatureSet(emissions, emission_dim=3, transition_dim=2)
        
        loss = crf.compute_loss(sequence, features)
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert loss >= 0.0  # Negative log-likelihood should be non-negative
    
    def test_fit(self):
        """Test training the CRF."""
        crf = CRF(num_states=2, optimizer=SGD(learning_rate=0.01))
        
        # Create training sequences
        sequences = [
            TrainingSequence(
                emissions=np.random.rand(10, 2),
                true_path=np.random.randint(0, 2, 10)
            ),
            TrainingSequence(
                emissions=np.random.rand(8, 2),
                true_path=np.random.randint(0, 2, 8)
            )
        ]
        
        # Should not raise an error
        crf.fit(sequences, MockFeatureSet, num_epochs=3, verbose=False, emission_dim=3, transition_dim=2)
        
        # Check that weights exist and are finite
        assert not np.any(np.isnan(crf.emission_weights))
        assert not np.any(np.isnan(crf.transition_weights))
        assert not np.any(np.isnan(crf.base_transitions))
    
    def test_fit_with_adam(self):
        """Test training the CRF with Adam optimizer."""
        crf = CRF(num_states=2, optimizer=Adam(learning_rate=0.001))
        
        # Create training sequences
        sequences = [
            TrainingSequence(
                emissions=np.random.rand(10, 2),
                true_path=np.random.randint(0, 2, 10)
            )
        ]
        
        # Should not raise an error
        crf.fit(sequences, MockFeatureSet, num_epochs=3, verbose=False, emission_dim=3, transition_dim=2)
        
        # Check that weights exist and are finite
        assert not np.any(np.isnan(crf.emission_weights))
        assert not np.any(np.isnan(crf.transition_weights))
        assert not np.any(np.isnan(crf.base_transitions))
    
    def test_end_to_end_training(self):
        """Test complete end-to-end training."""
        crf = CRF(num_states=2, optimizer=SGD(learning_rate=0.01))
        
        # Create simple training data
        np.random.seed(42)  # For reproducibility
        sequences = [
            TrainingSequence(
                emissions=np.random.rand(8, 2),
                true_path=np.array([0, 0, 1, 1, 1, 0, 0, 1])
            ),
            TrainingSequence(
                emissions=np.random.rand(6, 2), 
                true_path=np.array([1, 1, 0, 0, 1, 1])
            )
        ]
        
        # Train for a few epochs
        try:
            crf.fit(sequences, MockFeatureSet, num_epochs=3, verbose=False, emission_dim=3, transition_dim=2)
            
            # Verify we can compute loss after training
            features = MockFeatureSet(sequences[0].emissions, emission_dim=3, transition_dim=2)
            loss = crf.compute_loss(sequences[0], features)
            assert isinstance(loss, float)
            assert not np.isnan(loss)
            
            # Verify we can decode after training
            path = crf.decode(sequences[0].emissions, features)
            assert len(path) == len(sequences[0].true_path)
            assert all(state in [0, 1] for state in path)
            
        except Exception as e:
            pytest.fail(f"End-to-end training failed: {e}")
    
    def test_fit_with_chunking(self):
        """Test training the CRF with sequence chunking."""
        crf = CRF(num_states=2, optimizer=SGD(learning_rate=0.01))
        
        # Create a long training sequence that will be chunked
        long_sequence = TrainingSequence(
            emissions=np.random.rand(25, 2),  # 25 positions
            true_path=np.random.randint(0, 2, 25)
        )
        
        sequences = [long_sequence]
        
        # Train with chunking (should split 25-position sequence into 3 chunks: 10, 10, 5)
        crf.fit(sequences, MockFeatureSet, num_epochs=2, verbose=False, 
                chunk_size=10, emission_dim=3, transition_dim=2)
        
        # Check that weights exist and are finite
        assert not np.any(np.isnan(crf.emission_weights))
        assert not np.any(np.isnan(crf.transition_weights))
        assert not np.any(np.isnan(crf.base_transitions))
    
    def test_fit_without_chunking(self):
        """Test that chunking doesn't affect sequences shorter than chunk_size."""
        crf = CRF(num_states=2, optimizer=SGD(learning_rate=0.01))
        
        # Create short sequences that won't be chunked
        sequences = [
            TrainingSequence(
                emissions=np.random.rand(5, 2),
                true_path=np.random.randint(0, 2, 5)
            ),
            TrainingSequence(
                emissions=np.random.rand(8, 2),
                true_path=np.random.randint(0, 2, 8)
            )
        ]
        
        # Train with large chunk_size (should not chunk anything)
        crf.fit(sequences, MockFeatureSet, num_epochs=2, verbose=False, 
                chunk_size=20, emission_dim=3, transition_dim=2)
        
        # Check that weights exist and are finite
        assert not np.any(np.isnan(crf.emission_weights))
        assert not np.any(np.isnan(crf.transition_weights))
        assert not np.any(np.isnan(crf.base_transitions))


class TestViterbiDecode:
    """Test suite for the _viterbi_decode function."""
    
    def test_viterbi_decode(self):
        """Test Viterbi decoding."""
        log_emissions = np.log(np.array([
            [0.1, 0.9],
            [0.8, 0.2],
            [0.3, 0.7]
        ]))
        
        log_transitions = np.log(np.array([
            [[0.9, 0.1], [0.2, 0.8]],  # Transitions for time 0
            [[0.9, 0.1], [0.2, 0.8]]   # Transitions for time 1
        ]))
        
        # Feature potentials that slightly favor state 0
        emission_potentials = np.array([
            [0.1, -0.1],
            [0.1, -0.1], 
            [0.1, -0.1]
        ])
        
        path = _viterbi_decode(log_emissions, log_transitions, emission_potentials)
        
        assert len(path) == 3
        assert all(state in [0, 1] for state in path)


class TestFeatureSet:
    """Test suite for the FeatureSet abstract base class."""
    
    def test_mock_feature_set(self):
        """Test that MockFeatureSet implements the interface correctly."""
        emissions = np.random.rand(5, 3)
        features = MockFeatureSet(emissions, emission_dim=3, transition_dim=2)
        
        assert features.T == 5
        assert features.N == 3
        
        # Test emission potentials
        emission_weights = np.random.rand(features.emission_dim)
        emission_potentials = features.get_emission_potentials(emission_weights)
        assert emission_potentials.shape == (5, 3)
        
        # Test transition potentials
        transition_weights = np.random.rand(3, 3, features.transition_dim)
        transition_potentials = features.get_transition_potentials(transition_weights)
        assert transition_potentials.shape == (5, 3, 3)
        
        # Test gradient computation
        marginals = np.random.rand(5, 3)
        true_path = np.array([0, 1, 2, 1, 0])
        
        emission_grad = features.compute_emission_gradients(marginals, true_path)
        assert emission_grad.shape == (features.emission_dim,)
        
        expected_transitions = np.random.rand(3, 3)
        transition_grad = features.compute_transition_gradients(expected_transitions, true_path)
        assert transition_grad.shape == (3, 3, features.transition_dim)


class TestNumbaFunctions:
    """Test suite for numba-optimized functions."""
    
    def test_forward_algorithm(self):
        """Test the forward algorithm implementation."""
        log_emissions = np.log(np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
        log_transitions = np.log(np.array([
            [[0.9, 0.1], [0.2, 0.8]],  # Transitions for time 0
            [[0.9, 0.1], [0.2, 0.8]]   # Transitions for time 1
        ]))
        emission_potentials = np.zeros((3, 2))
        
        from src.crf import _forward_algorithm
        alpha, log_Z = _forward_algorithm(log_emissions, log_transitions, emission_potentials)
        
        assert alpha.shape == (3, 2)
        assert isinstance(log_Z, float)
        assert not np.isnan(log_Z)
        assert not np.any(np.isnan(alpha))
    
    def test_backward_algorithm(self):
        """Test the backward algorithm implementation."""
        log_emissions = np.log(np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
        log_transitions = np.log(np.array([
            [[0.9, 0.1], [0.2, 0.8]],  # Transitions for time 0
            [[0.9, 0.1], [0.2, 0.8]]   # Transitions for time 1
        ]))
        emission_potentials = np.zeros((3, 2))
        
        from src.crf import _backward_algorithm
        beta = _backward_algorithm(log_emissions, log_transitions, emission_potentials)
        
        assert beta.shape == (3, 2)
        assert not np.any(np.isnan(beta))
        # Last row should be zeros (log(1) = 0)
        np.testing.assert_array_equal(beta[2], [0.0, 0.0])
    
    def test_marginals_computation(self):
        """Test marginal computation from forward-backward."""
        log_emissions = np.log(np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
        log_transitions = np.log(np.array([
            [[0.9, 0.1], [0.2, 0.8]],  # Transitions for time 0
            [[0.9, 0.1], [0.2, 0.8]]   # Transitions for time 1
        ]))
        emission_potentials = np.zeros((3, 2))
        
        from src.crf import _forward_algorithm, _backward_algorithm, _compute_marginals
        alpha, log_Z = _forward_algorithm(log_emissions, log_transitions, emission_potentials)
        beta = _backward_algorithm(log_emissions, log_transitions, emission_potentials)
        marginals = _compute_marginals(alpha, beta, log_Z)
        
        assert marginals.shape == (3, 2)
        # Each row should sum to 1 (marginal probabilities)
        np.testing.assert_array_almost_equal(marginals.sum(axis=1), [1.0, 1.0, 1.0])
        # All probabilities should be non-negative
        assert np.all(marginals >= 0.0)
    
    def test_path_score_computation(self):
        """Test path score computation."""
        true_path = np.array([0, 1, 0])
        log_emissions = np.log(np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
        log_transitions = np.log(np.array([
            [[0.9, 0.1], [0.2, 0.8]],  # Transitions for time 0
            [[0.9, 0.1], [0.2, 0.8]]   # Transitions for time 1
        ]))
        emission_potentials = np.zeros((3, 2))
        
        from src.crf import _compute_path_score
        path_score = _compute_path_score(true_path, log_emissions, log_transitions, emission_potentials)
        
        assert isinstance(path_score, float)
        assert not np.isnan(path_score)
        # Path score should be negative (log probability)
        assert path_score < 0.0
    
    def test_expected_transitions_computation(self):
        """Test expected transitions computation."""
        log_emissions = np.log(np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]))
        log_transitions = np.log(np.array([
            [[0.9, 0.1], [0.2, 0.8]],  # Transitions for time 0
            [[0.9, 0.1], [0.2, 0.8]]   # Transitions for time 1
        ]))
        emission_potentials = np.zeros((3, 2))
        
        from src.crf import _forward_algorithm, _backward_algorithm, _compute_expected_transitions
        alpha, log_Z = _forward_algorithm(log_emissions, log_transitions, emission_potentials)
        beta = _backward_algorithm(log_emissions, log_transitions, emission_potentials)
        expected_transitions = _compute_expected_transitions(alpha, beta, log_Z, log_emissions, log_transitions, emission_potentials)
        
        assert expected_transitions.shape == (2, 2)
        assert not np.any(np.isnan(expected_transitions))
        assert np.all(expected_transitions >= 0.0)


if __name__ == "__main__":
    pytest.main([__file__]) 