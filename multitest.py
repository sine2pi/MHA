

import pytest
import torch
from hypothesis import given, settings, strategies as st
from rasa import Multihead, UniversalMask


# Import the Multihead class from the module
device = torch.device(device="cuda:0")
dtype = torch.float32

class MockUniversalMask:
    @staticmethod
    def create(batch_size, seq_len, seq_len_kv=None, num_heads=1, mask_type="combined", is_causal=False, 
               padding_mask=None, device=device, dtype=dtype):
        """Mock create method for UniversalMask"""
        # Use seq_len_kv if provided, otherwise use seq_len (for self-attention)
        seq_len_k = seq_len_kv if seq_len_kv is not None else seq_len
        
        # Create a mask with proper dimensions for cross-attention
        mask = torch.zeros(batch_size, num_heads, seq_len, seq_len_k, device=device, dtype=dtype)
        
        if is_causal and seq_len_kv is None:
            # Only apply causal mask for self-attention
            causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask + causal.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1) * -1e9
            
        return mask
    
# Patch the UniversalMask.create for testing
@pytest.fixture(autouse=True)
def mock_universal_mask(monkeypatch):
    monkeypatch.setattr(UniversalMask, "create", MockUniversalMask.create)

@pytest.fixture
def small_multihead():
    """Create a small Multihead instance for testing"""
    dims = 64
    head = 8
    return Multihead(dims=dims, head=head, layer_idx=0, decoder=False)

@pytest.fixture
def medium_multihead():
    """Create a medium-sized Multihead instance for testing"""
    dims = 128
    head = 8
    return Multihead(dims=dims, head=head, layer_idx=1, decoder=True, dropout=0.1)

def test_initialization():
    """Test that Multihead initializes correctly with different parameters"""
    # Basic initialization
    mh = Multihead(dims=64, head=8, layer_idx=0, decoder=False)
    assert mh.dims == 64
    assert mh.head == 8
    assert mh.head_dim == 8  # 64/8
    assert mh.scale == 0.3535533905932738  # 1/sqrt(8)
    
    # Different parameters
    mh = Multihead(dims=128, head=4, layer_idx=1, decoder=True, dropout=0.1, bias=True)
    assert mh.dims == 128
    assert mh.head == 4
    assert mh.head_dim == 32  # 128/4
    assert mh.decoder is True
    assert mh.dropout == 0.1

def test_dimensions_divisible_assertion():
    """Test that an assertion error is raised if dims is not divisible by head"""
    with pytest.raises(AssertionError):
        Multihead(dims=65, head=8, layer_idx=0, decoder=False)

def test_forward_shape(small_multihead):
    """Test that forward returns correct output shape"""
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, small_multihead.dims, device=device, dtype=dtype)
    output, _ = small_multihead(x)
    assert output.shape == (batch_size, seq_len, small_multihead.dims)

def test_forward_cross_attention(small_multihead):
    """Test forward pass with cross attention"""
    batch_size = 2
    seq_len_x = 10
    seq_len_xa = 12
    x = torch.randn(batch_size, seq_len_x, small_multihead.dims, device=device, dtype=dtype)
    xa = torch.randn(batch_size, seq_len_xa, small_multihead.dims, device=device, dtype=dtype)
    output, _ = small_multihead(x, xa=xa)
    assert output.shape == (batch_size, seq_len_x, small_multihead.dims)

def test_attention_modes(small_multihead):
    """Test different attention modes"""
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, small_multihead.dims, device=device, dtype=dtype)
    
    # Store original values
    original_cosa = Multihead.cosa
    original_sdpa = Multihead.sdpa
    original_combine = Multihead.combine
    original_dynamic_blend = Multihead.dynamic_blend
    
    try:
        # Standard attention
        Multihead.cosa = False
        Multihead.sdpa = False
        Multihead.combine = False
        Multihead.dynamic_blend = False
        output_standard, _ = small_multihead(x)
        
        # Cosine attention
        Multihead.cosa = True
        output_cosa, _ = small_multihead(x)
        
        # SDPA attention
        Multihead.cosa = False
        Multihead.sdpa = True
        output_sdpa, _ = small_multihead(x)
        
        # Combined attention
        Multihead.sdpa = False
        Multihead.combine = True
        output_combine, _ = small_multihead(x)
        
        # Check shapes
        assert output_standard.shape == (batch_size, seq_len, small_multihead.dims)
        assert output_cosa.shape == (batch_size, seq_len, small_multihead.dims)
        assert output_sdpa.shape == (batch_size, seq_len, small_multihead.dims)
        assert output_combine.shape == (batch_size, seq_len, small_multihead.dims)
    finally:
        # Reset class variables
        Multihead.cosa = original_cosa
        Multihead.sdpa = original_sdpa
        Multihead.combine = original_combine
        Multihead.dynamic_blend = original_dynamic_blend

def test_dynamic_blend_mode():
    """Test dynamic blend attention mode"""
    dims = 64
    head = 8
    batch_size = 2
    seq_len = 10
    
    original_dynamic_blend = Multihead.dynamic_blend
    try:
        Multihead.dynamic_blend = True
        mh = Multihead(dims=dims, head=head, layer_idx=0, decoder=False)
        
        # Check that blend_weights was initialized
        assert hasattr(mh, 'blend_weights')
        
        x = torch.randn(batch_size, seq_len, dims, device=device, dtype=dtype)
        output, _ = mh(x)
        assert output.shape == (batch_size, seq_len, dims)
    finally:
        Multihead.dynamic_blend = original_dynamic_blend

def test_magnitude_scaling(small_multihead):
    """Test magnitude scaling in cosine attention"""
    batch_size = 2
    seq_len = 10
    head_dim = small_multihead.head_dim
    head = small_multihead.head
    
    q = torch.randn(batch_size, head, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, head, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, head, seq_len, head_dim, device=device, dtype=dtype)
    
    original_magnitude = Multihead.magnitude
    original_cosa = Multihead.cosa
    
    try:
        # Enable cosine attention
        Multihead.cosa = True
        
        # Without magnitude scaling
        Multihead.magnitude = False
        output1 = small_multihead.cos_attention(q, k, v)
        
        # With magnitude scaling
        Multihead.magnitude = True
        output2 = small_multihead.cos_attention(q, k, v)
        
        assert output1.shape == output2.shape == (batch_size, head, seq_len, head_dim)
        
        # Outputs should be different with magnitude scaling enabled
        # Note: This might fail in rare cases due to random initialization
        assert not torch.allclose(output1, output2, rtol=1e-3, atol=1e-3)
    finally:
        Multihead.magnitude = original_magnitude
        Multihead.cosa = original_cosa

def test_kv_cache(small_multihead):
    """Test using kv_cache"""
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, small_multihead.dims, device=device, dtype=dtype)
    kv_cache = {}
    
    # First call should populate the cache
    output1, _ = small_multihead(x, kv_cache=kv_cache)
    
    # Check that cache was populated
    assert small_multihead.k in kv_cache
    assert small_multihead.v in kv_cache
    
    # Another call with same inputs should give same results
    output2, _ = small_multihead(x, kv_cache=kv_cache)
    assert torch.allclose(output1, output2, rtol=1e-4, atol=1e-4)

@given(
    dims=st.integers(min_value=16, max_value=128).filter(lambda x: x % 8 == 0),
    head=st.integers(min_value=1, max_value=8),
    batch_size=st.integers(min_value=1, max_value=4),
    seq_len=st.integers(min_value=2, max_value=16)
)
@settings(max_examples=10, deadline=None)
def test_hypothesis_forward(dims, head, batch_size, seq_len):
    """Test forward pass with various parameters using hypothesis"""
    # Skip if dims not divisible by head
    if dims % head != 0:
        return
    
    mh = Multihead(dims=dims, head=head, layer_idx=0, decoder=False)
    x = torch.randn(batch_size, seq_len, dims, device=device, dtype=dtype)
    output, _ = mh(x)
    assert output.shape == (batch_size, seq_len, dims)

def test_cos_attention_mask(small_multihead):
    """Test cosine attention with mask"""
    batch_size = 2
    seq_len = 10
    head_dim = small_multihead.head_dim
    head = small_multihead.head
    
    q = torch.randn(batch_size, head, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, head, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, head, seq_len, head_dim, device=device, dtype=dtype)
    
    # Create a mask with -inf in positions we want to mask
    mask = torch.zeros(batch_size, head, seq_len, seq_len, device=device, dtype=dtype)
    mask[:, :, 0, 5:] = float('-inf')  # Mask some positions
    
    output = small_multihead.cos_attention(q, k, v, mask)
    assert output.shape == (batch_size, head, seq_len, head_dim)

def test_decoder_mode(medium_multihead):
    """Test with decoder=True which creates causal masks"""
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, medium_multihead.dims, device=device, dtype=dtype)
    
    # With decoder=False
    medium_multihead.decoder = False
    output1, _ = medium_multihead(x, decoder=False)
    
    # With decoder=True
    medium_multihead.decoder = True
    output2, _ = medium_multihead(x, decoder=True)
    
    # Shapes should be the same, but values different due to causal mask
    assert output1.shape == output2.shape
    assert not torch.allclose(output1, output2, rtol=1e-2, atol=1e-2)

def test_dropout_behavior(medium_multihead):
    """Test dropout behavior in training vs eval modes"""
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, medium_multihead.dims, device=device, dtype=dtype)
    
    # In eval mode, outputs should be deterministic
    medium_multihead.eval()
    output1 = medium_multihead(x)[0]
    output2 = medium_multihead(x)[0]
    assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)

def test_attention_metrics(small_multihead):
    """Test attention quality metrics"""
    batch_size = 2
    seq_len = 12
    x = torch.randn(batch_size, seq_len, small_multihead.dims, device=device, dtype=dtype)
    
    # Extract attention weights for analysis
    # We need to modify _attention to return weights for this test
    original__attention = small_multihead._attention
    
    # Function to capture attention weights
    def wrapped_attention(*args, **kwargs):
        output, qk = original__attention(*args, **kwargs)
        weights = torch.softmax(qk, dim=-1) if qk is not None else None
        return output, (qk, weights)
    
    try:
        # Override _attention method temporarily
        small_multihead._attention = wrapped_attention
        
        # Run forward pass to get attention weights
        _, attention_data = small_multihead(x)
        _, weights = attention_data
        
        if weights is not None:
            # Compute entropy of attention distributions
            # Lower entropy means more focused attention
            log_weights = torch.log(weights + 1e-10)
            entropy = -torch.sum(weights * log_weights, dim=-1)
            avg_entropy = entropy.mean().item()
            
            # Check for attention coverage (should attend to most tokens)
            coverage = (weights > 0.01).float().mean().item()
            
            # Print metrics
            print(f"Attention Entropy: {avg_entropy:.4f}")
            print(f"Token Coverage: {coverage:.4f}")
            
            # We could assert some properties, but these are model-dependent
            # assert avg_entropy < 2.5, "Attention too diffuse"
            # assert coverage > 0.3, "Attention too sparse"
    finally:
        # Restore original method
        small_multihead._attention = original__attention

def test_gradient_flow(small_multihead):
    """Test that gradients flow properly through the attention mechanism"""
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, small_multihead.dims, 
                   device=device, dtype=dtype, requires_grad=True)
    
    # Track initial parameter values
    initial_q_weight = small_multihead.q.weight.clone().detach()
    
    # Forward pass
    output, _ = small_multihead(x)
    
    # Create a dummy loss and backpropagate
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist and are non-zero
    assert small_multihead.q.weight.grad is not None, "Query projection has no gradient"
    assert small_multihead.k.weight.grad is not None, "Key projection has no gradient"
    assert small_multihead.v.weight.grad is not None, "Value projection has no gradient"
    assert small_multihead.o.weight.grad is not None, "Output projection has no gradient"
    
    # Check that at least some gradients are non-zero
    assert torch.any(small_multihead.q.weight.grad != 0), "All query gradients are zero"
    
    # Optional: Apply the gradients and check the weights changed
    learning_rate = 0.01
    with torch.no_grad():
        small_multihead.q.weight -= learning_rate * small_multihead.q.weight.grad
    
    weight_changed = not torch.allclose(initial_q_weight, small_multihead.q.weight)
    assert weight_changed, "Weights did not update after gradient step"

def test_attention_self_consistency(small_multihead):
    """Test consistency of attention with identical key/value"""
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, small_multihead.dims, device=device, dtype=dtype)
    
    # For identical inputs, the attention should approximate identity mapping
    # (not exact due to projections, but similar)
    output, _ = small_multihead(x)
    
    # Compute normalized similarity between input and output
    x_norm = torch.nn.functional.normalize(x, dim=-1)
    output_norm = torch.nn.functional.normalize(output, dim=-1)
    similarity = torch.bmm(x_norm, output_norm.transpose(1, 2))
    
    # Diagonal elements should be larger (inputs influence their corresponding outputs)
    diag_sim = torch.diagonal(similarity, dim1=1, dim2=2).mean()
    offdiag_sim = (similarity.sum() - torch.diagonal(similarity, dim1=1, dim2=2).sum()) / (batch_size * seq_len * (seq_len-1))
    
    print(f"Diagonal similarity: {diag_sim:.4f}")
    print(f"Off-diagonal similarity: {offdiag_sim:.4f}")
    
    # Diagonal elements should have higher similarity on average
    assert diag_sim > offdiag_sim, "Self-attention not prioritizing self-connections"

def test_numerical_stability(small_multihead):
    """Test attention with extreme values to verify numerical stability"""
    batch_size = 2
    seq_len = 10
    
    # Test with very large values
    x_large = torch.ones(batch_size, seq_len, small_multihead.dims, 
                        device=device, dtype=dtype) * 1000
    output_large, _ = small_multihead(x_large)
    assert not torch.isnan(output_large).any(), "NaN values in output with large inputs"
    assert not torch.isinf(output_large).any(), "Infinite values in output with large inputs"
    
    # Test with very small values
    x_small = torch.ones(batch_size, seq_len, small_multihead.dims, 
                        device=device, dtype=dtype) * 1e-6
    output_small, _ = small_multihead(x_small)
    assert not torch.isnan(output_small).any(), "NaN values in output with small inputs"
    
    # Test with mixed values
    x_mixed = torch.randn(batch_size, seq_len, small_multihead.dims, 
                         device=device, dtype=dtype)
    x_mixed[:, 0, :] = 1000  # First token has large values
    x_mixed[:, 1, :] = 1e-6  # Second token has small values
    output_mixed, _ = small_multihead(x_mixed)
    assert not torch.isnan(output_mixed).any(), "NaN values in output with mixed inputs"

def test_performance_benchmark():
    """Benchmark different attention implementations"""
    import time
    
    batch_size = 8
    seq_len = 512
    dims = 768
    heads = 12
    
    # Setup model
    model = Multihead(dims=dims, head=heads, layer_idx=0, decoder=False)
    model.to(device)
    
    # Create inputs
    x = torch.randn(batch_size, seq_len, dims, device=device, dtype=dtype)
    
    # Warm up
    for _ in range(5):
        _ = model(x)
    
    # Standard attention
    model.cosa = False
    model.sdpa = False
    model.combine = False
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    std_time = time.time() - start
    
    # SDPA
    model.cosa = False
    model.sdpa = True
    model.combine = False
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    sdpa_time = time.time() - start
    
    print(f"Standard attention: {std_time:.4f}s")
    print(f"SDPA attention: {sdpa_time:.4f}s")
    print(f"Speedup: {std_time/sdpa_time:.2f}x")


    # In training mode, outputs might differ due to dropout
    # This is a probabilistic test, so we don't actually assert anything
    medium_multihead.train()
    _ = medium_multihead(x)[0]
    _ = medium_multihead(x)[0]


