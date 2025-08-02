import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from separated_attention import QueryModule, KeyModule, ValueModule, AttentionCombiner

class KeyValueCache:
    """Helper class for managing key-value caches with separate K and V modules."""
    
    def __init__(self):
        self.key_cache = {}
        self.value_cache = {}
    
    def update_key(self, module_id, tensor):
        """Update the key cache for a specific module."""
        self.key_cache[module_id] = tensor
    
    def update_value(self, module_id, tensor):
        """Update the value cache for a specific module."""
        self.value_cache[module_id] = tensor
    
    def get_key(self, module_id):
        """Get cached key tensor for a module."""
        return self.key_cache.get(module_id)
    
    def get_value(self, module_id):
        """Get cached value tensor for a module."""
        return self.value_cache.get(module_id)


class SharedKeyModule(nn.Module):
    """Key module that can be shared across multiple attention heads."""
    
    def __init__(self, dims: int, heads: int, shared_keys: int = 1):
        """
        Args:
            dims: Input dimension
            heads: Number of attention heads that will use these keys
            shared_keys: Number of key heads to use (typically smaller than heads)
        """
        super().__init__()
        
        assert dims % heads == 0, f"dims must be divisible by heads"
        assert heads % shared_keys == 0, f"heads must be divisible by shared_keys"
        
        self.dims = dims
        self.heads = heads
        self.shared_keys = shared_keys
        self.head_dim = dims // heads
        self.key_dim = dims // shared_keys
        self.scale = self.head_dim ** -0.25
        self.repeat_factor = heads // shared_keys
        
        # Smaller projection for shared keys
        self.key = nn.Linear(dims, shared_keys * self.key_dim, bias=False)
        nn.init.normal_(self.key.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to shared key representation
        
        Args:
            x: Input tensor [batch, seq_len, dims]
            
        Returns:
            Key tensor repeated to match heads [batch, heads, seq_len, head_dim]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Project to smaller dimension
        k = self.key(x)
        
        # Reshape for attention with fewer heads
        k = k.view(batch_size, seq_len, self.shared_keys, self.key_dim // self.repeat_factor)
        k = k.permute(0, 2, 1, 3)
        
        # Repeat to match original number of heads
        k = k.repeat_interleave(self.repeat_factor, dim=1)
        
        # Ensure final shape matches expected size
        k = k * self.scale
        
        return k
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from separated_attention import QueryModule, KeyModule, ValueModule, AttentionCombiner

class AdaptiveUpdateAttention(nn.Module):
    """Attention implementation with content-dependent update frequencies."""
    
    def __init__(self, dims: int, heads: int):
        super().__init__()
        
        self.query_module = QueryModule(dims, heads)
        self.key_module = KeyModule(dims, heads)
        self.value_module = ValueModule(dims, heads)
        self.combiner = AttentionCombiner(dims, heads)
        
        # Add update predictors to decide when to update K and V
        self.key_update_predictor = nn.Sequential(
            nn.Linear(dims, dims // 4),
            nn.ReLU(),
            nn.Linear(dims // 4, 1),
            nn.Sigmoid()
        )
        
        self.value_update_predictor = nn.Sequential(
            nn.Linear(dims, dims // 4),
            nn.ReLU(),
            nn.Linear(dims // 4, 1),
            nn.Sigmoid()
        )
        
        self.update_threshold = 0.5
    
    def should_update_key(self, x: torch.Tensor) -> torch.Tensor:
        """Predict whether the key should be updated based on content."""
        # Average over sequence dimension 
        avg_rep = x.mean(dim=1)
        return self.key_update_predictor(avg_rep) > self.update_threshold
    
    def should_update_value(self, x: torch.Tensor) -> torch.Tensor:
        """Predict whether the value should be updated based on content."""
        # Average over sequence dimension
        avg_rep = x.mean(dim=1)
        return self.value_update_predictor(avg_rep) > self.update_threshold
    
        
            
    def forward(
        self, 
        x: torch.Tensor,
        xa: Optional[torch.Tensor] = None,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with adaptive updates for keys and values
        
        Args:
            x: Input tensor
            xa: Cross-attention input (optional)
            key_cache: Previously cached key (optional)
            value_cache: Previously cached value (optional)
            
        Returns:
            Tuple of (output tensor, cache updates)
        """
        # Always compute query from current input
        q = self.query_module(x)
        
        # Content from cross-attention or self-attention
        kv_input = xa if xa is not None else x
        
        # Determine whether to update keys and values
        batch_size = kv_input.shape[0]
        device = kv_input.device
        
        # Handle key updates
        if key_cache is None:
            update_k = torch.ones(batch_size, dtype=torch.bool, device=device)
            k = self.key_module(kv_input)
        else:
            update_k = self.should_update_key(kv_input)
            if update_k.any():
                new_k = self.key_module(kv_input)
                # Create update mask with proper dimensions for broadcasting
                update_mask = update_k.view(-1, 1, 1, 1).expand_as(key_cache)
                k = torch.where(update_mask, new_k, key_cache)
            else:
                k = key_cache
        
        # Handle value updates
        if value_cache is None:
            update_v = torch.ones(batch_size, dtype=torch.bool, device=device)
            v = self.value_module(kv_input)
        else:
            update_v = self.should_update_value(kv_input)
            if update_v.any():
                new_v = self.value_module(kv_input)
                # Create update mask with proper dimensions for broadcasting
                update_mask = update_v.view(-1, 1, 1, 1).expand_as(value_cache)
                v = torch.where(update_mask, new_v, value_cache)
            else:
                v = value_cache
        
        # Compute attention
        output = self.combiner(q, k, v)
        
        # Return output and updated caches
        cache_updates = {
            "key_cache": k,
            "value_cache": v,
            "key_updated": update_k,
            "value_updated": update_v,
        }
        
        return output, cache_updates

def demonstrate_advanced_patterns():
    # Example usage
    batch_size, seq_len, dims = 2, 4, 384
    heads = 2
    x = torch.randn(batch_size, seq_len, dims)
    
    print("\nTesting AdaptiveUpdateAttention:")
    adaptive_attn = AdaptiveUpdateAttention(dims, heads)
    output, cache_updates = adaptive_attn(x)
    print(f"Output shape: {output.shape}")
    print(f"Key updated: {cache_updates['key_updated']}")
    print(f"Value updated: {cache_updates['value_updated']}")


if __name__ == "__main__":
    demonstrate_advanced_patterns()

class MultiLayerSeparatedAttention(nn.Module):
    """Stack multiple attention layers with separate Q, K, V modules and flexible update patterns."""
    
    def __init__(self, dims: int, heads: int, num_layers: int):
        super().__init__()
        
        self.dims = dims
        self.heads = heads
        self.num_layers = num_layers
        
        # Create separate Q, K, V modules for each layer
        self.query_modules = nn.ModuleList([
            QueryModule(dims, heads) for _ in range(num_layers)
        ])
        
        self.key_modules = nn.ModuleList([
            KeyModule(dims, heads) for _ in range(num_layers)
        ])
        
        self.value_modules = nn.ModuleList([
            ValueModule(dims, heads) for _ in range(num_layers)
        ])
        
        self.combiners = nn.ModuleList([
            AttentionCombiner(dims, heads) for _ in range(num_layers)
        ])
        
        # Layer norms for each component
        self.q_norms = nn.ModuleList([nn.LayerNorm(dims) for _ in range(num_layers)])
        self.k_norms = nn.ModuleList([nn.LayerNorm(dims) for _ in range(num_layers)])
        self.v_norms = nn.ModuleList([nn.LayerNorm(dims) for _ in range(num_layers)])
        self.out_norms = nn.ModuleList([nn.LayerNorm(dims) for _ in range(num_layers)])
        
        # FFN after each attention layer
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims, dims * 4),
                nn.GELU(),
                nn.Linear(dims * 4, dims)
            ) for _ in range(num_layers)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Forward pass through multi-layer attention with separate Q, K, V
        
        Args:
            x: Input tensor
            kv_caches: Optional list of cached K, V for each layer
            
        Returns:
            Tuple of (output tensor, updated KV caches)
        """
        batch_size, seq_len = x.shape[:2]
        
        # Initialize KV caches if not provided
        if kv_caches is None:
            kv_caches = [{} for _ in range(self.num_layers)]
            
        new_kv_caches = []
        
        # Process through layers
        for i in range(self.num_layers):
            residual = x
            
            # Determine if we have cached values
            layer_cache = kv_caches[i] if i < len(kv_caches) else {}
            k_cache = layer_cache.get("k")
            v_cache = layer_cache.get("v")
            
            # Process normalized inputs through separate Q, K, V modules
            q = self.query_modules[i](self.q_norms[i](x))
            
            # Only compute K,V if not cached or this is the first token
            if k_cache is None or v_cache is None or seq_len == 1:
                k = self.key_modules[i](self.k_norms[i](x))
                v = self.value_modules[i](self.v_norms[i](x))
            else:
                k, v = k_cache, v_cache
            
            # Process through attention combiner
            output = self.combiners[i](q, k, v)
            
            # Cache K, V for next forward pass
            new_kv_caches.append({"k": k, "v": v})
            
            # Apply FFN
            x = residual + output
            x = x + self.ffns[i](self.out_norms[i](x))
            
        return x, new_kv_caches


def demonstrate_advanced_patterns():
    # Example usage
    batch_size, seq_len, dims = 2, 4, 384
    heads = 2
    x = torch.randn(batch_size, seq_len, dims)
    
    print("Testing SharedKeyModule:")
    shared_key = SharedKeyModule(dims, heads, shared_keys=2)
    k = shared_key(x)
    print(f"Shared key shape: {k.shape}")
    
    print("\nTesting AdaptiveUpdateAttention:")
    adaptive_attn = AdaptiveUpdateAttention(dims, heads)
    output, cache_updates = adaptive_attn(x)
    print(f"Output shape: {output.shape}")
    print(f"Key updated: {cache_updates['key_updated']}")
    print(f"Value updated: {cache_updates['value_updated']}")
    
    print("\nTesting MultiLayerSeparatedAttention:")
    multi_attn = MultiLayerSeparatedAttention(dims, heads, num_layers=3)
    output, kv_caches = multi_attn(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of cached layers: {len(kv_caches)}")
    
    # Test autoregressively
    print("\nTesting autoregressive generation:")
    new_token = x[:, -1:, :]  # Simulate next token
    output, kv_caches = multi_attn(new_token, kv_caches)
    print(f"Autoregressive output shape: {output.shape}")


if __name__ == "__main__":
    demonstrate_advanced_patterns()
