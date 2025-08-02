Streaming-Transformer with adaptiive_update

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
    batch_size, seq_len, dims = 8, 8, 384
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
    
    
class StreamingTransformer(nn.Module):
    def __init__(self, dims, heads):
        super().__init__()
        self.adaptive_layers = nn.ModuleList([
            AdaptiveUpdateAttention(dims, heads) for _ in range(num_layers)
        ])
        # Initialize caches
        self.key_caches = [None] * num_layers
        self.value_caches = [None] * num_layers
        
    def process_stream_chunk(self, new_data):
        # Process incoming data while reusing cached keys/values when possible
        x = self.embedding(new_data)
        for i, layer in enumerate(self.adaptive_layers):
            x, cache_updates = layer(x, key_cache=self.key_caches[i], 
                                    value_cache=self.value_caches[i])
            self.key_caches[i] = cache_updates["key_cache"]
            self.value_caches[i] = cache_updates["value_cache"]
        return x
    
    
def process_video_sequence(frames, model):
    outputs = []
    key_cache = value_cache = None
    
    for frame in frames:
        output, caches = model(frame, key_cache=key_cache, value_cache=value_cache)
        key_cache, value_cache = caches["key_cache"], caches["value_cache"]
        outputs.append(output)
        
    # Track which frames triggered updates
    update_frames = torch.stack([c["key_updated"] for c in caches])
    return outputs, update_frames
