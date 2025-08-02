class HierarchicalAttention(nn.Module):
    def __init__(self, dims, head, sharpen=True, max_dist=128, levels=3, temp_scale=0.01):
        super().__init__()
        self.dims = dims
        self.head = head
        self.max_dist = max_dist
        self.sharpen = sharpen
        self.levels = levels
        self.temp_scale = temp_scale

        self.span_predictor = Linear(in_features=dims, out_features=1)

        self.local_level_projections = nn.ModuleList([
            Linear(dims, dims) for _ in range(levels)
        ])
        self.local_level_attentions = nn.ModuleList([
            MultiheadA3(dims=dims, head=head, max_dist=max_dist) for _ in range(levels)
        ])
        self.global_level_projections = nn.ModuleList([
            Linear(dims, dims) for _ in range(levels)
        ])
        self.global_level_attentions = nn.ModuleList([
            MultiheadA3(dims=dims, head=head, max_dist=max_dist) for _ in range(levels)
        ])

        self.ln_local = LayerNorm(normalized_shape=dims)
        self.ln_global = LayerNorm(normalized_shape=dims)
        self.projection = Linear(in_features=2 * dims, out_features=dims)

    def forward(self, x):
        local = self.ln_local(x)
        global_ = self.ln_global(x)

        globe_out = self._hierarchical_attention(global_, self.global_level_projections, self.global_level_attentions)  # (seq_len, batch_size, dims)
        span_scale = torch.sigmoid(self.span_predictor(globe_out.mean(dim=1)))
        local_out = self._hierarchical_attention(local, self.local_level_projections, self.local_level_attentions)
        combined = torch.cat([local_out, globe_out], dim=-1)
        x = self.projection(combined)
        return x

    def _hierarchical_attention(self, x, level_projections, level_attentions):
        seq_len, batch_size, dims = x.size()
        outputs = []
        max_downsample_level = min(self.levels, int(math.log2(seq_len)))
        
        for level in range(max_downsample_level):
            factor = 2 ** level
            curr_len = seq_len // factor
            pooled_x = x[:curr_len * factor].view(curr_len, factor, batch_size, dims).mean(dim=1)
            
            projected = level_projections[level](pooled_x)
            attention_out, _ = level_attentions[level](projected, projected, projected)
            
            if factor > 1:
                attention_out = F.interpolate(
                    attention_out.permute(1, 2, 0),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).permute(2, 0, 1)  
            
            outputs.append(attention_out)
        
        for level in range(max_downsample_level, self.levels):
            projected = level_projections[level](x)
            attention_out, _ = level_attentions[level](projected, projected, projected)
            outputs.append(attention_out)
        
        weights = torch.softmax(torch.ones(len(outputs)), dim=0)
        combined_output = sum(out * w for out, w in zip(outputs, weights))
        
        return combined_output

class Focus(nn.Module):
    def __init__(self, dims, head, sharpen=True, max_dist=128, levels=3, win_size=64, max_span=256, temp_scale=0.01):
        super().__init__()
        self.dims = dims
        self.head = head
        self.max_dist = max_dist
        self.sharpen = sharpen
        self.levels = levels
        self.win_size = win_size
        self.max_span = max_span
        self.temp_scale = temp_scale

        # Span predictor
        self.span_predictor = Linear(in_features=dims, out_features=1)

        # Hierarchical attention layers
        self.hierarchical_attn_local = HierarchicalAttention(dims=dims, head=head, levels=levels)
        self.hierarchical_attn_global = HierarchicalAttention(dims=dims, head=head, levels=levels)

        # Layer norms and projection
        self.ln_local = LayerNorm(normalized_shape=dims)
        self.ln_global = LayerNorm(normalized_shape=dims)
        self.projection = Linear(in_features=2 * dims, out_features=dims)

    def forward(self, x):
        # Apply layer norms
        local = self.ln_local(x)
        global_ = self.ln_global(x)

        # Global hierarchical attention
        globe_out = self.hierarchical_attn_global(global_.transpose(0, 1))  # (seq_len, batch_size, dims)
        globe_out = globe_out.transpose(0, 1)  # Back to (batch_size, seq_len, dims)

        # Predict span scale
        span_scale = torch.sigmoid(self.span_predictor(globe_out.mean(dim=1)))

        # Local hierarchical attention with sliding window
        local_out = self._sliding_window_hierarchical_attention(local, span_scale)

        # Combine local and global outputs
        combined = torch.cat([local_out, globe_out], dim=-1)
        x = self.projection(combined)

        return x

    def _sliding_window_hierarchical_attention(self, x, span_scale):
        batch_size, seq_len, dims = x.size()
        num_windows = (seq_len + self.win_size - 1) // self.win_size  # Calculate the number of windows

        # Create tensors to store the outputs
        output = torch.zeros_like(x, device=x.device)

        # Iterate over the windows in a more efficient manner
        for i in range(num_windows):
            start_idx = i * self.win_size
            end_idx = min((i + 1) * self.win_size, seq_len)
            query = x[:, start_idx:end_idx, :]

            # Define the range of keys and values
            key_start = max(0, start_idx - self.max_span + self.win_size)
            key_end = min(start_idx + self.max_span, seq_len)
            key = x[:, key_start:key_end, :]
            value = x[:, key_start:key_end, :]

            attn_out = self.hierarchical_attn_local(query.transpose(0, 1))
            attn_out = attn_out.transpose(0, 1)  # Back to (batch_size, seq_len, dims)
            output[:, start_idx:end_idx, :] = attn_out

        return output
