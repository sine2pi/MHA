class TaylorRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_terms=4, learned_coeff=True, device=None):
        super().__init__()
        self.dim = dim
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sin_coeffs = torch.zeros(max_terms, device=device)
        cos_coeffs = torch.zeros(max_terms, device=device)
        if max_terms > 0: cos_coeffs[0] = 1.0
        if max_terms > 1: sin_coeffs[1] = 1.0
        if max_terms > 2: cos_coeffs[2] = -0.5
        if max_terms > 3: sin_coeffs[3] = -1.0/6.0
        self.sin_coeffs = nn.Parameter(sin_coeffs, requires_grad=learned_coeff)
        self.cos_coeffs = nn.Parameter(cos_coeffs, requires_grad=learned_coeff)
        self.pos_scale = nn.Parameter(torch.tensor([0.1], device=device))
        self.rot = nn.Parameter(torch.tensor([1.0], device=device))
        self.scale_base = 1.0
    
    def forward(self, t):
        device = t.device
        t = t.to(device) * self.pos_scale
        powers = [t]
        for i in range(1, len(self.sin_coeffs)):
            powers.append(powers[-1] * t)
        sin_terms = sum(c * p for c, p in zip(self.sin_coeffs, powers))
        cos_terms = sum(c * p for c, p in zip(self.cos_coeffs, powers))
        batch_size = t.shape[0] if len(t.shape) > 1 else 1
        freqs_sin = sin_terms.view(batch_size, -1, 1).repeat(1, 1, self.dim//2)
        freqs_cos = cos_terms.view(batch_size, -1, 1).repeat(1, 1, self.dim//2)
        freqs = torch.stack([freqs_cos, freqs_sin], dim=-1).flatten(-2)
        return freqs
        
    def rotate_(self, t, seq_dim=None, offset=0, scale=None, continuous=True):
        """Apply rotation to input tensor"""
        t_clone = t.clone()
        if len(t_clone.shape) == 4:
            ctx = t_clone.shape[2]
            seq_dim_val = 2
        else:
            ctx = t_clone.shape[1]
            seq_dim_val = 1
        device = t_clone.device
        seq = torch.arange(ctx, device=device, dtype=t_clone.dtype) + offset
        seq = seq + 0.01
        freqs = self.forward(seq)
        scale_value = scale if scale is not None else self.scale_base
        scaled_freqs = freqs * self.rot
        scale_tensor = scale_value
        
        result = self.apply_rotary(scaled_freqs, t_clone, 
                                  scale=scale_tensor, 
                                  seq_dim=seq_dim_val)
        return result
    
    def apply_rotary(self, freqs, t, start_index=0, scale=1., seq_dim=-2, freqs_seq_dim=None):
        """Apply rotary transformation to input tensor"""
        dtype = t.dtype
        def _exists(val):
            return val is not None
        def _slice_at_dim(tensor, dim_slice, dim):
            dim += (tensor.ndim if dim < 0 else 0)
            colons = [slice(None)] * tensor.ndim
            colons[dim] = dim_slice
            return tensor[tuple(colons)]
        def _rotate_half(x):
            x = rearrange(x, '... (d r) -> ... d r', r=2)
            x1, x2 = x.unbind(dim=-1)
            x = torch.stack((-x2, x1), dim=-1)
            return rearrange(x, '... d r -> ... (d r)')
        if not _exists(freqs_seq_dim):
            if freqs.ndim == 2 or t.ndim == 3:
                freqs_seq_dim = 0
        if t.ndim == 3 or _exists(freqs_seq_dim):
            ctx = t.shape[seq_dim]
            freqs = _slice_at_dim(freqs, slice(-ctx, None), dim=freqs_seq_dim)
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} not sufficient for rotation {rot_dim}'
        t_left = t[..., :start_index]
        t_middle = t[..., start_index:end_index]
        t_right = t[..., end_index:]

        t_transformed = (t_middle * freqs.cos() * scale) + (_rotate_half(t_middle) * freqs.sin() * scale)
        out = torch.cat((t_left, t_transformed, t_right), dim=-1)
        return out.type(dtype)
    
