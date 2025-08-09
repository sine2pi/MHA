
import torch

def taylor_sine(x, order=5):
    result = torch.zeros_like(x)
    for i in range(order + 1):
        if i % 2 == 1:  
            term = x**i / torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            if (i // 2) % 2 == 1: 
                result -= term
            else:
                result += term
    return result

def taylor_cosine(x, order=5):
    result = torch.zeros_like(x)
    for i in range(order + 1):
        if i % 2 == 0:  
            term = x**i / torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            if (i // 2) % 2 == 1: 
                result -= term
            else:
                result += term
    return result

class rotary(nn.Module):
    def __init__(self, dims, head):
        super(rotary, self).__init__()
        self.dims = dims
        self.head = head
        self.head_dim = dims // head

        self.theta = nn.Parameter((torch.tensor(1600, device=device, dtype=dtype)), requires_grad=False)  
        self.register_buffer('freqs_base', self._compute_freqs_base(), persistent=False)

    def _compute_freqs_base(self):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), self.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000 

    def forward(self, x) -> torch.Tensor:

        positions = (torch.arange(0, x.shape[2], device=x.device))
        freqs = (self.theta / 220.0) * self.freqs_base
        freqs = positions[:, None] * freqs 

        with torch.autocast(device_type="cuda", enabled=False):
            cos = taylor_cosine(freqs, order=self.taylor_order)
            sin = taylor_sine(freqs, order=self.taylor_order)
            rotary_dim = cos.shape[-1] 
            x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
            x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
            x_embed = torch.cat([x_embed, x_pass], dim=-1)
            return x_embed.type_as(x) 

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

############# 


class Tippecanoe_and_Tyler_too(nn.Module):
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
    
