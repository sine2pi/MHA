
import torch

def taylor_sine(x, order=5):

# sin(x) = sin(a) + cos(a)*(x-a)/1! - sin(a)*(x-a)^2/2! - cos(a)*(x-a)^3/3! + ...

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
        t = t * self.pos_scale
        powers = [t]
        for i in range(1, len(self.sin_coeffs)):
            powers.append(powers[-1] * t)
        sin_terms = sum(c * p for c, p in zip(self.sin_coeffs, powers))
        cos_terms = sum(c * p for c, p in zip(self.cos_coeffs, powers))
        batch = t.shape[0] if len(t.shape) > 1 else 1
        freqs_sin = sin_terms.view(batch, -1, 1).repeat(1, 1, self.dim//2)
        freqs_cos = cos_terms.view(batch, -1, 1).repeat(1, 1, self.dim//2)
        freqs = torch.stack([freqs_cos, freqs_sin], dim=-1).flatten(-2)
        return freqs

#######

def sine(x, order=5):
    original_shape = x.shape
    x = x.flatten(0, -2)
    exponents = torch.arange(1, order + 1, 2, device=x.device, dtype=torch.float32)
    x_powers = x.unsqueeze(-1) ** exponents
    factorials = torch.exp(torch.lgamma(exponents + 1))
    signs = (-1)**(torch.arange(0, len(exponents), device=x.device, dtype=torch.float32))
    terms = signs * x_powers / factorials
    result = terms.sum(dim=-1)
    return result.view(original_shape)

def cosine(x, order=5):
    original_shape = x.shape
    x = x.flatten(0, -2)
    exponents = torch.arange(0, order + 1, 2, device=x.device, dtype=torch.float32)
    x_powers = x.unsqueeze(-1) ** exponents
    factorials = torch.exp(torch.lgamma(exponents + 1))
    signs = (-1)**(torch.arange(0, len(exponents), device=x.device, dtype=torch.float32))
    terms = signs * x_powers / factorials
    result = terms.sum(dim=-1)
    return result.view(original_shape)

class rotary_vec(nn.Module):
    def __init__(self, dims, head):
        super(rotary, self).__init__()
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.taylor_order = 5

        self.theta = nn.Parameter((torch.tensor(16000, device=device, dtype=dtype)), requires_grad=False)  
        self.register_buffer('freqs_base', self._compute_freqs_base(), persistent=False)

    def _compute_freqs_base(self):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), self.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000 

    def forward(self, x) -> torch.Tensor:
        positions = (torch.arange(0, x.shape[2], device=x.device))
        freqs = (self.theta / 220.0) * self.freqs_base
        freqs = positions[:, None] * freqs 
        freqs_rescaled = (freqs + torch.pi) % (2 * torch.pi) - torch.pi 

        with torch.autocast(device_type="cuda", enabled=False):
            cos = cosine(freqs_rescaled, order=self.taylor_order)
            sin = sine(freqs_rescaled, order=self.taylor_order)
            rotary_dim = cos.shape[-1] 
            x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
            x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
            x_embed = torch.cat([x_embed, x_pass], dim=-1)
            return x_embed.type_as(x)

def sine_around_a(x, a, order=5):
    original_shape = x.shape
    x_flat = x.flatten(0, -2)
    x_minus_a = x_flat - a
    exponents = torch.arange(0, order + 1, device=x.device, dtype=torch.float32)
    x_minus_a_powers = x_minus_a.unsqueeze(-1) ** exponents
    derivatives_at_a = []
    current_deriv_val = torch.sin(a)
    derivatives_at_a.append(current_deriv_val)
    current_deriv_val = torch.cos(a)
    derivatives_at_a.append(current_deriv_val)
    current_deriv_val = -torch.sin(a)
    derivatives_at_a.append(current_deriv_val)
    current_deriv_val = -torch.cos(a)
    derivatives_at_a.append(current_deriv_val)
    
    for i in range(4, order + 1):
        if i % 4 == 0:
            derivatives_at_a.append(torch.sin(a))
        elif i % 4 == 1:
            derivatives_at_a.append(torch.cos(a))
        elif i % 4 == 2:
            derivatives_at_a.append(-torch.sin(a))
        else:
            derivatives_at_a.append(-torch.cos(a))
    derivs_tensor = torch.stack(derivatives_at_a, dim=-1)

    factorials = torch.exp(torch.lgamma(exponents + 1))
    inv_factorials = 1.0 / factorials

    terms = derivs_tensor * x_minus_a_powers * inv_factorials
    result = terms.sum(dim=-1)
    return result.view(original_shape)

def cosine_around_a(x, a, order=5):
    original_shape = x.shape
    x_flat = x.flatten(0, -2)
    x_minus_a = x_flat - a

    exponents = torch.arange(0, order + 1, device=x.device, dtype=torch.float32)
    x_minus_a_powers = x_minus_a.unsqueeze(-1) ** exponents

    derivatives_at_a = []
    current_deriv_val = torch.cos(a)
    derivatives_at_a.append(current_deriv_val)
    current_deriv_val = -torch.sin(a)
    derivatives_at_a.append(current_deriv_val)
    current_deriv_val = -torch.cos(a)
    derivatives_at_a.append(current_deriv_val)
    current_deriv_val = torch.sin(a)
    derivatives_at_a.append(current_deriv_val)
    for i in range(4, order + 1):
        if i % 4 == 0:
            derivatives_at_a.append(torch.cos(a))
        elif i % 4 == 1:
            derivatives_at_a.append(-torch.sin(a))
        elif i % 4 == 2:
            derivatives_at_a.append(-torch.cos(a))
        else:
            derivatives_at_a.append(torch.sin(a))
    derivs_tensor = torch.stack(derivatives_at_a, dim=-1)

    factorials = torch.exp(torch.lgamma(exponents + 1))
    inv_factorials = 1.0 / factorials

    terms = derivs_tensor * x_minus_a_powers * inv_factorials
    result = terms.sum(dim=-1)
    return result.view(original_shape)

class rotary_vec_a(nn.Module):
    def __init__(self, dims, head):
        super(rotary, self).__init__()
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.taylor_order = 5

        self.theta = torch.nn.Parameter(torch.randn(1))
        self.expansion_point = torch.tensor(0.0, device='cuda')

        self.theta = nn.Parameter((torch.tensor(1600, device=device, dtype=dtype)), requires_grad=False)  
        self.register_buffer('freqs_base', self._compute_freqs_base(), persistent=False)

    def _compute_freqs_base(self):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), self.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000 

    def forward(self, x) -> torch.Tensor:
        positions = (torch.arange(0, x.shape, device=x.device))
        freqs = (self.theta / 220.0) * self._compute_freqs_base()
        freqs = positions[:, None] * freqs

        with torch.autocast(device_type="cuda", enabled=False):
            
            cos = cosine_around_a(freqs, self.expansion_point, order=self.taylor_order)
            sin = sine_around_a(freqs, self.expansion_point, order=self.taylor_order)

            rotary_dim = cos.shape[-1] 
            x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

            x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
            x_embed = torch.cat([x_embed, x_pass], dim=-1)
            return x_embed.type_as(x)
