
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Callable, Dict, List
from torch.nn import MultiheadAttention as mha  # noqa: F401
from torch.nn.functional import scaled_dot_product_attention
device = torch.device(device="cuda:0")
dtype = torch.float32



class Refiner:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.R = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.default_value = 0.0

    def get_value(self, state, action):
        return self.R.get((state, action), self.default_value)

    def set_value(self, state, action, value):
        self.R[(state, action)] = value

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.actions)
        else:
            action_values = [self.get_value(state, a) for a in range(self.actions)]
            return np.argmax(action_values)

    def update(self, state, action, reward, next_state):
        next_values = [self.get_value(next_state, a) for a in range(self.actions)]
        best_next_value = max(next_values)

        old_value = self.get_value(state, action)
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - old_value
        new_value = old_value + self.alpha * td_error
        self.set_value(state, action, new_value)

class Predictor(nn.Module):
    """Neural predictor for span scale estimation."""
    def __init__(self, dims):
        super().__init__()
        self.linear = Linear(in_features=dims, out_features=1)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, global_out):
        if global_out.dim() > 2:
            global_out = global_out.mean(dim=1)
        scale = torch.sigmoid(self.linear(global_out))
        return scale

class ProjectionModule(nn.Module):
    def __init__(self, dims: int, head: int, proj_type: str = "query", use_bias: bool = True):
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.proj_type = proj_type
        self.scale = self.head_dim ** -0.25 if proj_type != "value" else 1.0
        self.proj = Linear(in_features=dims, out_features=dims, bias=use_bias)
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(tensor=self.proj.weight, std=0.02)
        if hasattr(self.proj, 'bias') and self.proj.bias is not None:
            nn.init.zeros_(tensor=self.proj.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        batch, ctx = x.shape[:2]
        proj = self.proj(x)
        
        proj = proj.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
        if self.proj_type in ["query", "key"]:
            proj = proj * self.scale
        return proj

def create_attention_mask(batch_size, ctx, is_causal=True, padding_mask=None, device=None):
    if is_causal:
        mask = torch.triu(torch.ones((ctx, ctx), device=device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, ctx, ctx)
    else:
        mask = torch.zeros((batch_size, 1, ctx, ctx), device=device).bool()
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        mask = mask | (~padding_mask)
    return mask

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype))
    
class BaseAttention(nn.Module):
    """Base class for attention mechanisms with common functionality."""
    use_sdpa = True
    
    def __init__(self, dims: int, head: int, max_dist: int = 512):
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.max_dist = max_dist
        self.scale = self.head_dim ** -0.25
        
    def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
        return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()
        
    def _reshape_to_output(self, attn_output, batch, ctx):
        return attn_output.permute(0, 2, 1, 3).reshape(batch, ctx, self.dims)

def calculate_attention(q, k, v, mask=None, temperature=1.0, use_sdpa=True, is_causal=True):
    batch_size = q.shape[0]
    ctx = q.shape[2]
    attn_mask = None
    if mask is not None:
        if mask.dim() <= 3:
            attn_mask = create_attention_mask(
                batch_size=batch_size, 
                ctx=ctx, 
                is_causal=is_causal, 
                padding_mask=mask if mask.dim() > 1 else None,
                device=q.device)
        else:
            attn_mask = mask
    scaled_q = q
    if temperature != 1.0 and temperature > 0:
        scaled_q = q * (1.0 / temperature)**.5
    a = scaled_dot_product_attention(
        scaled_q, k, v, 
        attn_mask=attn_mask, 
        is_causal=is_causal if attn_mask is None else False)
    out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
    return out, None


def calculate_attention2(q, k, v, mask=None, temperature=1.0, use_sdpa=True, is_causal=True):
    if use_sdpa:
        try:
            if mask is not None:
                if mask.dtype == torch.bool:
                    float_mask = torch.zeros_like(mask, dtype=torch.float)
                    float_mask = float_mask.masked_fill(mask, float('-inf'))
                    attn_output = scaled_dot_product_attention(
                        q, k, v, attn_mask=float_mask)
                else:
                    attn_output = scaled_dot_product_attention(
                        q, k, v, attn_mask=mask, is_causal=is_causal)
            else:
                attn_output = scaled_dot_product_attention(
                    q, k, v, attn_mask=None, is_causal=is_causal)
            return attn_output, None
        except RuntimeError:
            pass
    scale = 1.0 / temperature if temperature > 0 else 1.0
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale
    
    if mask is not None:
        if mask.dim() == 4:
            q_len, k_len = q.size(2), k.size(2)
            mask_q_len = min(mask.size(2), q_len)
            mask_k_len = min(mask.size(3), k_len)
            
            if mask.dtype == torch.bool:
                mask_part = mask[:, :, :mask_q_len, :mask_k_len]
                attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len].masked_fill(mask_part, float("-inf"))
            else:
                attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len] + mask[:, :, :mask_q_len, :mask_k_len]
    attn = F.softmax(attn, dim=-1)
    
    if mask is not None and mask.dtype == torch.bool:
        binary_mask = (~mask).float()
        masked_attn = attn * binary_mask
        attn_sum = masked_attn.sum(dim=-1, keepdim=True)
        attn = masked_attn / (attn_sum + 1e-6)
    attn_output = torch.matmul(attn, v)
    return attn_output, attn

class AdaptiveSpan(BaseAttention):
    """Attention with adaptive span size."""
    def __init__(self, dims, head, max_dist, sharpen=True, temp_scale=0.01):
        super().__init__(dims, head, max_dist)
        self.sharpen = sharpen
        self.temp_scale = temp_scale
        self.span_scale = nn.Parameter(torch.tensor(1.0))


    def forward(self, query, key, value, max_dist=None, max_span=None, span_scale=None, is_causal=True):
        if max_dist is None:
            max_dist = self.max_dist
        if max_span is None:
            max_span = query.shape[1]
        if span_scale is None:
            span_scale = self.span_scale
            
        span_mean = span_scale.mean().item()
        span_len = min(int(max_span * span_mean), query.shape[1], key.shape[1], value.shape[1])
        eff_span = min(span_len, max_dist)
        
        if eff_span == 0:
            batch = query.shape[0]
            return (torch.zeros(batch, eff_span, self.dims, device=query.device), None)
            
        q_span = query[:, :eff_span, :]
        k_span = key[:, :eff_span, :]
        v_span = value[:, :eff_span, :]

        batch = q_span.shape[0]

        q = self._shape(q_span, q_span.size(1), batch)
        k = self._shape(k_span, k_span.size(1), batch)
        v = self._shape(v_span, v_span.size(1), batch)

        temperature = (1.0 + self.temp_scale * (1.0 - span_mean)
            if self.sharpen
            else 0.5 + self.temp_scale * span_mean)
        
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            attn_output, weights = calculate_attention(
                q, k, v, None, temperature, BaseAttention.use_sdpa, is_causal=is_causal)
            out = self._reshape_to_output(attn_output, batch, eff_span)
        return out, weights

class MyelinatedLayer(BaseAttention):
    def __init__(self, dims, head, layerA=3, sparsity_threshold=0.1, max_dist=512):
        super().__init__(dims, head, max_dist)
        self.layers = nn.ModuleList()
        self.layerA = layerA
        self.sparsity_threshold = sparsity_threshold
        self.max_dist = max_dist
        
        self.node_predictors = nn.ModuleList([
            nn.Sequential(LayerNorm(dims),
                        Linear(dims, 1),
                        nn.Sigmoid()) for _ in range(layerA)])
        
        for i in range(layerA):
            self.layers.append(nn.ModuleDict({
                'ln': LayerNorm(dims),
                'gate': nn.Sequential(Linear(dims, 1), nn.Sigmoid()),
                'adapter': Linear(dims, dims) if i % 2 == 0 else None
            }))
        self.policy_net = nn.Sequential(Linear(dims, 128), nn.ReLU(), Linear(128, 3))
        self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]))
        
        mlp = dims * 4
        self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.mlp = nn.Sequential(Linear(dims, mlp), nn.GELU(), Linear(mlp, dims))
        self.mlp_ln = LayerNorm(dims)
        
        self.working_memory = nn.Parameter(torch.zeros(1, 1, dims))
        self.memory_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.last_memory_gate_values = None

    def compute_attention(self, norm_x, mask=None, kv_cache=None, is_causal=True):
        """Compute attention with adaptive span and content-dependent updates."""
        batch, ctx = norm_x.shape[:2]
        
        q = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)
        k = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)
        v = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)

        attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa, is_causal=is_causal)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, ctx, -1)
        return attn_output

    def predict_node_importance(self, x, layer_idx):
        """Dynamically determine if processing should occur at this node."""
        importance = self.node_predictors[layer_idx](x)
        return (importance > self.sparsity_threshold).float()

    def decide_jump(self, policy, jump_weights, i, layerA, x, original_x, working_memory):
        """Decide whether to jump layers based on the policy network."""
        jump_prob = policy[:, 1] if i < layerA - 1 else torch.zeros_like(policy[:, 1])
        should_jump = (torch.rand_like(jump_prob) < jump_prob).any()
        if should_jump:
            jump_length = torch.multinomial(policy, 1)[:, 0].max().item() + 1
            i_next = min(i + jump_length, layerA - 1)
            skip_weight = jump_weights[min(jump_length - 1, 2)]
            x = x + skip_weight * original_x + (1 - skip_weight) * working_memory
            return x, i_next
        return x, i + 1

    def forward(self, x, xa=None, mask=None, kv_cache=None, is_causal=True):
        batch, ctx = x.shape[:2]
        working_memory = self.working_memory.expand(batch, -1, -1)
        original_x = x
        pooled_representation = x.mean(dim=1, keepdim=False)
        policy_logits = self.policy_net(pooled_representation)
        policy = F.softmax(policy_logits, dim=-1)
        jump_history = []
        memory_gate = torch.zeros(batch, 1, 1, device=x.device)
        
        i = 0
        while i < self.layerA:
            layer = self.layers[i]
            node_importance = self.predict_node_importance(x, i)
            print(f"Node importance (Layer {i}): {node_importance}")

            if node_importance.mean() < 0.2 and i > 0:
                i += 1
                jump_history.append(i)
                continue
            norm_x = layer['ln'](x)
            attn_mask = mask * node_importance.squeeze(-1).unsqueeze(1) if mask is not None else node_importance.squeeze(-1).unsqueeze(1)
            
            if node_importance.mean() > 0.3:
                attn_output = self.compute_attention(norm_x, mask=attn_mask, kv_cache=kv_cache)
                print(f"Attention output (Layer {i}): {attn_output}")
                
                if layer['adapter'] is not None:
                    attn_output = layer['adapter'](attn_output)
                gate_value = layer['gate'](norm_x)
                x = x + gate_value * attn_output
                print(f"Updated representation (Layer {i}): {x}")
                
                memory_gate = self.memory_gate(x.mean(dim=1, keepdim=True))
                mean_x = x.mean(dim=1, keepdim=True)
                working_memory = memory_gate * working_memory + (1 - memory_gate) * mean_x
                print(f"Memory gate value: {memory_gate}")
            
            x, i = self.decide_jump(policy, self.jump_weights, i, self.layerA, x, original_x, working_memory)
            jump_history.append(i)

        self.last_memory_gate_values = memory_gate.detach().clone()
        print(f"Jump history: {jump_history}")
        mlp_importance = self.mlp_gate(x)
        mlp_output = self.mlp(self.mlp_ln(x))
        x = x + mlp_importance * mlp_output
        print(f"Final output: {x}")
        return x

class Multihead1(nn.Module):  # encoder - decoder attention
    cosa = False
    magnitude = False
    sdpa = True
    use_qkv = False # removing qkv projection for now
    
    def __init__(self, dims, head, layer_idx, decoder, dropout=0.0, bias=False):
        tox = {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
               "dtype": torch.float32}
        super().__init__()
        # self.qkv = qkv_proj(dims, bias=bias, **tox) if Multihead.use_qkv else n_proj(dims, bias=bias, **tox)
        # super().__init__(embed_dim=dims, num_heads=head, dropout=dropout, bias=bias, kdim=dims, vdim=dims, batch_first=True)
        self.dims = dims
        self.head = head

        self.layer_idx = layer_idx
        self.decoder = decoder
        self.dropout = dropout
        self.bias = bias
        self.head_dim = dims // head
        assert self.dims % self.head == 0, f"{self.dims} must be divisible by {self.head}"
        self.scale = self.head_dim ** -0.5


        if Multihead1.use_qkv:
            self.qkv = Linear(dims, dims * 3, bias=bias, **tox)
        else:
            self.q = Linear(dims, dims, **tox)
            self.k = Linear(dims, dims, bias=bias, **tox)
            self.v = Linear(dims, dims, **tox)
        self.o = Linear(dims, dims, bias=bias, **tox)

        self.print_once = False

    def cos_attention(self, q: Tensor, k: Tensor, v: Tensor, mask) -> Tensor:
        q_norm = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
        k_norm = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
        qk_cosine = torch.matmul(q_norm, k_norm.transpose(-1, -2))
        
        if Multihead1.magnitude:
            q_magnitude = torch.norm(q, dim=-1, keepdim=True)
            k_magnitude = torch.norm(k, dim=-1, keepdim=True)
            magnitude_scaling = (q_magnitude * k_magnitude.transpose(-1, -2)) ** 0.5
            magnitude_scaling = torch.clamp(magnitude_scaling, min=1e-8)
            qk_cosine = qk_cosine * magnitude_scaling

        qk_cosine = qk_cosine + mask
        weights = F.softmax(qk_cosine, dim=-1)
        out = torch.matmul(weights, v)
        return out


    def forward(self, x, xa=None, kv_cache=None, mask=None, decoder=False):
        B, L, D = x.shape
        batch, ctx, dims = B, L, D
        if xa is not None:
            batch, ctx, dims = xa.shape

        scale = self.scale 
        is_causal=decoder

        x = xa if xa is not None else x

        if Multihead1.use_qkv:
            result = self.qkv(x)
            q, k, v = torch.chunk(result, 3, dim=-1)
            q_w, k_w, v_w = self.qkv.weight.chunk(3, dim=0)

            if self.bias:
                q_bias, k_bias, v_bias = torch.chunk(self.qkv.bias, 3, dim=0)
            else:
                q_bias, k_bias, v_bias = None, None, None
            q, k, v = (
                F.linear(q, q_w, q_bias),
                F.linear(k, k_w, k_bias),
                F.linear(v, v_w, v_bias))
        else:
            q = self.q(x)
            if kv_cache is None or xa is None or self.k not in kv_cache:
                k = self.k(x if xa is None else xa)
                v = self.v(x if xa is None else xa)
            else:
                k = kv_cache[self.k]
                v = kv_cache[self.v]

            if kv_cache is not None:
                kv_cache[self.k] = k
                kv_cache[self.v] = v

        B, L, D = q.shape
        
        q = q.unflatten(-1, [self.head, self.head_dim]).transpose(1, 2)  # (B, L, D) -> (B, head, L, head_dim)
        k = k.unflatten(-1, [self.head, self.head_dim]).transpose(1, 2)
        v = v.unflatten(-1, [self.head, self.head_dim]).transpose(1, 2)

        if decoder and not xa:
            causal_mask = torch.empty(L, L, device=q.device).fill_(-np.inf).triu_(1)
            self.register_buffer("causal_mask", causal_mask, persistent=False)
            causal_mask = causal_mask.expand(B, self.head, L, L)
            mask = causal_mask if mask is None else mask + causal_mask
            is_causal = True

        else:
            self.register_buffer("causal_mask", None, persistent=False)
            mask = None
            is_causal = False

        if Multihead1.cosa:
            if not self.print_once:
                print("cosa")
            output = self.cos_attention(q, k, v, mask=mask)
            qk = None

        if Multihead1.sdpa:
            if not self.print_once:
                print("sdpa")
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout, is_causal=is_causal)  # (B, head, L, head_dim)
            qk = None

        else:
            if not self.print_once:
                print("regular")
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:ctx, :ctx]
            qk = qk.float()
            w = F.softmax(qk, dim=-1).to(q.dtype)
            output = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        self.print_once = True
        output = output.transpose(1, 2).flatten(-2)  # (B, L, head * head_dim) -> (B, L, D)
        out = self.o(output)
        return out, qk
      
class MultiheadA(nn.Module): #encoder
    def __init__(self, dims: int, head: int):
        super().__init__()
        
        self.head = head
        self.dims = dims
        self.head_dim = dims // head
        self.scale = self.head_dim ** -0.5
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.o = Linear(dims, dims)

    def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
        return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(self, x: Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, ctx = x.shape[:2]
        q = self.q(x) * self.scale
        k = self.k(x) * self.scale
        v = self.v(x)
        q = self._shape(q, ctx, batch)
        k = self._shape(k, k.size(1), batch)
        v = self._shape(v, v.size(1), batch)

        a = scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False, enable_gqa=False)
        out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        out = self.o(out)
        return out

class MultiheadB(nn.Module):
    scaling = True 
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head
        self.scale = self.head_dim ** -0.5
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.o = Linear(dims, dims)
        
    def cos_attention(self, q: Tensor, k: Tensor, v: Tensor, mask) -> Tensor:
        q_norm = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
        k_norm = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
        qk_cosine = torch.matmul(q_norm, k_norm.transpose(-1, -2))
        
        if MultiheadB.scaling:
            q_magnitude = torch.norm(q, dim=-1, keepdim=True)
            k_magnitude = torch.norm(k, dim=-1, keepdim=True)
            magnitude_scaling = (q_magnitude * k_magnitude.transpose(-1, -2)) ** 0.5
            magnitude_scaling = torch.clamp(magnitude_scaling, min=1e-8)
            qk_cosine = qk_cosine * magnitude_scaling

        qk_cosine = qk_cosine + mask
        weights = F.softmax(qk_cosine, dim=-1)
        out = torch.matmul(weights, v)
        return out
        
    def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
        return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(self, x: Tensor, xa = None, mask = None, kv_cache = None, is_causal=True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, ctx = x.shape[:2]
        q = self.q(x)

        if kv_cache is None or xa is None or self.k not in kv_cache:
            k = self.k(x if xa is None else xa)
            v = self.v(x if xa is None else xa)
        else:
            k = kv_cache[self.k]
            v = kv_cache[self.v]

        wv = self._attention(q, k, v, mask, is_causal=is_causal)
        return self.o(wv), None

    def _attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor, is_causal: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = q.shape
        batch, ctx, dims = B, L, D  # noqa: F841
        q = q * self.scale
        k = k * self.scale

        if is_causal:
            causal_mask = torch.empty(ctx, ctx, device=q.device).fill_(-np.inf).triu_(1)
            self.register_buffer("causal_mask", causal_mask, persistent=False)

            causal_mask = causal_mask.expand(batch, self.head, ctx, ctx)
            mask = causal_mask if mask is None else mask + causal_mask

        else:
            self.register_buffer("causal_mask", None, persistent=False)

        q = self._shape(q, ctx, batch)
        k = self._shape(k, k.size(1), batch)
        v = self._shape(v, v.size(1), batch)

        out = self.cos_attention(q, k, v, mask=mask)
        out = out.permute(0, 2, 1, 3).flatten(start_dim=2)
        return out

class MultiheadC(nn.Module):
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head
        self.scale = self.head_dim ** -0.5
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.o = Linear(dims, dims)
            
    def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
        return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(self, x: Tensor, xa: Optional[Tensor], kv_cache = None, is_causal=True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, ctx = x.shape[:2]

        if kv_cache is None or xa is None or self.k not in kv_cache:
            k = self.k(x if xa is None else xa)
            v = self.v(x if xa is None else xa)
        else:
            k = kv_cache[self.k]
            v = kv_cache[self.v]

        q = self.q(x)
        wv = self._attention(q, k, v, is_causal=is_causal)
        return self.o(wv), None

    def _attention(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, ctx, self.dims = q.shape

        q = q * self.scale
        k = k * self.scale

        q = self._shape(q, ctx, batch)
        k = self._shape(k, k.size(1), batch)
        v = self._shape(v, v.size(1), batch)

        a = scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False, enable_gqa=False)
        out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        out = self.o(out)
        return out

class MultiheadD(nn.Module):
    def __init__(self, dims: int, head: int, get_total_layers: Callable):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head
        self.get_total_layers = get_total_layers  # A callable to dynamically fetch total layers
        self.layers = nn.ModuleList()
        
        # Define layers for q, k, and v projection
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims)  # For encoder
        self.v = Linear(dims, dims)  # For encoder
        # Output layer
        self.out = Linear(dims, dims)

        self.layer_count = self._calculate_layer_count()
        for _ in range(self.layer_count):
            self.layers.append(self._create_layer())

    def _calculate_layer_count(self):
        total_layers = self.get_total_layers()
        return max(1, total_layers // 2)

    def _create_layer(self):
        # Define the structure of a single layer
        return nn.Sequential(
            nn.Linear(self.dims, self.dims),
            nn.ReLU(),
            nn.LayerNorm(self.dims))

    def forward(self, decoder_output: Tensor, encoder_output: Tensor, mask=None):
        # Example forward pass applying layers sequentially
        #                     
        batch, ctx_d, _ = decoder_output.shape
        batch, ctx_e, _ = encoder_output.shape

        # Project queries (decoder), ks (encoder), and vs (encoder)
        q = self.q(decoder_output)  # Queries from decoder output
        k = self.k(encoder_output)   # ks from encoder output
        v = self.v(encoder_output) # vs from encoder output

        # Reshape for multi-head attention
        q = q.view(batch, ctx_d, self.head, -1).transpose(1, 2)
        k = k.view(batch, ctx_e, self.head, -1).transpose(1, 2)
        v = v.view(batch, ctx_e, self.head, -1).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax over the last dimension (k dimension)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute the attention output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, ctx_d, -1)

        # Project back to original dimension space
        output = self.out(attn_output)

        print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        print(f"Attention scores shape: {attn_scores.shape}")
        print(f"Attention output shape: {attn_output.shape}")
        return output

    def _forward(self, decoder_output: Tensor, encoder_output: Tensor, mask=None):
        # Example forward pass applying layers sequentially
        x = decoder_output + encoder_output  # Combine encoder and decoder outputs
        for layer in self.layers:
            x = layer(x)
        return x

class QueryModule(nn.Module):
    """Dedicated query projection module that handles only query transformations."""
    
    def __init__(self, dims: int, heads: int):
        """
        Args:
            dims: Input/output dimension size
            heads: Number of attention heads
        """
        super().__init__()
        
        assert dims % heads == 0, f"dims ({dims}) must be divisible by heads ({heads})"
        
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads
        self.scale = self.head_dim ** -0.25
        
        # Only query projection
        self.query = nn.Linear(in_features=dims, out_features=dims)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(tensor=self.query.weight, std=0.02)
        if self.query.bias is not None:
            nn.init.zeros_(tensor=self.query.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Project input to query representation
        
        Args:
            x: Input tensor [batch, seq_len, dims]
            
        Returns:
            Query tensor [batch, heads, seq_len, head_dim]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Project and reshape for attention
        q = self.query(x)
        q = q.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply scaling pre-emptively for stable attention
        q = q * self.scale
        
        return q

class KeyModule(nn.Module):
    """Dedicated key projection module that handles only key transformations."""
    
    def __init__(self, dims: int, heads: int):
        """
        Args:
            dims: Input/output dimension size
            heads: Number of attention heads
        """
        super().__init__()
        
        assert dims % heads == 0, f"dims ({dims}) must be divisible by heads ({heads})"
        
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads
        self.scale = self.head_dim ** -0.25
        
        # Only key projection
        self.key = nn.Linear(in_features=dims, out_features=dims, bias=False)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(tensor=self.key.weight, std=0.02)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Project input to key representation
        
        Args:
            x: Input tensor [batch, seq_len, dims]
            
        Returns:
            Key tensor [batch, heads, seq_len, head_dim]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Project and reshape for attention
        k = self.key(x)
        k = k.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply scaling pre-emptively for stable attention
        k = k * self.scale
        
        return k

class ValueModule(nn.Module):
    """Dedicated value projection module that handles only value transformations."""
    
    def __init__(self, dims: int, heads: int):
        """
        Args:
            dims: Input/output dimension size
            heads: Number of attention heads
        """
        super().__init__()
        
        assert dims % heads == 0, f"dims ({dims}) must be divisible by heads ({heads})"
        
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads
        
        # Only value projection
        self.value = nn.Linear(in_features=dims, out_features=dims)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(tensor=self.value.weight, std=0.02)
        if self.value.bias is not None:
            nn.init.zeros_(tensor=self.value.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Project input to value representation
        
        Args:
            x: Input tensor [batch, seq_len, dims]
            
        Returns:
            Value tensor [batch, heads, seq_len, head_dim]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Project and reshape for attention
        v = self.value(x)
        v = v.view(batch_size, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        
        return v

class KeyValueModule(nn.Module):
    """Dedicated key-value projection module that handles K and V transformations."""
    
    def __init__(self, dims: int, heads: int):
        """
        Args:
            dims: Input/output dimension size
            heads: Number of attention heads
        """
        super().__init__()
        
        # Use separate modules internally
        self.key_module = KeyModule(dims, heads)
        self.value_module = ValueModule(dims, heads)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Project input to key and value representations
        
        Args:
            x: Input tensor [batch, seq_len, dims]
            
        Returns:
            Tuple of (key, value) tensors, each shaped [batch, heads, seq_len, head_dim]
        """
        k = self.key_module(x)
        v = self.value_module(x)
        
        return k, v

class AttentionCombiner(nn.Module):
    """Combines separate Q and KV representations for attention computation."""
    
    def __init__(self, dims: int, heads: int):
        """
        Args:
            dims: Input/output dimension size
            heads: Number of attention heads
        """
        super().__init__()
        
        assert dims % heads == 0, f"dims ({dims}) must be divisible by heads ({heads})"
        
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads
        self.use_sdpa = True  # Use scaled dot product attention if available
        
        # Output projection
        self.out = nn.Linear(in_features=dims, out_features=dims)
        nn.init.normal_(tensor=self.out.weight, std=0.02)
        nn.init.zeros_(tensor=self.out.bias)
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute attention between provided q, k, v representations
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, key_len, head_dim]
            v: Value tensor [batch, heads, value_len, head_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, dims]
        """
        batch_size = q.size(0)
        seq_len = q.size(2)
        
        # Compute attention
        if self.use_sdpa:
            # Use PyTorch's optimized attention implementation if available
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask, 
                is_causal=(mask is not None and seq_len > 1)
            )
        else:
            # Manual implementation for older PyTorch versions
            # Note: No need for additional scaling here since we pre-scaled q and k
            attn = torch.matmul(q, k.transpose(-1, -2))
            
            if mask is not None:
                attn = attn + mask[:seq_len, :seq_len]
                
            attn = F.softmax(attn, dim=-1)
            attn_output = torch.matmul(attn, v)
        
        # Reshape and project output
        output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.dims)
        return self.out(output)


class SeparatedAttention(nn.Module):
    """Full attention implementation with completely separated Q, K, and V modules."""
    
    def __init__(self, dims: int, heads: int):
        """
        Args:
            dims: Input/output dimension size
            heads: Number of attention heads
        """
        super().__init__()
        
        self.query_module = QueryModule(dims, heads)
        self.key_module = KeyModule(dims, heads)
        self.value_module = ValueModule(dims, heads)
        self.combiner = AttentionCombiner(dims, heads)
    
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through separated attention modules
        
        Args:
            x: Input tensor for query projection [batch, seq_len, dims]
            xa: Optional cross-attention tensor [batch, kv_len, dims]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, None)
        """
        # Project query from input sequence
        q = self.query_module(x)
        
        # Project keys and values from input or cross-attention input
        kv_input = xa if xa is not None else x
        k = self.key_module(kv_input)
        v = self.value_module(kv_input)
        
        # Compute attention and return
        output = self.combiner(q, k, v, mask)
        
        # Return attention weights for later use if needed (None for now)
        return output, None


# Example usage with MHA integration
class MultiHeadAttentionWithSeparation(nn.Module):
    """Demonstrates how to use SeparatedAttention in larger architecture."""
    
    def __init__(self, dims: int, heads: int):
        super().__init__()
        self.attention = SeparatedAttention(dims, heads)
        self.layer_norm = nn.LayerNorm(dims)
    
    def forward(self, x, xa=None, mask=None):
        residual = x
        x = self.layer_norm(x)
        x, _ = self.attention(x, xa, mask)
        return x + residual


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
        
        assert dims % heads == 0, f"dims must be divisible by heads"  # noqa: F541
        assert heads % shared_keys == 0, f"heads must be divisible by shared_keys"  # noqa: F541
        
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


# if __name__ == "__main__":
#     demonstrate_advanced_patterns()

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

def demonstrate_advanced_patterns2():
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


# if __name__ == "__main__":
#     demonstrate_advanced_patterns2()


class attentionworm(nn.Module):
    def __init__(self, dims: int, head: int, max_dist: int = 512):
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.max_dist = max_dist
        self.scale = self.head_dim ** -0.5

    def calculate_attention(q, k, v, mask=None, temperature=1.0, use_sdpa=True, is_causal=True):
        batch_size = q.shape[0]
        ctx = q.shape[2]
        attn_mask = None
        if mask is not None:
            if mask.dim() <= 3:
                attn_mask = create_attention_mask(
                    batch_size=batch_size, 
                    ctx=ctx, 
                    is_causal=is_causal, 
                    padding_mask=mask if mask.dim() > 1 else None,
                    device=q.device)
            else:
                attn_mask = mask
        scaled_q = q
        if temperature != 1.0 and temperature > 0:
            scaled_q = q * (1.0 / temperature)**.5
        a = scaled_dot_product_attention(
            scaled_q, k, v, 
            attn_mask=attn_mask, 
            is_causal=is_causal if attn_mask is None else False)
        out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        return out, None

    class ProjectionModule(nn.Module):
        def __init__(self, dims: int, head: int, proj_type: str = "query", use_bias: bool = True):
            super().__init__()
            assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
            self.dims = dims
            self.head = head
            self.head_dim = dims // head
            self.proj_type = proj_type
            self.scale = self.head_dim ** -0.25 if proj_type != "value" else 1.0
            self.proj = Linear(in_features=dims, out_features=dims, bias=use_bias)
            self.init_weights()
            
        def init_weights(self):
            nn.init.normal_(tensor=self.proj.weight, std=0.02)
            if hasattr(self.proj, 'bias') and self.proj.bias is not None:
                nn.init.zeros_(tensor=self.proj.bias)
        
        def forward(self, x: Tensor) -> Tensor:
            batch, ctx = x.shape[:2]
            proj = self.proj(x)
            
            proj = proj.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
            if self.proj_type in ["query", "key"]:
                proj = proj * self.scale
            return proj

    def calculate_attention2(q, k, v, mask=None, temperature=1.0, use_sdpa=True, is_causal=True):

        if use_sdpa:
            try:
                if mask is not None:
                    if mask.dtype == torch.bool:
                        float_mask = torch.zeros_like(mask, dtype=torch.float)
                        float_mask = float_mask.masked_fill(mask, float('-inf'))
                        attn_output = scaled_dot_product_attention(
                            q, k, v, attn_mask=float_mask)
                    else:
                        attn_output = scaled_dot_product_attention(
                            q, k, v, attn_mask=mask, is_causal=is_causal)
                else:
                    attn_output = scaled_dot_product_attention(
                        q, k, v, attn_mask=None, is_causal=is_causal)
                return attn_output, None
            except RuntimeError:
                pass
        scale = 1.0 / temperature if temperature > 0 else 1.0
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        if mask is not None:
            if mask.dim() == 4:
                q_len, k_len = q.size(2), k.size(2)
                mask_q_len = min(mask.size(2), q_len)
                mask_k_len = min(mask.size(3), k_len)
                
                if mask.dtype == torch.bool:
                    mask_part = mask[:, :, :mask_q_len, :mask_k_len]
                    attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len].masked_fill(mask_part, float("-inf"))
                else:
                    attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len] + mask[:, :, :mask_q_len, :mask_k_len]
        attn = F.softmax(attn, dim=-1)
        
        if mask is not None and mask.dtype == torch.bool:
            binary_mask = (~mask).float()
            masked_attn = attn * binary_mask
            attn_sum = masked_attn.sum(dim=-1, keepdim=True)
            attn = masked_attn / (attn_sum + 1e-6)
        attn_output = torch.matmul(attn, v)
        return attn_output, attn

    class BaseAttention(nn.Module):
        """Base class for attention mechanisms with common functionality."""
        use_sdpa = True
        
        def __init__(self, dims: int, head: int, max_dist: int = 512):
            super().__init__()
            assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
            self.dims = dims
            self.head = head
            self.head_dim = dims // head
            self.max_dist = max_dist
            self.scale = self.head_dim ** -0.25
            
        def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
            return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()
            
        def _reshape_to_output(self, attn_output, batch, ctx):
            return attn_output.permute(0, 2, 1, 3).reshape(batch, ctx, self.dims)

    class AttentionCombiner(BaseAttention):
        def __init__(self, dims: int, head: int):
            super().__init__(dims, head)
            self.out = Linear(in_features=dims, out_features=dims)
            nn.init.normal_(tensor=self.out.weight, std=0.02)
            nn.init.zeros_(tensor=self.out.bias)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, is_causal=True) -> Tensor:
        if q.dim() == 3:
            batch, ctx, dims = q.shape
            self.scale = (dims // self.head) ** -0.5
            q = self._shape(q, ctx, batch)
            k = self._shape(k, k.size(1), batch)
            v = self._shape(v, v.size(1), batch)
        else:
            batch = q.size(0)
            ctx = q.size(2)
        attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa, is_causal=is_causal)
        output = self._reshape_to_output(attn_output, batch, ctx)
        return self.out(output)

    class AdaptiveUpdateAttention(BaseAttention):
        """Attention implementation with content-dependent update frequencies."""
        def __init__(self, dims: int, head: int, max_dist=512):
            super().__init__(dims, head, max_dist)
            
            self.query_module = ProjectionModule(dims, head, "query")
            self.key_module = ProjectionModule(dims, head, "key")
            self.value_module = ProjectionModule(dims, head, "value")
            self.combiner = AttentionCombiner(dims, head)
            self.key_update_predictor = nn.Sequential(
                Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid())
            self.value_update_predictor = nn.Sequential(
                Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid())

            self.update_threshold = 0.5
            self.stored_key_cache = None
            self.stored_value_cache = None

        def should_update_key(self, x: torch.Tensor) -> torch.Tensor:
            """Predict whether the key should be updated based on content."""
            avg_rep = x.mean(dim=1)
            return self.key_update_predictor(avg_rep) > self.update_threshold

        def should_update_value(self, x: torch.Tensor) -> torch.Tensor:
            """Predict whether the value should be updated based on content."""
            avg_rep = x.mean(dim=1)
            return self.value_update_predictor(avg_rep) > self.update_threshold

        def forward(self, x, xa=None, mask=None, kv_cache=None, is_causal=True):
            """Process inputs with adaptive update mechanism."""
            batch, ctx, _ = x.shape
            q = self.query_module(x)
            kv_input = xa if xa is not None else x
            device = kv_input.device  # noqa: F841

            if kv_cache is None:
                k = self.key_module(kv_input)
                v = self.value_module(kv_input)
                
                self.stored_key_cache = k
                self.stored_value_cache = v
            else:
                update_k = self.should_update_key(kv_input)
                update_v = self.should_update_value(kv_input)
                if update_k.any():
                    new_k = self.key_module(kv_input)
                    if self.stored_key_cache is not None:
                        update_mask = update_k.view(-1, 1, 1, 1).expand_as(self.stored_key_cache)
                        k = torch.where(update_mask, new_k, self.stored_key_cache)
                    else:
                        k = new_k
                else:
                    k = self.stored_key_cache if self.stored_key_cache is not None else self.key_module(kv_input)
                if update_v.any():
                    new_v = self.value_module(kv_input)
                    if self.stored_value_cache is not None:
                        update_mask = update_v.view(-1, 1, 1, 1).expand_as(self.stored_value_cache)
                        v = torch.where(update_mask, new_v, self.stored_value_cache)
                    else:
                        v = new_v
                else:
                    v = self.stored_value_cache if self.stored_value_cache is not None else self.value_module(kv_input)
                self.stored_key_cache = k
                self.stored_value_cache = v
            output = self.combiner(q, k, v, mask=mask, is_causal=is_causal)
            return output

    class Refiner:
        """Q-learning based refiner for adaptive attention span."""
        def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
            self.states = states
            self.actions = actions
            self.R = {}
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.default_value = 0.0

        def get_value(self, state, action):
            return self.R.get((state, action), self.default_value)

        def set_value(self, state, action, value):
            self.R[(state, action)] = value

        def choose_action(self, state):
            if np.random.random() < self.epsilon:
                return np.random.randint(self.actions)
            else:
                action_values = [self.get_value(state, a) for a in range(self.actions)]
                return np.argmax(action_values)

        def update(self, state, action, reward, next_state):
            next_values = [self.get_value(next_state, a) for a in range(self.actions)]
            best_next_value = max(next_values)

            old_value = self.get_value(state, action)
            td_target = reward + self.gamma * best_next_value
            td_error = td_target - old_value
            new_value = old_value + self.alpha * td_error
            self.set_value(state, action, new_value)

    class Predictor(nn.Module):
        """Neural predictor for span scale estimation."""
        def __init__(self, dims):
            super().__init__()
            self.linear = Linear(in_features=dims, out_features=1)
            nn.init.xavier_normal_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        def forward(self, global_out):
            if global_out.dim() > 2:
                global_out = global_out.mean(dim=1)
            scale = torch.sigmoid(self.linear(global_out))
            return scale

    class AdaptiveSpan(BaseAttention):
        """Attention with adaptive span size."""
        def __init__(self, dims, head, max_dist, sharpen=True, temp_scale=0.01):
            super().__init__(dims, head, max_dist)
            self.sharpen = sharpen
            self.temp_scale = temp_scale
            self.span_scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, query, key, value, max_dist=None, max_span=None, span_scale=None, is_causal=True):
            if max_dist is None:
                max_dist = self.max_dist
            if max_span is None:
                max_span = query.shape[1]
            if span_scale is None:
                span_scale = self.span_scale
                
            span_mean = span_scale.mean().item()
            span_len = min(int(max_span * span_mean), query.shape[1], key.shape[1], value.shape[1])
            eff_span = min(span_len, max_dist)
            
            if eff_span == 0:
                batch = query.shape[0]
                return (torch.zeros(batch, eff_span, self.dims, device=query.device), None)
                
            q_span = query[:, :eff_span, :]
            k_span = key[:, :eff_span, :]
            v_span = value[:, :eff_span, :]

            batch = q_span.shape[0]

            q = self._shape(q_span, q_span.size(1), batch)
            k = self._shape(k_span, k_span.size(1), batch)
            v = self._shape(v_span, v_span.size(1), batch)

            temperature = (1.0 + self.temp_scale * (1.0 - span_mean)
                if self.sharpen
                else 0.5 + self.temp_scale * span_mean)
            
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                attn_output, weights = calculate_attention(
                    q, k, v, None, temperature, BaseAttention.use_sdpa, is_causal=is_causal)
                out = self._reshape_to_output(attn_output, batch, eff_span)
            return out, weights

    class MyelinatedLayer(BaseAttention):
        def __init__(self, dims, head, layerA=3, sparsity_threshold=0.1, max_dist=512):
            super().__init__(dims, head, max_dist)
            self.layers = nn.ModuleList()
            self.layerA = layerA
            self.sparsity_threshold = sparsity_threshold
            self.max_dist = max_dist
            
            self.node_predictors = nn.ModuleList([
                nn.Sequential(LayerNorm(dims),
                            Linear(dims, 1),
                            nn.Sigmoid()) for _ in range(layerA)])
            
            for i in range(layerA):
                self.layers.append(nn.ModuleDict({
                    'ln': LayerNorm(dims),
                    'gate': nn.Sequential(Linear(dims, 1), nn.Sigmoid()),
                    'adapter': Linear(dims, dims) if i % 2 == 0 else None
                }))
            self.policy_net = nn.Sequential(Linear(dims, 128), nn.ReLU(), Linear(128, 3))
            self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]))
            
            mlp = dims * 4
            self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.mlp = nn.Sequential(Linear(dims, mlp), nn.GELU(), Linear(mlp, dims))
            self.mlp_ln = LayerNorm(dims)
            
            self.working_memory = nn.Parameter(torch.zeros(1, 1, dims))
            self.memory_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.last_memory_gate_values = None

        def compute_attention(self, norm_x, mask=None, kv_cache=None, is_causal=True):
            """Compute attention with adaptive span and content-dependent updates."""
            batch, ctx = norm_x.shape[:2]
            
            q = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)
            k = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)
            v = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)

            attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa, is_causal=is_causal)
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch, ctx, -1)
            return attn_output

        def predict_node_importance(self, x, layer_idx):
            """Dynamically determine if processing should occur at this node."""
            importance = self.node_predictors[layer_idx](x)
            return (importance > self.sparsity_threshold).float()

        def decide_jump(self, policy, jump_weights, i, layerA, x, original_x, working_memory):
            """Decide whether to jump layers based on the policy network."""
            jump_prob = policy[:, 1] if i < layerA - 1 else torch.zeros_like(policy[:, 1])
            should_jump = (torch.rand_like(jump_prob) < jump_prob).any()
            if should_jump:
                jump_length = torch.multinomial(policy, 1)[:, 0].max().item() + 1
                i_next = min(i + jump_length, layerA - 1)
                skip_weight = jump_weights[min(jump_length - 1, 2)]
                x = x + skip_weight * original_x + (1 - skip_weight) * working_memory
                return x, i_next
            return x, i + 1

        def forward(self, x, xa=None, mask=None, kv_cache=None, is_causal=True):
            batch, ctx = x.shape[:2]
            working_memory = self.working_memory.expand(batch, -1, -1)
            original_x = x
            pooled_representation = x.mean(dim=1, keepdim=False)
            policy_logits = self.policy_net(pooled_representation)
            policy = F.softmax(policy_logits, dim=-1)
            jump_history = []
            memory_gate = torch.zeros(batch, 1, 1, device=x.device)
            
            i = 0
            while i < self.layerA:
                layer = self.layers[i]
                node_importance = self.predict_node_importance(x, i)
                print(f"Node importance (Layer {i}): {node_importance}")

                if node_importance.mean() < 0.2 and i > 0:
                    i += 1
                    jump_history.append(i)
                    continue
                norm_x = layer['ln'](x)
                attn_mask = mask * node_importance.squeeze(-1).unsqueeze(1) if mask is not None else node_importance.squeeze(-1).unsqueeze(1)
                
                if node_importance.mean() > 0.3:
                    attn_output = self.compute_attention(norm_x, mask=attn_mask, kv_cache=kv_cache)
                    print(f"Attention output (Layer {i}): {attn_output}")
                    
                    if layer['adapter'] is not None:
                        attn_output = layer['adapter'](attn_output)
                    gate_value = layer['gate'](norm_x)
                    x = x + gate_value * attn_output
                    print(f"Updated representation (Layer {i}): {x}")
                    
                    memory_gate = self.memory_gate(x.mean(dim=1, keepdim=True))
                    mean_x = x.mean(dim=1, keepdim=True)
                    working_memory = memory_gate * working_memory + (1 - memory_gate) * mean_x
                    print(f"Memory gate value: {memory_gate}")
                
                x, i = self.decide_jump(policy, self.jump_weights, i, self.layerA, x, original_x, working_memory)
                jump_history.append(i)

            self.last_memory_gate_values = memory_gate.detach().clone()
            print(f"Jump history: {jump_history}")
            mlp_importance = self.mlp_gate(x)
            mlp_output = self.mlp(self.mlp_ln(x))
            x = x + mlp_importance * mlp_output
            print(f"Final output: {x}")
            return x

    class IntegratedAttention(nn.Module):
        def __init__(self, dims, head, max_dist=512, win_size=256, max_span=384, temp_scale=0.01):
            super().__init__()
            self.head = head
            self.max_dist = max_dist

            self.dims = dims
            self.max_span = max_span
            self.sliding_window = win_size
            self.temp_scale = temp_scale
            self.sharpen = True
            self.head_dim = dims // head

            self.refiner = Refiner(states=10000, actions=10, alpha=0.1, gamma=0.9, epsilon=0.1)
            self.span_pred = Predictor(dims=dims)
            
            self.attn_local = AdaptiveSpan(
                dims=dims, head=head, max_dist=max_dist, sharpen=True, temp_scale=temp_scale)
            
            self.attn_global = MyelinatedLayer(dims=dims, head=head)
            self.cross_attn = MyelinatedLayer(dims=dims, head=head)

            # self.attn_global = AdaptiveUpdateAttention(dims=dims, head=head, max_dist=max_dist)
            # self.cross_attn = AttentionCombiner(dims=dims, head=head)

            self.self_projection = Linear(in_features=2 * dims, out_features=dims)
            self.cross_projection = Linear(in_features=dims, out_features=dims)
            
            self.ln_a = LayerNorm(normalized_shape=dims)
            self.ln_b = LayerNorm(normalized_shape=dims)
            self.ln_cross = LayerNorm(normalized_shape=dims)

            mask = torch.empty(max_span, max_span).fill_(float("-inf")).triu_(diagonal=1)
            self.register_buffer("causal_mask", mask, persistent=False)
            self.register_buffer("window_mask", None, persistent=False)
            self.register_buffer("threshold", torch.tensor(1e-4), persistent=False)
            self.register_buffer("s_factor", torch.tensor(0.1), persistent=False)

        def forward(self, x, xa=None, mask=None, kv_cache=None, is_causal=True):
            batch, ctx = x.shape[:2]
            
            if xa is not None:
                x_norm = self.ln_cross(x)
                
                cross_out = self.cross_attn(
                    q=x_norm, k=xa, v=xa, mask=mask)
                return self.cross_projection(cross_out)
            
            local = self.ln_a(x)
            globe = self.ln_b(x)

            globe_out = self.attn_global(globe, xa=None, mask=mask, kv_cache=kv_cache, is_causal=is_causal)
            globe_out = self.cross_projection(globe_out)
            
            freq_scale = self.span_pred(globe_out)
            state = self.extract(local)
            action = self.refiner.choose_action(state=state)
            refine = self.action_scale(action=action)
            span_scale = torch.clamp(freq_scale * refine, min=0.0, max=1.0)
            span_mean = span_scale.mean().item()

            with torch.no_grad():
                current_win_size = max(1, int(self.sliding_window * span_mean))
                current_span_len = max(1, int(self.max_span * span_mean))
                effective_max = min(self.max_dist, local.size(1))
                local_max = min(self.max_dist, current_span_len, current_win_size)
                globe_max = effective_max

            self.attn_local.max_dist = local_max
            self.attn_global.max_dist = globe_max

            local_out = self.slide_win(
                x=local,
                win_size=current_win_size,
                span_len=current_span_len,
                span_scale=span_scale,
                mask=mask,
            )
            
            with torch.no_grad():
                quality = self.quality(output=local_out)
                next_state = self.extract(local_out)
                self.refiner.update(
                    state=state, action=action, reward=quality, next_state=next_state)
            
            combined = torch.cat([local_out, globe_out], dim=-1)
            return self.self_projection(combined)

        def quality(self, output):
            """Calculate quality metric for reinforcement learning."""
            with torch.no_grad():
                safe_output = torch.clamp(output, min=1e-10)
                entropy = -(safe_output * torch.log(safe_output)).sum(-1).mean()
                coverage = (output > 0.01).float().mean()
                return float(coverage - 0.1 * entropy)

        def extract(self, x):
            """Extract state features for RL agent."""
            with torch.no_grad():
                pooled = x.reshape(-1, self.dims)
                meadims = pooled.mean(dim=0)
                var_state = pooled.var(dim=0, unbiased=False)
                state = torch.cat([meadims, var_state])
                state_id = self.discretize(state.cpu().numpy())
            return state_id

        def discretize(self, state):
            """Convert continuous state to discrete state ID."""
            bins = np.linspace(-1, 1, num=10)
            state_discrete = np.digitize(state, bins)
            state_hash = sum(val * (10**i) for i, val in enumerate(state_discrete[:20]))
            state_id = int(state_hash % (self.refiner.states - 1))
            return state_id

        def action_scale(self, action):
            """Convert discrete action to continuous scale factor."""
            span_value = action / (self.refiner.actions - 1)
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            span_scale = torch.tensor([span_value], device=device, dtype=dtype)
            return span_scale

        def _focus(self, query, key, value, span_scale, mask=None):
            """Iterative attention refinement with zero-padding for invalid tokens."""
            max_iterations = 10
            iteration = 0
            prev_attn = torch.zeros_like(input=query)
            attn_out = torch.zeros_like(input=query)
            attn_weights = None

            threshold = self.threshold.item()
            s_factor = self.s_factor.item()

            while iteration < max_iterations:
                span_len = int(self.max_span * span_scale.mean().item())
                span_len = min(span_len, query.size(1), key.size(1), value.size(1))
                eff_span = min(span_len, self.max_dist)

                if eff_span == 0:
                    break

                q_span = query[:, :eff_span, :]
                k_span = key[:, :eff_span, :]
                v_span = value[:, :eff_span, :]

                batch, ctx, dims = q_span.size()
                
                q = q_span.view(batch, ctx, self.head, -1).transpose(1, 2)
                k = k_span.view(batch, ctx, self.head, -1).transpose(1, 2)
                v = v_span.view(batch, ctx, self.head, -1).transpose(1, 2)

                if self.sharpen:
                    temperature = 1.0 + self.temp_scale * (1.0 - span_scale.mean().item())
                else:
                    temperature = 0.5 + self.temp_scale * span_scale.mean().item()  # noqa: F841
                
                scale = (dims // self.head) ** -0.5
                attn = torch.matmul(q, k.transpose(-1, -2)) * scale
                
                if mask is not None:
                    if mask.dim() == 4:
                        q_len, k_len = q.size(2), k.size(2)
                        mask_q_len = min(mask.size(2), q_len)
                        mask_k_len = min(mask.size(3), k_len)
                        mask_part = mask[:, :, :mask_q_len, :mask_k_len]
                        if mask_part.dtype == torch.bool:
                            attn_part = attn[:, :, :mask_q_len, :mask_k_len]
                            masked_attn_part = attn_part.masked_fill(mask_part, float("-inf"))
                            new_attn = attn.clone()
                            new_attn[:, :, :mask_q_len, :mask_k_len] = masked_attn_part
                            attn = new_attn
                        else:
                            attn_part = attn[:, :, :mask_q_len, :mask_k_len]
                            masked_attn_part = attn_part + mask_part
                            new_attn = attn.clone()
                            new_attn[:, :, :mask_q_len, :mask_k_len] = masked_attn_part
                            attn = new_attn
                
                attn = F.softmax(attn, dim=-1)
                
                if mask is not None and mask.dtype == torch.bool:
                    q_len, k_len = q.size(2), k.size(2)
                    mask_q_len = min(mask.size(2), q_len)
                    mask_k_len = min(mask.size(3), k_len)
                    binary_mask = (~mask[:, :, :mask_q_len, :mask_k_len]).float()
                    attn_part = attn[:, :, :mask_q_len, :mask_k_len]
                    masked_attn_part = attn_part * binary_mask
                    attn_sum = masked_attn_part.sum(dim=-1, keepdim=True)
                    normalized_attn_part = masked_attn_part / (attn_sum + 1e-6)
                    new_attn = attn.clone()
                    new_attn[:, :, :mask_q_len, :mask_k_len] = normalized_attn_part
                    attn = new_attn
                    
                attn_output = torch.matmul(attn, v)
                attn_out = attn_output.transpose(1, 2).contiguous().view(batch, ctx, -1)
                diff = torch.abs(attn_out - prev_attn).mean()
                dynamic_threshold = threshold + s_factor * diff
                if diff < dynamic_threshold:
                    break

                prev_attn = attn_out.clone()
                query = query + attn_out
                iteration += 1
            return attn_out, attn_weights


        def slide_win(self, x, win_size, span_len, span_scale, mask=None):
            """Process input with sliding window attention."""
            batch, ctx, dims = x.size()
            num_windows = (ctx + win_size - 1) // win_size
            output = torch.zeros_like(x)

            for i in range(num_windows):
                start_idx = i * win_size
                end_idx = min((i + 1) * win_size, ctx)
                window_size = end_idx - start_idx  # noqa: F841

                key_start = max(0, start_idx - span_len + win_size)
                key_end = min(start_idx + span_len, ctx)

                query = x[:, start_idx:end_idx, :]
                key = x[:, key_start:key_end, :]
                value = key

                window_mask = None
                if mask is not None:
                    if mask.dim() == 4:
                        window_mask = mask[:, :, start_idx:end_idx, key_start:key_end]
                        
                        if window_mask.size(1) == 1:
                            window_mask = window_mask.expand(-1, self.head, -1, -1)

                attn_out, _ = self._focus(
                    query=query,
                    key=key,
                    value=value,
                    span_scale=span_scale,
                    mask=window_mask)
                output[:, start_idx:end_idx, :] = attn_out
            return output


    class AttentionWrapper(nn.Module):
        """Wrapper to standardize attention layer interfaces"""
        
        def __init__(self, attention_layer):
            super().__init__()
            self.attention = attention_layer
            
        def forward(
            self,
            x: Tensor,
            xa: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            kv_cache: Optional[dict] = None
        ) -> Tuple[Tensor, Optional[Tensor]]:
            result = self.attention(x, xa, mask, kv_cache)
            
            if isinstance(result, tuple):
                return result
            else:
                return (result, None)

#version 4
class version4 (nn.Module):
        
    class ProjectionModule(nn.Module):
        """Unified projection module that handles query, key, and value transformations."""
        def __init__(self, dims: int, head: int, proj_type: str = "query", use_bias: bool = True):
            super().__init__()
            assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
            self.dims = dims
            self.head = head
            self.head_dim = dims // head
            self.proj_type = proj_type
            self.scale = self.head_dim ** -0.25 if proj_type != "value" else 1.0
            self.proj = Linear(in_features=dims, out_features=dims, bias=use_bias)
            self.init_weights()
            
        def init_weights(self):
            nn.init.normal_(tensor=self.proj.weight, std=0.02)
            if hasattr(self.proj, 'bias') and self.proj.bias is not None:
                nn.init.zeros_(tensor=self.proj.bias)
        
        def forward(self, x: Tensor) -> Tensor:
            batch, ctx = x.shape[:2]
            proj = self.proj(x)
            
            proj = proj.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
            if self.proj_type in ["query", "key"]:
                proj = proj * self.scale
            return proj

    def calculate_attention(q, k, v, mask=None, temperature=1.0, use_sdpa=True):
        """Attention calculation with zero-padding for invalid tokens."""
        if use_sdpa:
            try:
                if mask is not None:
                    if mask.dtype == torch.bool:
                        float_mask = torch.zeros_like(mask, dtype=torch.float)
                        float_mask = float_mask.masked_fill(mask, float('-inf'))
                        attn_output = scaled_dot_product_attention(
                            q, k, v, attn_mask=float_mask)
                    else:
                        attn_output = scaled_dot_product_attention(
                            q, k, v, attn_mask=mask)
                else:
                    attn_output = scaled_dot_product_attention(
                        q, k, v, attn_mask=None)
                return attn_output, None
            except RuntimeError:
                pass
        scale = 1.0 / temperature if temperature > 0 else 1.0
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        if mask is not None:
            if mask.dim() == 4:
                q_len, k_len = q.size(2), k.size(2)
                mask_q_len = min(mask.size(2), q_len)
                mask_k_len = min(mask.size(3), k_len)
                
                if mask.dtype == torch.bool:
                    mask_part = mask[:, :, :mask_q_len, :mask_k_len]
                    attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len].masked_fill(
                        mask_part, float("-inf")
                    )
                else:
                    attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len] + mask[:, :, :mask_q_len, :mask_k_len]
        attn = F.softmax(attn, dim=-1)
        
        if mask is not None and mask.dtype == torch.bool:
            binary_mask = (~mask).float()
            masked_attn = attn * binary_mask
            attn_sum = masked_attn.sum(dim=-1, keepdim=True)
            attn = masked_attn / (attn_sum + 1e-6)
        attn_output = torch.matmul(attn, v)
        return attn_output, attn

    class BaseAttention(nn.Module):
        """Base class for attention mechanisms with common functionality."""
        use_sdpa = True
        
        def __init__(self, dims: int, head: int, max_dist: int = 512):
            super().__init__()
            assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
            self.dims = dims
            self.head = head
            self.head_dim = dims // head
            self.max_dist = max_dist
            self.scale = self.head_dim ** -0.25
            
        def _reshape_to_output(self, attn_output, batch, ctx):
            """Reshape attention output to original dimensions."""
            return attn_output.permute(0, 2, 1, 3).reshape(batch, ctx, self.dims)

    class AttentionCombiner(BaseAttention):
        """Combines separate Q and KV representations for attention computation."""
        def __init__(self, dims: int, head: int):
            super().__init__(dims, head)
            self.out = Linear(in_features=dims, out_features=dims)
            nn.init.normal_(tensor=self.out.weight, std=0.02)
            nn.init.zeros_(tensor=self.out.bias)

        def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
            """Processes and combines attention inputs."""
            if q.dim() == 3:
                batch, ctx, _ = q.shape
                q = q.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
                k = k.view(batch, k.size(1), self.head, self.head_dim).permute(0, 2, 1, 3)
                v = v.view(batch, v.size(1), self.head, self.head_dim).permute(0, 2, 1, 3)
            else:
                batch = q.size(0)
                ctx = q.size(2)
            attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa)
            output = self._reshape_to_output(attn_output, batch, ctx)
            return self.out(output)

    class AdaptiveUpdateAttention(BaseAttention):
        """Attention implementation with content-dependent update frequencies."""
        def __init__(self, dims: int, head: int, max_dist=512):
            super().__init__(dims, head, max_dist)
            
            self.query_module = ProjectionModule(dims, head, "query")
            self.key_module = ProjectionModule(dims, head, "key")
            self.value_module = ProjectionModule(dims, head, "value")
            self.combiner = AttentionCombiner(dims, head)
            self.key_update_predictor = nn.Sequential(
                Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid())
            self.value_update_predictor = nn.Sequential(
                Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid())

            self.update_threshold = 0.5
            self.stored_key_cache = None
            self.stored_value_cache = None

        def should_update_key(self, x: torch.Tensor) -> torch.Tensor:
            """Predict whether the key should be updated based on content."""
            avg_rep = x.mean(dim=1)
            return self.key_update_predictor(avg_rep) > self.update_threshold

        def should_update_value(self, x: torch.Tensor) -> torch.Tensor:
            """Predict whether the value should be updated based on content."""
            avg_rep = x.mean(dim=1)
            return self.value_update_predictor(avg_rep) > self.update_threshold

        def forward(self, x, xa=None, mask=None, kv_cache=None):
            """Process inputs with adaptive update mechanism."""
            batch, ctx, _ = x.shape
            
            q = self.query_module(x)
            
            kv_input = xa if xa is not None else x
            device = kv_input.device  # noqa: F841

            if kv_cache is None:
                k = self.key_module(kv_input)
                v = self.value_module(kv_input)
                
                self.stored_key_cache = k
                self.stored_value_cache = v
            else:
                update_k = self.should_update_key(kv_input)
                update_v = self.should_update_value(kv_input)
                
                if update_k.any():
                    new_k = self.key_module(kv_input)
                    if self.stored_key_cache is not None:
                        update_mask = update_k.view(-1, 1, 1, 1).expand_as(self.stored_key_cache)
                        k = torch.where(update_mask, new_k, self.stored_key_cache)
                    else:
                        k = new_k
                else:
                    k = self.stored_key_cache if self.stored_key_cache is not None else self.key_module(kv_input)
                
                if update_v.any():
                    new_v = self.value_module(kv_input)
                    if self.stored_value_cache is not None:
                        update_mask = update_v.view(-1, 1, 1, 1).expand_as(self.stored_value_cache)
                        v = torch.where(update_mask, new_v, self.stored_value_cache)
                    else:
                        v = new_v
                else:
                    v = self.stored_value_cache if self.stored_value_cache is not None else self.value_module(kv_input)
                
                self.stored_key_cache = k
                self.stored_value_cache = v
            
            output = self.combiner(q, k, v, mask=mask)
            
            return output

    class Refiner:
        """Q-learning based refiner for adaptive attention span."""
        def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
            self.states = states
            self.actions = actions
            self.R = {}
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.default_value = 0.0

        def get_value(self, state, action):
            return self.R.get((state, action), self.default_value)

        def set_value(self, state, action, value):
            self.R[(state, action)] = value

        def choose_action(self, state):
            if np.random.random() < self.epsilon:
                return np.random.randint(self.actions)
            else:
                action_values = [self.get_value(state, a) for a in range(self.actions)]
                return np.argmax(action_values)

        def update(self, state, action, reward, next_state):
            next_values = [self.get_value(next_state, a) for a in range(self.actions)]
            best_next_value = max(next_values)

            old_value = self.get_value(state, action)
            td_target = reward + self.gamma * best_next_value
            td_error = td_target - old_value
            new_value = old_value + self.alpha * td_error
            self.set_value(state, action, new_value)

    class Predictor(nn.Module):
        """Neural predictor for span scale estimation."""
        def __init__(self, dims):
            super().__init__()
            self.linear = Linear(in_features=dims, out_features=1)
            nn.init.xavier_normal_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        def forward(self, global_out):
            if global_out.dim() > 2:
                global_out = global_out.mean(dim=1)
            scale = torch.sigmoid(self.linear(global_out))
            return scale

    class AdaptiveSpan(BaseAttention):
        """Attention with adaptive span size."""
        def __init__(self, dims, head, max_dist, sharpen=True, temp_scale=0.01):
            super().__init__(dims, head, max_dist)
            self.sharpen = sharpen
            self.temp_scale = temp_scale
            self.span_scale = nn.Parameter(torch.tensor(1.0))


        def forward(self, query, key, value, max_dist=None, max_span=None, span_scale=None):
            if max_dist is None:
                max_dist = self.max_dist
            if max_span is None:
                max_span = query.shape[1]
            if span_scale is None:
                span_scale = self.span_scale
                
            span_mean = span_scale.mean().item()
            span_len = min(int(max_span * span_mean), query.shape[1], key.shape[1], value.shape[1])
            eff_span = min(span_len, max_dist)
            
            if eff_span == 0:
                batch = query.shape[0]
                return (torch.zeros(batch, eff_span, self.dims, device=query.device), None)
                
            q_span = query[:, :eff_span, :]
            k_span = key[:, :eff_span, :]
            v_span = value[:, :eff_span, :]

            batch = q_span.shape[0]

            reshape_dims = (batch, -1, self.head, self.head_dim)
            q = q_span.view(*reshape_dims).permute(0, 2, 1, 3)
            k = k_span.view(*reshape_dims).permute(0, 2, 1, 3)
            v = v_span.view(*reshape_dims).permute(0, 2, 1, 3)

            temperature = (
                1.0 + self.temp_scale * (1.0 - span_mean)
                if self.sharpen
                else 0.5 + self.temp_scale * span_mean
            )
            
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                attn_output, weights = calculate_attention(
                    q, k, v, None, temperature, BaseAttention.use_sdpa
                )
                out = self._reshape_to_output(attn_output, batch, eff_span)

            return out, weights

    class MyelinatedLayer(BaseAttention):
        def __init__(self, dims, head, layerAs=6, sparsity_threshold=0.1):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layerAs = layerAs
            self.sparsity_threshold = sparsity_threshold
            
            self.shared_head = AdaptiveSpan(dims, head)
            
            self.node_predictors = nn.ModuleList([
                nn.Sequential(
                    LayerNorm(dims),
                    Linear(dims, 1),
                    nn.Sigmoid()
                ) for _ in range(layerAs)
            ])
            
            for i in range(layerAs):
                self.layers.append(nn.ModuleDict({
                    'ln': LayerNorm(dims),
                    'gate': nn.Sequential(Linear(dims, 1), nn.Sigmoid()),
                    'adapter': Linear(dims, dims) if i % 2 == 0 else None
                }))
            
            self.policy_net = nn.Sequential(
                Linear(dims, 128),
                nn.ReLU(),
                Linear(128, 3)
            )
            
            self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]))
            
            n_mlp = dims * 4
            self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.mlp = nn.Sequential(Linear(dims, n_mlp), nn.GELU(), Linear(n_mlp, dims))
            self.mlp_ln = LayerNorm(dims)
            
            self.working_memory = nn.Parameter(torch.zeros(1, 1, dims))
            self.memory_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            
        def shared_head(self, norm_x, mask=None, kv_cache=None):
            batch_size, seq_len = norm_x.shape[:2]
            
            q = norm_x.view(batch_size, seq_len, self.head, -1).transpose(1, 2)
            k = norm_x.view(batch_size, seq_len, self.head, -1).transpose(1, 2)
            v = norm_x.view(batch_size, seq_len, self.head, -1).transpose(1, 2)
            
            attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa)
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            return attn_output


        def predict_node_importance(self, x, layer_idx):
            """Dynamically determine if processing should occur at this node"""
            importance = self.node_predictors[layer_idx](x)
            return (importance > self.sparsity_threshold).float()
        
        def forward(self, x, xa=None, mask=None, kv_cache=None):
            batch_size, seq_len = x.shape[:2]
            
            working_memory = self.working_memory.expand(batch_size, -1, -1)
            
            original_x = x
            
            pooled_representation = x.mean(dim=1)
            policy_logits = self.policy_net(pooled_representation)
            policy = F.softmax(policy_logits, dim=-1)
            
            jump_history = []
            i = 0
            while i < self.layerAs:
                layer = self.layers[i]
                
                node_importance = self.predict_node_importance(x, i)
                
                if node_importance.mean() < 0.2 and i > 0:
                    i += 1
                    jump_history.append(i)
                    continue
                    
                norm_x = layer['ln'](x)
                
                attn_mask = mask
                if mask is None:
                    attn_mask = node_importance.squeeze(-1).unsqueeze(1).expand(-1, seq_len, -1)
                else:
                    attn_mask = mask * node_importance.squeeze(-1).unsqueeze(1).expand(-1, seq_len, -1)
                    
                if node_importance.mean() > 0.3:
                    attn_output = self.shared_head(norm_x, mask=attn_mask, kv_cache=kv_cache)[0]
                    
                    if layer['adapter'] is not None:
                        attn_output = layer['adapter'](attn_output)
                    
                    gate_value = layer['gate'](norm_x).unsqueeze(-1)
                    x = x + gate_value * attn_output
                    
                    memory_gate = self.memory_gate(x)
                    working_memory = memory_gate * working_memory + (1 - memory_gate) * x.mean(dim=1, keepdim=True)
                
                jump_prob = policy[:, 1] if i < self.layerAs - 1 else torch.zeros_like(policy[:, 1])
                should_jump = (torch.rand_like(jump_prob) < jump_prob).any()
                
                if should_jump:
                    jump_length = torch.multinomial(policy, 1)[:, 0].max().item() + 1
                    
                    i_next = min(i + jump_length, self.layerAs - 1)
                    skip_weight = self.jump_weights[min(jump_length-1, 2)]
                    
                    x = x + skip_weight * original_x + (1-skip_weight) * working_memory
                    
                    i = i_next
                    jump_history.append(i)
                else:
                    i += 1
            
            mlp_importance = self.mlp_gate(x)
            mlp_output = self.mlp(self.mlp_ln(x))
            x = x + mlp_importance * mlp_output
            
            return x, {'jumps': jump_history}

    class IntegratedAttention(nn.Module):
        """Combines local adaptive span and global content-dependent attention with RL-based adaptation."""
        def __init__(self, dims, head, max_dist=512, win_size=256, max_span=384, temp_scale=0.01):
            super().__init__()
            self.head = head
            self.max_dist = max_dist
            self.dims = dims
            self.max_span = max_span
            self.sliding_window = win_size
            self.temp_scale = temp_scale
            self.sharpen = True
            self.head_dim = dims // head

            self.refiner = Refiner(
                states=10000, actions=10, alpha=0.1, gamma=0.9, epsilon=0.1
            )
            
            self.span_pred = Predictor(dims=dims)
            self.attn_local = AdaptiveSpan(
                dims=dims, head=head, max_dist=max_dist, sharpen=True, temp_scale=temp_scale
            )
            
            self.attn_self = AdaptiveUpdateAttention(dims=dims, head=head, max_dist=max_dist)
            
            self.attn_cross = AdaptiveUpdateAttention(dims=dims, head=head, max_dist=max_dist)
            
            self.projection = Linear(in_features=2 * dims, out_features=dims)
            self.cross_projection = Linear(in_features=dims, out_features=dims)
            
            self.ln_a = LayerNorm(normalized_shape=dims)
            self.ln_b = LayerNorm(normalized_shape=dims)
            self.ln_cross = LayerNorm(normalized_shape=dims)

            mask = torch.empty(max_span, max_span).fill_(float("-inf")).triu_(diagonal=1)
            self.register_buffer("mask", mask, persistent=False)
            self.register_buffer("window_mask", None, persistent=False)
            self.register_buffer("threshold", torch.tensor(1e-4), persistent=False)
            self.register_buffer("s_factor", torch.tensor(0.1), persistent=False)

        def forward(self, x, xa=None, mask=None, kv_cache=None):
            """
            Process input with integrated attention mechanisms.
            
            Args:
                x: Input tensor from the current layer (decoder hidden states)
                xa: Optional encoder outputs for cross-attention
                mask: Attention mask (causal for self-attention)
                kv_cache: Optional key-value cache for efficient generation
                
            Returns:
                Processed tensor after attention
            """
            batch_size, seq_len = x.shape[:2]
            
            if mask is None or mask.dim() != 4:
                mask = create_attention_mask(
                    batch_size=batch_size, 
                    seq_len=seq_len,
                    is_causal=True,
                    device=x.device)
            
            if xa is not None:
                x_norm = self.ln_cross(x)
                xa_norm = self.ln_a(xa)
                
                cross_out = self.attn_cross(
                    x=x_norm,
                    xa=xa_norm,
                    mask=None,
                    kv_cache=kv_cache
                )
                
                return self.cross_projection(cross_out)
            
            local = self.ln_a(x)
            globe = self.ln_b(x)

            globe_out = self.attn_self(globe, mask=mask, kv_cache=kv_cache)
            
            freq_scale = self.span_pred(globe_out)
            state = self.extract(local)
            action = self.refiner.choose_action(state=state)
            refine = self.action_scale(action=action)
            span_scale = torch.clamp(freq_scale * refine, min=0.0, max=1.0)
            span_mean = span_scale.mean().item()

            with torch.no_grad():
                current_win_size = max(1, int(self.sliding_window * span_mean))
                current_span_len = max(1, int(self.max_span * span_mean))

                effective_max = min(self.max_dist, local.size(1))
                local_max = min(self.max_dist, current_span_len, current_win_size)
                globe_max = effective_max

            self.attn_local.max_dist = local_max
            self.attn_self.max_dist = globe_max
            self.attn_cross.max_dist = globe_max

            local_out = self.slide_win(
                x=local,
                win_size=current_win_size,
                span_len=current_span_len,
                span_scale=span_scale,
                mask=mask,
            )
            
            with torch.no_grad():
                quality = self.quality(output=local_out)
                next_state = self.extract(local_out)
                self.refiner.update(
                    state=state, action=action, reward=quality, next_state=next_state)
            
            combined = torch.cat([local_out, globe_out], dim=-1)
            return self.projection(combined)

        def quality(self, output):
            """Calculate quality metric for reinforcement learning."""
            with torch.no_grad():
                safe_output = torch.clamp(output, min=1e-10)
                entropy = -(safe_output * torch.log(safe_output)).sum(-1).mean()
                coverage = (output > 0.01).float().mean()
                return float(coverage - 0.1 * entropy)

        def extract(self, x):
            """Extract state features for RL agent."""
            with torch.no_grad():
                pooled = x.reshape(-1, self.dims)
                meadims = pooled.mean(dim=0)
                var_state = pooled.var(dim=0, unbiased=False)
                state = torch.cat([meadims, var_state])
                state_id = self.discretize(state.cpu().numpy())
            return state_id

        def discretize(self, state):
            """Convert continuous state to discrete state ID."""
            bins = np.linspace(-1, 1, num=10)
            state_discrete = np.digitize(state, bins)
            state_hash = sum(val * (10**i) for i, val in enumerate(state_discrete[:20]))
            state_id = int(state_hash % (self.refiner.states - 1))
            return state_id

        def action_scale(self, action):
            """Convert discrete action to continuous scale factor."""
            span_value = action / (self.refiner.actions - 1)
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            span_scale = torch.tensor([span_value], device=device, dtype=dtype)
            return span_scale


        def _focus(self, query, key, value, span_scale, mask=None):
            """Iterative attention refinement with zero-padding for invalid tokens."""
            max_iterations = 10
            iteration = 0
            prev_attn = torch.zeros_like(input=query)
            attn_out = torch.zeros_like(input=query)
            attn_weights = None

            threshold = self.threshold.item()
            s_factor = self.s_factor.item()

            while iteration < max_iterations:
                span_len = int(self.max_span * span_scale.mean().item())
                span_len = min(span_len, query.size(1), key.size(1), value.size(1))
                eff_span = min(span_len, self.max_dist)

                if eff_span == 0:
                    break

                q_span = query[:, :eff_span, :]
                k_span = key[:, :eff_span, :]
                v_span = value[:, :eff_span, :]

                batch, ctx, dims = q_span.size()
                
                q = q_span.view(batch, ctx, self.head, -1).transpose(1, 2)
                k = k_span.view(batch, ctx, self.head, -1).transpose(1, 2)
                v = v_span.view(batch, ctx, self.head, -1).transpose(1, 2)

                if self.sharpen:
                    temperature = 1.0 + self.temp_scale * (1.0 - span_scale.mean().item())
                else:
                    temperature = 0.5 + self.temp_scale * span_scale.mean().item()  # noqa: F841
                
                scale = (dims // self.head) ** -0.5
                attn = torch.matmul(q, k.transpose(-1, -2)) * scale
                
                if mask is not None:
                    if mask.dim() == 4:
                        q_len, k_len = q.size(2), k.size(2)
                        mask_q_len = min(mask.size(2), q_len)
                        mask_k_len = min(mask.size(3), k_len)
                        
                        mask_part = mask[:, :, :mask_q_len, :mask_k_len]
                        if mask_part.dtype == torch.bool:
                            attn_part = attn[:, :, :mask_q_len, :mask_k_len]
                            masked_attn_part = attn_part.masked_fill(mask_part, float("-inf"))
                            new_attn = attn.clone()
                            new_attn[:, :, :mask_q_len, :mask_k_len] = masked_attn_part
                            attn = new_attn
                        else:
                            attn_part = attn[:, :, :mask_q_len, :mask_k_len]
                            masked_attn_part = attn_part + mask_part
                            new_attn = attn.clone()
                            new_attn[:, :, :mask_q_len, :mask_k_len] = masked_attn_part
                            attn = new_attn
                
                attn = F.softmax(attn, dim=-1)
                
                if mask is not None and mask.dtype == torch.bool:
                    q_len, k_len = q.size(2), k.size(2)
                    mask_q_len = min(mask.size(2), q_len)
                    mask_k_len = min(mask.size(3), k_len)
                    
                    binary_mask = (~mask[:, :, :mask_q_len, :mask_k_len]).float()
                    attn_part = attn[:, :, :mask_q_len, :mask_k_len]
                    masked_attn_part = attn_part * binary_mask
                    
                    attn_sum = masked_attn_part.sum(dim=-1, keepdim=True)
                    normalized_attn_part = masked_attn_part / (attn_sum + 1e-6)
                    
                    new_attn = attn.clone()
                    new_attn[:, :, :mask_q_len, :mask_k_len] = normalized_attn_part
                    attn = new_attn
                    
                attn_output = torch.matmul(attn, v)
                attn_out = attn_output.transpose(1, 2).contiguous().view(batch, ctx, -1)

                diff = torch.abs(attn_out - prev_attn).mean()
                dynamic_threshold = threshold + s_factor * diff

                if diff < dynamic_threshold:
                    break

                prev_attn = attn_out.clone()
                query = query + attn_out
                iteration += 1
                
            return attn_out, attn_weights

        def slide_win(self, x, win_size, span_len, span_scale, mask=None):
            """Process input with sliding window attention."""
            batch, ctx, dims = x.size()
            num_windows = (ctx + win_size - 1) // win_size
            output = torch.zeros_like(x)

            for i in range(num_windows):
                start_idx = i * win_size
                end_idx = min((i + 1) * win_size, ctx)
                window_size = end_idx - start_idx  # noqa: F841

                key_start = max(0, start_idx - span_len + win_size)
                key_end = min(start_idx + span_len, ctx)

                query = x[:, start_idx:end_idx, :]
                key = x[:, key_start:key_end, :]
                value = key

                window_mask = None
                if mask is not None:
                    if mask.dim() == 4:
                        window_mask = mask[:, :, start_idx:end_idx, key_start:key_end]
                        
                        if window_mask.size(1) == 1:
                            window_mask = window_mask.expand(-1, self.head, -1, -1)

                attn_out, _ = self._focus(
                    query=query,
                    key=key,
                    value=value,
                    span_scale=span_scale,
                    mask=window_mask,
                )

                output[:, start_idx:end_idx, :] = attn_out

            return output
