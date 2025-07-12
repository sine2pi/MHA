import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
from torch.nn import Linear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
    
def exists(v):
    return v is not None

def default(v, b):
    return v if exists(v) else b

class FocusWindow(nn.Module):
    
    def __init__(self, dims: int, head: int, max_span: int = 512, max_dist: int = 256, 
                 debug: List[str] = [], learn_lr: bool = False, base_lr: float = 0.001, return_head_weights=False):

        super().__init__()
        self.dims = dims
        self.head = head
        self.head_dim = dims // head

        self.debug = debug
        self.learn_lr = learn_lr
        self.base_lr = base_lr
        self.threshold = nn.Parameter(torch.tensor(0.01))
        self.s_factor = nn.Parameter(torch.tensor(0.1))
        self.temp_scale = nn.Parameter(torch.tensor(1.0))
        self.sharpen = True
        self.return_head_weights = return_head_weights
        
        self.q_proj = Linear(dims, dims)
        self.k_proj = Linear(dims, dims)
        self.v_proj = Linear(dims, dims)
        
        # self.bias_strength = nn.Parameter(torch.tensor(0.5))
        
        self.win_sizes = {
            "spectrogram": 1000,
            "waveform": 1000, 
            "pitch": 1000,
            "envelope": 51000,
            "phase": 1000
        }
        
        self.span_lengths = {
            "spectrogram": 1200,
            "waveform":1200,
            "pitch": 1200,
            "envelope": 1200,
            "phase": 1200
        }

        self.head_router = nn.Sequential(
            Linear(dims, dims),
            nn.SiLU(),
            Linear(dims, head)
        )

        self.lr_predictor = nn.Sequential(
            Linear(dims, dims // 4),
            nn.SiLU(),
            Linear(dims // 4, 1),
            nn.Sigmoid()
        )
        
    def predict_attention_lr(self, x, xa=None):
        lr_factor = self.lr_predictor(x.mean(dim=1))
        return self.base_lr * lr_factor

    def _focus(self, q, k, v, span_scale, max_span, max_dist, mask=None):
        
        q_energy = torch.norm(q, dim=-1).mean()
        k_energy = torch.norm(k, dim=-1).mean()
        interest = (q_energy + k_energy) / 2
        
        base_iter = 3
        max_iter = int(base_iter + interest * 12)
        max_iter = min(max_iter, 20)
        
        iters = 0
        prev_attn = torch.zeros_like(q)
        attn_out = torch.zeros_like(q)
        attn_weights = None

        threshold = self.threshold.item()
        s_factor = self.s_factor.item()

        while iters < max_iter:
            span_len = int(max_span * span_scale)
            span_len = min(span_len, q.size(1), k.size(1), k.size(1))
            eff_span = min(span_len, max_dist)

            if eff_span == 0:
                break

            q_span = q[:, :eff_span, :]
            k_span = k[:, :eff_span, :]
            v_span = v[:, :eff_span, :]

            batch, ctx, dims = q_span.size()
            
            q_head = q_span.view(batch, ctx, self.head, -1).transpose(1, 2)
            k_head = k_span.view(batch, ctx, self.head, -1).transpose(1, 2)
            v_head = v_span.view(batch, ctx, self.head, -1).transpose(1, 2)

            if self.sharpen:
                temperature = 1.0 + self.temp_scale * (1.0 - span_scale)
            else:
                temperature = 0.5 + self.temp_scale * span_scale
            
            scale = (dims // self.head) ** -0.5
            attn = torch.matmul(q_head, k_head.transpose(-1, -2)) * scale
            attn = attn * temperature
            
            if mask is not None:
                if mask.dim() == 4:
                    q_len, k_len = q_head.size(2), k_head.size(2)
                    mask_q_len = min(mask.size(2), q_len)
                    mask_k_len = min(mask.size(3), k_len)
                    
                    mask_part = mask[:, :, :mask_q_len, :mask_k_len]
                    if mask_part.dtype == torch.bool:
                        attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len].masked_fill(
                            mask_part, float("-inf")
                        )
                    else:
                        attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len] + mask_part
            
            attn = F.softmax(attn, dim=-1)
            
            if mask is not None and mask.dtype == torch.bool:
                q_len, k_len = q_head.size(2), k_head.size(2)
                mask_q_len = min(mask.size(2), q_len)
                mask_k_len = min(mask.size(3), k_len)
                
                binary_mask = (~mask[:, :, :mask_q_len, :mask_k_len]).float()
                attn_to_mask = attn[:, :, :mask_q_len, :mask_k_len]
                attn_to_mask = attn_to_mask * binary_mask
                
                attn_sum = attn_to_mask.sum(dim=-1, keepdim=True)
                attn_to_mask = attn_to_mask / (attn_sum + 1e-6)
                
                attn[:, :, :mask_q_len, :mask_k_len] = attn_to_mask
                
            attn_output = torch.matmul(attn, v_head)
            attn_out = attn_output.transpose(1, 2).contiguous().view(batch, ctx, dims)

            q = q.clone()
            q[:, :eff_span, :] = q_span + attn_out

            diff = torch.abs(attn_out - prev_attn).mean()
            dynamic_threshold = threshold + s_factor * diff

            if diff < dynamic_threshold:
                break

            prev_attn = attn_out
            iters += 1
            
        return attn_out, attn_weights

    def slide_win(self, x, xa, win_size, span_len, span_scale, max_span, max_dist, mask=None):
        batch, ctx, dims = x.size()
        num_win = (ctx + win_size - 1) // win_size
        output = torch.zeros_like(x)
        z = default(xa, x).to(device, dtype)
        q = self.q_proj(x)
        k = self.k_proj(z)
        v = self.v_proj(z)

        for i in range(num_win):
            start = i * win_size
            end = min((i + 1) * win_size, ctx)
            win_size = end - start

            k_start = max(0, start - span_len + win_size)
            k_end = min(start + span_len, ctx)

            q = x[:, start:end, :]     
            k = x[:, k_start:k_end, :]          
            v = x[:, k_start:k_end, :]   

            win_mask = None
            if mask is not None:
                if mask.dim() == 4:
                    win_mask = mask[:, :, start:end, k_start:k_end]
                    
                    if win_mask.size(1) == 1:
                        win_mask = win_mask.expand(-1, self.head, -1, -1)

            attn_out, _ = self._focus(q=q, k=k, v=v, span_scale=span_scale, max_span=max_span, max_dist=max_dist, mask=win_mask)

            output[:, start:end, :] = attn_out

        return output

    def predict_head_importance(self, x, xa=None):
        if xa is not None:
            combined = x + 0.1 * xa
        else:
            combined = x
        head_importance = self.head_router(combined.mean(dim=1))
        return head_importance

    def inverse_mel_scale(seld, mel_freq: Tensor) -> Tensor:
        return 700.0 * ((mel_freq / 1127.0).exp() - 1.0)

    def mel_scale(self, freq: Tensor) -> Tensor:
        return 1127.0 * (1.0 + freq / 700.0).log()

    def forward(self, x, xa=None, mask=None, enc=None, feature_type="spectrogram"):

        span_scale = self. inverse_mel_scale(enc.get("f0").mean())
        max_span = enc.get("f0").max()
        max_dist = enc.get("f0").shape[1]
        
        print(f"Max Span: {max_span}, Max Dist: {max_dist}, Span Scale: {span_scale} {xa.shape if xa is not None else None} {x.shape if x is not None else None}") 
        
        win_size = self.win_sizes.get(feature_type, 128)
        span_len = self.span_lengths.get(feature_type, 256)
        
        output = self.slide_win(
            x=x,
            xa=xa,
            win_size=win_size,
            span_len=span_len,
            span_scale=span_scale,
            max_span=max_span,
            max_dist=max_dist,
            mask=mask
        )

        if self.learn_lr:
            lr_factor = self.lr_predictor(output.mean(dim=1))
            return output, lr_factor

        if self.return_head_weights:
            head_weights = self.predict_head_importance(x, xa)
            return output, head_weights

        # if self.return_bias:
        #     bias_strength = torch.sigmoid(self.bias_strength)
        #     return bias_strength * output
        else:
            return output

class CrossFeatureFocusAttention(nn.Module):
    def __init__(self, dims: int, head: int, features: List[str] = ["spectrogram", "pitch"]):
        super().__init__()
        self.dims = dims
        self.head = head
        self.features = features
        
        self.cross_attn_layers = nn.ModuleDict({
            feature: nn.MultiheadAttention(dims, head, batch_first=True)
            for feature in features
        })
        
        self.feature_fusion = nn.Sequential(
            Linear(dims * len(features), dims),
            nn.SiLU(),
            Linear(dims, dims)
        )
        
    def forward(self, x, enc, mask=None):
        if enc is None:
            return None
            
        cross_features = []
        for feature in self.features:
            if feature in enc:
                xa = enc[feature]
                if xa is not None:
                    attn_out, _ = self.cross_attn_layers[feature](
                        x, xa, xa, 
                        attn_mask=mask
                    )
                    cross_features.append(attn_out)
        
        if not cross_features:
            return None
            
        if len(cross_features) > 1:
            fused = torch.cat(cross_features, dim=-1)
            return self.feature_fusion(fused)
        else:
            return cross_features[0]

class AdaptiveAttentionLR(nn.Module):
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.dims = dims
        self.head = head
        
        self.lr_predictor = nn.Sequential(
            Linear(dims, dims // 4),
            nn.SiLU(),
            Linear(dims // 4, 1),
            nn.Sigmoid()
        )
        
        self.quality_estimator = nn.Sequential(
            Linear(dims, dims // 2),
            nn.SiLU(),
            Linear(dims // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, xa=None, mask=None):
        quality = self.quality_estimator(x.mean(dim=1))
        
        lr_factor = self.lr_predictor(x.mean(dim=1))
        
        adaptive_lr = quality * lr_factor
        
        return adaptive_lr, adaptive_lr

class SmartSensorResidual(nn.Module):
    def __init__(self, ctx, dims, head, act, cross_attn=True, debug: List[str] = [], 
                 use_smart_sensor=True):
        super().__init__()
        self.ctx = ctx
        self.dims = dims
        self.head = head
        self.act = act
        self.debug = debug
        
        if use_smart_sensor:
            self.focus_attn = FocusWindow(dims, head, feature_type="waveform")
            self.cross_feature_guide = CrossFeatureFocusAttention(dims, head, 
                                                               features=["spectrogram", "pitch"])
            self.adaptive_lr = AdaptiveAttentionLR(dims, head)
            
            self.attna = nn.MultiheadAttention(dims, head, debug=debug)
            self.lna = nn.RMSNorm(dims)
    
    def forward(self, x, xa=None, mask=None, enc=None, feature_type="spectrogram"):
        if hasattr(self, 'focus_attn') and enc is not None:
            focus_output, head_weights = self.focus_attn(x, enc.get("waveform"), mask, 
                                                       return_head_weights=True)
            
            cross_guidance = self.cross_feature_guide(x, enc, mask)
            
            _, attention_lr = self.adaptive_lr(x, enc.get("waveform"), mask)
            
            x = x + self.attna(
                self.lna(x), 
                xa=None, 
                mask=mask, 
                head_weights=head_weights,
                cross_guidance=cross_guidance,
                attention_lr=attention_lr,
                enc=enc, 
                feature_type=feature_type
            )[0]
        
        return x
