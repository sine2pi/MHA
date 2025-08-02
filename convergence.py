
class attentionb(nn.Module):
    def __init__(self, dims: int, head: int, max_iter: int = 3, 
    threshold: float = 0.01, factor: float = 0.1, dropout: float = 0.1, temp = 1.0):
        super(attentionb, self).__init__()

        self.head = head
        self.dims = dims
        self.head_dim = dims // head
        self.win = 0

        self.que = nn.Linear(dims, dims, bias=False) 
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims, bias=False)

        self.lna = nn.LayerNorm(dims) 
        self.lnb = nn.LayerNorm(self.head_dim) 
        self.rope = rotary(dims, head) 

        self.max_iter = max_iter
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=True)
        self.temp = nn.Parameter(torch.tensor(temp), requires_grad=True)        
        self.factor = nn.Parameter(torch.tensor(factor), requires_grad=True)
        self.local = LocalOut(dims, head)   

    def update_win(self, win_size=None):
        if win_size is not None:
            self.win_size = win_size
            return win_size
        elif hasattr(self, 'win_size') and self.win_size is not None:
            win_size = self.win_size
            return win_size
        return None

    def _focus(self, x, xa = None, mask = None, win_size=None):

        q = self.que(self.lna(x))
        k, v = self.kv(self.lna(default(xa, x))).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.head), (q, k, v))
      
        self.scale = q.shape[-1] ** -0.35
        q = self.rope(q)
        k = self.rope(k)

        iteration = 0
        temp = self.temp
        prev_out = torch.zeros_like(q)
        attn_out = torch.zeros_like(q)
        threshold = self.threshold
        factor = self.factor
        curq = q #if curq is None else curq
        
        while iteration < self.max_iter:
            eff_span = min(curq.shape[1], k.shape[1])
            if xa is not None:
                eff_span = min(eff_span, xa.shape[1])
            if eff_span == 0: 
                break

            qiter = curq[:, :, :eff_span, :]
            kiter = k[:, :, :eff_span, :]
            viter = v[:, :, :eff_span, :]
            q = self.local.q_hd(qiter)
            k = self.local.k_hd(kiter)
            v = self.local.v_hd(viter)

            iter_mask = None
            if mask is not None:
                if mask.dim() == 4: 
                    iter_mask = mask[:, :, :eff_span, :eff_span]
                elif mask.dim() == 2: 
                    iter_mask = mask[:eff_span, :eff_span]

            attn_iter = calculate_attention(
                self.lnb(q), self.lnb(k), v,
                mask=iter_mask, temp=temp)

            iter_out = torch.zeros_like(curq)
            iter_out[:, :, :eff_span, :] = attn_iter
            diff = torch.abs(iter_out - prev_out).mean()
            dthresh = threshold + factor * diff
            if diff < dthresh and iteration > 0:
                attn_out = iter_out
                break

            prev_out = iter_out.clone()
            curq = curq + iter_out
            attn_out = iter_out
            # if win_size is not None:
            #     if win_size > self.win:
            #         temp += 0.005
            #     else:
            #         temp -= 0.005
            #     self.win = win_size   
            iteration += 1

        out = attn_out.permute(0, 2, 1, 3).flatten(start_dim=2)
        return out

    def _slide_win_local(self, x, mask = None) -> Tensor:

        win = self.update_win()
        win_size = win if win is not None else self.head_dim
        span_len = win_size + win_size // self.head

        _, ctx, _ = x.shape
        out = torch.zeros_like(x)
        windows = (ctx + win_size - 1) // win_size

        for i in range(windows):
            qstart = i * win_size
            qend = min(qstart + win_size, ctx)
            qlen = qend - qstart
            if qlen == 0: 
                continue

            kstart = max(0, qend - span_len)
            qwin = x[:, qstart:qend, :]
            kwin = x[:, kstart:qend, :]

            win_mask = None
            if mask is not None:
                if mask.dim() == 4:
                    win_mask = mask[:, :, qstart:qend, kstart:qend]
                elif mask.dim() == 2:
                    win_mask = mask[qstart:qend, kstart:qend]

            attn_out = self._focus(x=qwin, xa=kwin, mask=win_mask, win_size=win_size)
            out[:, qstart:qend, :] = attn_out
        return out

    def forward(self, x, xa = None, mask = None):
            x = self._slide_win_local(x, mask=None)
            xa = self._slide_win_local(xa, mask=None)
            output = self._focus(x, xa, mask=None)
            return self.out(output)
       
