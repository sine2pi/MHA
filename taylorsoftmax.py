class attentiona(nn.Module):
    def __init__(self, dims: int, head: int, dropout_rate: float = 0.1):
        super().__init__()

        self.head = head
        self.dims = dims
        self.que = nn.Linear(dims, dims, bias=False) 
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims, bias=False)
        self.ln = nn.LayerNorm(dims) 
        self.rope = rotary(dims, head) 

    def taylor_softmax(self, x, order=2):
        taylor_approx = 1.0
        for i in range(1, order + 1):
            factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            taylor_approx += x**i / factorial_i
        return taylor_approx / torch.sum(taylor_approx, dim=-1, keepdim=True)

    def taylor_softmax_2nd_order(x):
        exp_approx = 1 + x + (x**2) / 2
        return exp_approx / torch.sum(exp_approx, dim=-1, keepdim=True)

    def cos_sim(q: Tensor, k: Tensor, v: Tensor, mask) -> Tensor:
        q_norm = torch.nn.functional.normalize(q, dim=-1, eps=1e-12)
        k_norm = torch.nn.functional.normalize(k, dim=-1, eps=1e-12)
        qk_cosine = torch.matmul(q_norm, k_norm.transpose(-1, -2))
        qk_cosine = qk_cosine + mask
        weights = F.softmax(qk_cosine, dim=-1)
        out = torch.matmul(weights, v)
        return out

    def forward(self, x, xa = None, mask = None):

        q = self.que(self.ln(x))
        k, v = self.kv(self.ln(x if xa is None else xa)).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.head), (q, k, v))
        scale = q.shape[-1] ** -0.5

        q = self.rope(q)
        k = self.rope(k)

        qk = einsum('b h k d, b h q d -> b h k q', q, k) * scale 
        # qk = torch.nn.functional.softmax(qk, dim=-1)
        qk = self.taylor_softmax(qk, order=2)        

        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        wv = rearrange(wv, 'b h c d -> b c (h d)')
        out = self.out(wv)
        return out
