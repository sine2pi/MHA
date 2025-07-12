
class OneShot(nn.Module):
    """
    One-shot cross-attention that returns a bias tensor
    to be added to the main attention logits.
    """
    def __init__(self, dims: int, head: int, scale: float = 0.3):
        super().__init__()
        self.head  = head
        self.hdim  = dims // head
        self.scale = scale                       # how strong the bias is

        self.q_proj = Linear(dims, dims)
        self.k_proj = Linear(dims, dims)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor | None:
        """
        x      : (B, Q, D)
        guide  : (B, K, D)   – e.g. pitch / harmonic stream
        Returns
        -------
        bias   : (B, head, Q, K)  or None if guide is None
        """
        if guide is None:
            return None

        B, Q, _ = x.shape
        K       = guide.size(1)

        q = self.q_proj(x ).view(B, Q, self.head, self.hdim).transpose(1,2)  # (B, H, Q, hdim)
        k = self.k_proj(guide).view(B, K, self.head, self.hdim).transpose(1,2)

        bias = (q @ k.transpose(-1, -2)) * self.scale / math.sqrt(self.hdim)  # (B,H,Q,K)
        return bias
