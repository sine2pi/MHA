
class CuriosityHead(nn.Module):
    def __init__(self, d, h, bias=True):
        super().__init__()
        self.h  = h              # base heads
        self.dh = d // h
        self.qkv = nn.Linear(d, d * 3, bias=bias)
        self.qkv_aux = nn.Linear(d, d * 3, bias=bias)  # curiosity heads
        self.o  = nn.Linear(d, d, bias=bias)
        self.g  = nn.Parameter(torch.zeros(h))         # per-head gate logit

    def split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dh).transpose(1, 2)  # b h t dh

    def merge(self, x):
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(self, x, xa, mask=None):

        q, k, v   = self.qkv(x).chunk(3, -1)
        qa, ka, va = self.qkv_aux(xa).chunk(3, -1)

        q, k, v   = map(self.split, (q, k, v))
        qa, ka, va = map(self.split, (qa, ka, va))

        dots      = (q @ k.transpose(-2, -1)) / self.dh**0.5      # b h t t
        dots_aux  = (q @ ka.transpose(-2, -1)) / self.dh**0.5     # b h t ta

        if mask is not None: dots = dots.masked_fill(mask, -9e15)

        p   = dots.softmax(-1)
        pa  = dots_aux.softmax(-1)

        h_main = p  @ v                       # b h t dh
        h_aux  = pa @ va                      # b h t dh

        g = torch.sigmoid(self.g).view(1, -1, 1, 1)  # b h t dh broadcast
        out = self.merge(h_main * (1 - g) + h_aux * g)
        return self.o(out)
