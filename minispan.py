class CombinedAdaptiveSpanRecurrentAttention(nn.Module):
    def __init__(self, n_state, n_head, max_rel_dist, base, max_span, chunk_size):
        super().__init__()
        self.n_head = n_head
        self.multihead_attn = MultiheadAttention(n_state, n_head, max_rel_dist, base)
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))
        self.chunk_size = chunk_size

    def forward(self, query, key, value, kv_cache=None):
        assert query.dim() == 2 or query.dim() == 3, "query should be unbatched 2D or batched 3D tensor but received {}-D tensor".format(query.dim())
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])  # Adjust this based on your requirements

        batch_size, seq_len, n_state = query.size()
        output = torch.zeros_like(query).to(query.device)

        if kv_cache is None:
            kv_cache = {}
        key_global = key
        value_global = value

        for i in range(0, seq_len, self.chunk_size):
            end = min(seq_len, i + self.chunk_size)
            query_chunk = query[:, i:end, :]  

            if 'k' not in kv_cache:
                kv_cache['k'] = key_global.clone().detach().to(query.device)
                kv_cache['v'] = value_global.clone().detach().to(query.device)

            key_chunk = kv_cache['k'][:, :end, :]
            value_chunk = kv_cache['v'][:, :end, :]

            # Adaptive Span Attention
            span_length = int(self.max_span * self.span_scale.item())
            span_length = min(span_length, query_chunk.shape[1])
            query_span = query_chunk[:, :span_length, :]
            key_span = key_chunk[:, :span_length, :]
            value_span = value_chunk[:, :span_length, :]

            attn_output, _ = self.multihead_attn(query_span, key_span, value_span)
            output[:, i:end, :] = attn_output

        return output, kv_cache
