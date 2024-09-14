import torch


class FeedForward(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        # x: batch, site, embedding
        x = self.model(x)
        # x: batch, site, embedding
        return x


class SelfAttention(torch.nn.Module):

    def __init__(self, embedding_dim, heads_num):
        super().__init__()

        self.heads_num = heads_num
        self.heads_dim = embedding_dim // heads_num
        assert self.heads_num * self.heads_dim == embedding_dim

        self.norm = torch.nn.LayerNorm(embedding_dim)

        self.qkv = torch.nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, kv_cache, mask):
        # x: batch, site, embedding
        x = self.norm(x)
        # x: batch, site, embedding
        batch_size, sites, embedding_dim = x.shape
        q, k, v = self.qkv(x).split(embedding_dim, dim=-1)
        q = q.view([batch_size, sites, self.heads_num, self.heads_dim])
        k = k.view([batch_size, sites, self.heads_num, self.heads_dim])
        v = v.view([batch_size, sites, self.heads_num, self.heads_dim])
        # q, k, v: batch, site, heads_num, heads_dim
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q, k, v: batch, heads_num, site, heads_dim
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        # q, k, v: batch, heads_num, site, heads_dim
        if mask is None:
            total_sites = k.shape[-2]
            mask = torch.ones(sites, total_sites, dtype=torch.bool, device=x.device).tril(diagonal=total_sites - sites)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # attn: batch, heads_num, site, heads_dim
        out = attn.transpose(1, 2).contiguous().view([batch_size, sites, embedding_dim])
        # out: batch, site, embedding_dim
        return self.out(out), (k, v)


class DecoderUnit(torch.nn.Module):

    def __init__(self, embedding_dim, heads_num, feed_forward_dim):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, heads_num)
        self.feed_forward = FeedForward(embedding_dim, feed_forward_dim)

    def forward(self, x, kv_cache, mask):
        # x: batch, site, embedding
        y, kv_cache = self.attention(x, kv_cache, mask)
        x = x + y
        # x: batch, site, embedding
        y = self.feed_forward(x)
        x = x + y
        # x: batch, site, embedding
        return x, kv_cache


class Transformers(torch.nn.Module):

    def __init__(self, embedding_dim, heads_num, feed_forward_dim, depth):
        super().__init__()
        self.layers = torch.nn.ModuleList(DecoderUnit(embedding_dim, heads_num, feed_forward_dim) for _ in range(depth))

    def forward(self, x, kv_cache, mask):
        if kv_cache is None:
            kv_cache = [None for _ in self.layers]
        # x: batch, site, embedding
        for i, layer in enumerate(self.layers):
            x, kv_cache[i] = layer(x, kv_cache[i], mask)
        # x: batch, site, embedding
        return x, kv_cache


class Tail(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x: batch, site, embedding
        y = self.model(x)
        # y: batch, site, output
        return y


class Embedding(torch.nn.Module):

    def __init__(self, sites, physical_dim, embedding_dim):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.randn([sites, physical_dim, embedding_dim]))

    def forward(self, x, base):
        # x: batch, sites
        x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.parameter.size(-1))
        # x: batch, sites, config=1, embedding

        # param: sites, config, embedding
        parameter = self.parameter[base:][:x.size(1)].unsqueeze(0).expand(x.size(0), -1, -1, -1)
        # param: batch, sites, config, embedding

        result = torch.gather(parameter, -2, x)
        # result: batch, site, 1, embedding

        return result.squeeze(-2)


class WaveFunction(torch.nn.Module):

    def __init__(
        self,
        *,
        double_sites,
        physical_dim,
        is_complex,
        spin_up,
        spin_down,
        embedding_dim,
        heads_num,
        feed_forward_dim,
        depth,
        ordering,
    ):
        super().__init__()
        assert double_sites % 2 == 0
        self.double_sites = double_sites
        self.sites = double_sites // 2
        assert physical_dim == 2
        assert is_complex == True
        self.spin_up = spin_up
        self.spin_down = spin_down
        self.embedding_dim = embedding_dim
        self.heads_num = heads_num
        self.feed_forward_dim = feed_forward_dim
        self.depth = depth

        self.embedding = Embedding(self.sites, 4, self.embedding_dim)  # spin_up * spin_down
        self.transformers = Transformers(self.embedding_dim, self.heads_num, self.feed_forward_dim, self.depth)
        self.tail = Tail(self.embedding_dim, self.feed_forward_dim, 8)  # (4 configs) * (amplitude and phase)

        if isinstance(ordering, int) and ordering == +1:
            ordering = list(range(self.sites))
        if isinstance(ordering, int) and ordering == -1:
            ordering = list(reversed(range(self.sites)))
        self.register_buffer('ordering', torch.tensor(ordering, dtype=torch.int64), persistent=True)
        ordering_bak = torch.zeros(self.sites, dtype=torch.int64)
        ordering_bak.scatter_(0, self.ordering, torch.arange(self.sites))
        self.register_buffer('ordering_bak', ordering_bak, persistent=True)

    def mask(self, x):
        # x : batch * i * 2
        i = x.size(1)
        # number : batch * 2
        number = x.sum(dim=1)

        up_electron = number[:, 0]
        down_electron = number[:, 1]
        up_hole = i - up_electron
        down_hole = i - down_electron

        add_up_electron = up_electron < self.spin_up
        add_down_electron = down_electron < self.spin_down
        add_up_hole = up_hole < self.sites - self.spin_up
        add_down_hole = down_hole < self.sites - self.spin_down

        add_up = torch.stack([add_up_hole, add_up_electron], dim=-1).unsqueeze(-1)
        add_down = torch.stack([add_down_hole, add_down_electron], dim=-1).unsqueeze(-2)
        add = torch.logical_and(add_up, add_down)
        # add: batch * 2 * 2
        # -------------------------------------------------------
        # |                 | add down hole | add down electron |
        # | add up hole     |               |                   |
        # | add up electron |               |                   |
        return add

    def normalize_amplitude(self, x):
        # x : ... * 2 * 2
        param = -(2 * x).exp().sum(dim=[-2, -1]).log() / 2
        x = x + param.unsqueeze(-1).unsqueeze(-1)
        return x

    def forward(self, x):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        batch_size = x.size(0)
        x = x.reshape([batch_size, self.sites, 2])
        x = torch.index_select(x, 1, self.ordering_bak)

        x4 = x[:, :-1, 0] * 2 + x[:, :-1, 1]  # batch * sites
        x4 = torch.cat([torch.zeros([batch_size, 1], device=device, dtype=torch.int64), x4], dim=1)
        emb = self.embedding(x4, 0)  # batch * sites * embedding  # emb 0 for bos, emb 1 for site 0, ...
        post_transformer, _ = self.transformers(emb, None, None)  # batch * sites * embedding
        tail = self.tail(post_transformer)  # batch * sites * 8
        # amp/phase : bathc * sites * 2 * 2
        amplitude = tail[:, :, :4].reshape(batch_size, self.sites, 2, 2)
        phase = tail[:, :, 4:].reshape(batch_size, self.sites, 2, 2)
        amplitude = amplitude + torch.stack([torch.where(self.mask(x[:, :i]), 0, -torch.inf) for i in range(self.sites)], dim=1)
        amplitude = self.normalize_amplitude(amplitude)

        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.sites)
        sites_indices = torch.arange(self.sites).unsqueeze(0).expand(batch_size, -1)
        amplitude = amplitude[batch_indices, sites_indices, x[:, :, 0], x[:, :, 1]].sum(dim=1)
        phase = phase[batch_indices, sites_indices, x[:, :, 0], x[:, :, 1]].sum(dim=1)
        return (amplitude + 1j * phase).exp()

    def binomial(self, count, possibility):
        possibility = torch.clamp(possibility, min=0, max=1)
        possibility = torch.where(count == 0, 0, possibility)
        dist = torch.distributions.binomial.Binomial(count, possibility)
        result = dist.sample()
        result = result.to(dtype=torch.int64)
        # Numerical error since result was cast to float.
        return torch.clamp(result, min=torch.zeros_like(count), max=count)

    def generate_unique(self, batch_size):
        # https://arxiv.org/pdf/2408.07625
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # x: batch * site * 2(up+down)
        x = torch.zeros([1, 1, 2], device=device, dtype=torch.int64)  # site=1: bos=0
        unperturbed_log_probability = torch.tensor([0], dtype=dtype, device=device)
        perturbed_log_probability = torch.tensor([0], dtype=dtype, device=device)
        cache = None

        for i in range(self.sites):
            local_batch_size = x.size(0)

            xi = x[:, -1:]  # batch * sites=1 * 2
            xi4 = xi[:, :, 0] * 2 + xi[:, :, 1]  # batch * sites=1
            emb = self.embedding(xi4, i)  # bathc * sites=1 * embedding  # emb 0 for bos, emb 1 for site 0, ...
            post_transformer, cache = self.transformers(emb, cache, None)  # batch * sites=1 * embedding
            tail = self.tail(post_transformer)  # batch * sites=1 * 8

            delta_log_amplitude = tail[:, :, :4].reshape([local_batch_size, 2, 2])  # batch * 2 * 2
            delta_log_amplitude = delta_log_amplitude + torch.where(self.mask(x[:, 1:]), 0, -torch.inf)
            delta_log_amplitude = self.normalize_amplitude(delta_log_amplitude)

            l = (2 * delta_log_amplitude).reshape([local_batch_size, 4])
            l = unperturbed_log_probability.view([-1, 1]) + l
            L = l - (-torch.rand_like(l).log()).log()
            Z = L.max(dim=-1).values.reshape([-1, 1])
            L = -torch.log(torch.exp(-perturbed_log_probability.view([-1, 1])) - torch.exp(-Z) + torch.exp(-L))

            x0 = torch.cat([x, torch.tensor([[0, 0]], device=device).expand(local_batch_size, -1, -1)], dim=1)
            x1 = torch.cat([x, torch.tensor([[0, 1]], device=device).expand(local_batch_size, -1, -1)], dim=1)
            x2 = torch.cat([x, torch.tensor([[1, 0]], device=device).expand(local_batch_size, -1, -1)], dim=1)
            x3 = torch.cat([x, torch.tensor([[1, 1]], device=device).expand(local_batch_size, -1, -1)], dim=1)

            x = torch.cat([x0, x1, x2, x3])
            unperturbed_log_probability = l.permute(1, 0).reshape([-1])
            perturbed_log_probability = L.permute(1, 0).reshape([-1])
            cache = [(k.repeat(4, 1, 1, 1), v.repeat(4, 1, 1, 1)) for [k, v] in cache]

            selected = perturbed_log_probability.sort(descending=True).indices[:batch_size]
            x = x[selected]
            unperturbed_log_probability = unperturbed_log_probability[selected]
            perturbed_log_probability = perturbed_log_probability[selected]
            cache = [(k[selected], v[selected]) for [k, v] in cache]

            selected = perturbed_log_probability != -torch.inf
            x = x[selected]
            unperturbed_log_probability = unperturbed_log_probability[selected]
            perturbed_log_probability = perturbed_log_probability[selected]
            cache = [(k[selected], v[selected]) for [k, v] in cache]

        x = torch.index_select(x[:, 1:], 1, self.ordering)
        x = x.reshape([x.size(0), self.double_sites])
        return x, self(x), None, None
