# This file implements auto regressive transformers network for electron system.

import torch


class FeedForward(torch.nn.Module):
    """
    Feed forward layer in transformers.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.model: torch.nn.Module = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch * site * embedding
        x = self.model(x)
        # x: batch * site * embedding
        return x


class SelfAttention(torch.nn.Module):
    """
    Self attention unit with kv cache support.
    """

    def __init__(self, embedding_dim: int, heads_num: int) -> None:
        super().__init__()

        self.heads_num: int = heads_num
        self.heads_dim: int = embedding_dim // heads_num
        assert self.heads_num * self.heads_dim == embedding_dim

        self.norm: torch.nn.Module = torch.nn.LayerNorm(embedding_dim)

        self.qkv: torch.nn.Module = torch.nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out: torch.nn.Module = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: batch * site * embedding
        x = self.norm(x)
        # x: batch * site * embedding
        batch_size, sites, embedding_dim = x.shape
        q, k, v = self.qkv(x).split(embedding_dim, dim=-1)
        # q, k, v: batch * site * embedding
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
        # q: batch, heads_num, site, heads_dim
        # k, v: batch, heads_num, total_site, heads_dim
        if mask is None:
            total_sites = k.shape[-2]
            mask = torch.ones(sites, total_sites, dtype=torch.bool, device=x.device).tril(diagonal=total_sites - sites)
        # call scaled dot product attention
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # attn: batch, heads_num, site, heads_dim
        out = attn.transpose(1, 2).reshape([batch_size, sites, embedding_dim])
        # out: batch, site, embedding_dim
        return self.out(out), (k, v)


class DecoderUnit(torch.nn.Module):
    """
    Decoder unit in transformers, containing self attention and feed forward.
    """

    def __init__(self, embedding_dim: int, heads_num: int, feed_forward_dim: int) -> None:
        super().__init__()
        self.attention: torch.nn.Module = SelfAttention(embedding_dim, heads_num)
        self.feed_forward: torch.nn.Module = FeedForward(embedding_dim, feed_forward_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x, y: batch * site * embedding
        y, result_cache = self.attention(x, kv_cache, mask)
        x = x + y
        y = self.feed_forward(x)
        x = x + y
        return x, result_cache


class Transformers(torch.nn.Module):

    def __init__(self, embedding_dim: int, heads_num: int, feed_forward_dim: int, depth: int) -> None:
        super().__init__()
        self.layers: torch.nn.ModuleList = torch.nn.ModuleList(DecoderUnit(embedding_dim, heads_num, feed_forward_dim) for _ in range(depth))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        # x: batch * site * embedding
        result_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            if kv_cache is None:
                x, cache = layer(x, None, mask)
            else:
                x, cache = layer(x, kv_cache[i], mask)
            result_cache.append(cache)
        return x, result_cache


class Tail(torch.nn.Module):
    """
    The tail layer for transformers.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.model: torch.nn.Module = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch * site * embedding
        x = self.model(x)
        # x: batch * site * embedding
        return x


class Embedding(torch.nn.Module):
    """
    Embedding layer for transformers.
    """

    def __init__(self, sites: int, physical_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.parameter: torch.nn.Parameter = torch.nn.Parameter(torch.randn([sites, physical_dim, embedding_dim]))

    def forward(self, x: torch.Tensor, base: int) -> torch.Tensor:
        # x: batch * sites
        x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.parameter.shape[-1])
        # x: batch * sites * config=1 * embedding

        # param: sites * config * embedding
        parameter = self.parameter[base:][:x.shape[1]].unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        # param: batch * sites * config * embedding

        result = torch.gather(parameter, -2, x)
        # result: batch * site * 1 * embedding

        return result.squeeze(-2)


class WaveFunction(torch.nn.Module):

    def __init__(
            self,
            *,
            double_sites: int,  # qubits number, qubits are grouped by two for each site in naqs, so name it as double sites
            physical_dim: int,  # is always 2 for naqs
            is_complex: bool,  # is always true for naqs
            spin_up: int,  # spin up number
            spin_down: int,  # spin down number
            embedding_dim: int,  # embedding dim in transformers
            heads_num: int,  # heads number in transformers
            feed_forward_dim: int,  # feed forward dim and tail dim in transformers
            depth: int,  # depth of transformers
            ordering: int | list[int],  # ordering of sites +1 for normal order, -1 for reversed order, or the order list directly
    ) -> None:
        super().__init__()
        assert double_sites % 2 == 0
        self.double_sites: int = double_sites
        self.sites: int = double_sites // 2
        assert physical_dim == 2
        assert is_complex == True
        self.spin_up: int = spin_up
        self.spin_down: int = spin_down
        self.embedding_dim: int = embedding_dim
        self.heads_num: int = heads_num
        self.feed_forward_dim: int = feed_forward_dim
        self.depth: int = depth

        # embedding configurations on every sites(each sites contain two qubits)
        self.embedding: torch.nn.Module = Embedding(self.sites, 4, self.embedding_dim)  # spin_up * spin_down
        # main body
        self.transformers: torch.nn.Module = Transformers(self.embedding_dim, self.heads_num, self.feed_forward_dim, self.depth)
        # tail, mapping from embedding space to amplitude and phase space.
        self.tail: torch.nn.Module = Tail(self.embedding_dim, self.feed_forward_dim, 8)  # (amplitude and phase) * (4 configs)

        # ordering of sites +1 for normal order, -1 for reversed order
        if isinstance(ordering, int) and ordering == +1:
            ordering = list(range(self.sites))
        if isinstance(ordering, int) and ordering == -1:
            ordering = list(reversed(range(self.sites)))
        self.register_buffer('ordering', torch.tensor(ordering, dtype=torch.int64))
        self.register_buffer('ordering_reversed', torch.scatter(torch.zeros(self.sites, dtype=torch.int64), 0, self.ordering, torch.arange(self.sites, dtype=torch.int64)))

        # used to get device and dtype
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    @torch.jit.export
    def mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determined whether we could append spin up or spin down after uncompleted configurations.
        """
        # x : batch_size * current_site * 2
        # x are the uncompleted configurations
        batch_size = x.shape[0]
        current_site = x.shape[1]
        # number : batch_size * 2
        # number is the total electron number of uncompleted configurations
        number = x.sum(dim=1)

        # up/down_electron/hole : batch_size
        # the electron and hole number of uncompleted configurations
        up_electron = number[:, 0]
        down_electron = number[:, 1]
        up_hole = current_site - up_electron
        down_hole = current_site - down_electron

        # add_up/down_electron/hole : batch_size
        # whether able to append up/down electron/hole
        add_up_electron = up_electron < self.spin_up
        add_down_electron = down_electron < self.spin_down
        add_up_hole = up_hole < self.sites - self.spin_up
        add_down_hole = down_hole < self.sites - self.spin_down

        # add_up : batch_size * 2 * 1
        # add_down : batch_size * 1 * 2
        add_up = torch.stack([add_up_hole, add_up_electron], dim=-1).unsqueeze(-1)
        add_down = torch.stack([add_down_hole, add_down_electron], dim=-1).unsqueeze(-2)
        # add : batch_size * 2 * 2
        add = torch.logical_and(add_up, add_down)
        # add contains whether to append up/down electron/hole after uncompleted configurations
        # add[_, 0, 0] means we could add up hole and down hole
        # add[_, 0, 1] means we could add up hole and down electron
        # add[_, 1, 0] means we could add up electron and down hole
        # add[_, 1, 1] means we could add up electron and down electron
        return add

    @torch.jit.export
    def normalize_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize uncompleted log amplitude.
        """
        # x : ... * 2 * 2
        # param :  ...
        param = (2 * x).exp().sum(dim=[-2, -1]).log() / 2
        x = x - param.unsqueeze(-1).unsqueeze(-1)
        # 1 = param = sqrt(sum(x.exp()^2)) now
        return x

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate psi of given configurations.
        """
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        batch_size: int = x.shape[0]
        # x : batch_size * sites * 2
        x = x.reshape([batch_size, self.sites, 2])
        # apply ordering
        x = torch.index_select(x, 1, self.ordering_reversed)

        # prepare x4 as the input of the network
        # the last one is not needed, and zeros vector prepend at the beginning
        x4: torch.Tensor = x[:, :-1, 0] * 2 + x[:, :-1, 1]
        x4 = torch.cat([torch.zeros([batch_size, 1], device=device, dtype=torch.int64), x4], dim=1)
        # x4: batch_size * sites
        emb: torch.Tensor = self.embedding(x4, 0)
        # emb: batch_size * sites * embedding
        # in embedding layer, 0 for bos, 1 for site 0, ...
        post_transformers, _ = self.transformers(emb, None, None)  # batch * sites * embedding
        tail: torch.Tensor = self.tail(post_transformers)  # batch * sites * 8
        # amplitude/phase : batch * sites * 2 * 2
        amplitude: torch.Tensor = tail[:, :, :4].reshape(batch_size, self.sites, 2, 2)
        phase: torch.Tensor = tail[:, :, 4:].reshape(batch_size, self.sites, 2, 2)
        # filter mask for amplitude
        amplitude: torch.Tensor = amplitude + torch.stack([torch.where(self.mask(x[:, :i]), 0, -torch.inf) for i in range(self.sites)], dim=1)
        # normalize amplitude
        amplitude: torch.Tensor = self.normalize_amplitude(amplitude)

        # batch/sites_indices: batch * sites
        batch_indices: torch.Tensor = torch.arange(batch_size).unsqueeze(1).expand(-1, self.sites)
        sites_indices: torch.Tensor = torch.arange(self.sites).unsqueeze(0).expand(batch_size, -1)

        # amplitude/phase: batch * sites first, after sum over dim=1, they are amplitude/phase: batch
        amplitude: torch.Tensor = amplitude[batch_indices, sites_indices, x[:, :, 0], x[:, :, 1]].sum(dim=1)
        phase: torch.Tensor = phase[batch_indices, sites_indices, x[:, :, 0], x[:, :, 1]].sum(dim=1)
        return torch.view_as_complex(torch.stack([amplitude, phase], dim=-1)).exp()

    @torch.jit.export
    def binomial(self, count: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
        """
        Binomial sampling with given count and probability
        """
        # clamp probability
        probability = torch.clamp(probability, min=0, max=1)
        # set probability to zero for count = 0 since it may be nan when count = 0
        probability = torch.where(count == 0, 0, probability)
        # create dist and sample
        result = torch.binomial(count, probability).to(dtype=torch.int64)
        # numerical error since result is cast from float.
        return torch.clamp(result, min=torch.zeros_like(count), max=count)

    @torch.jit.export
    def generate_unique(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Generate configurations uniquely.
        see https://arxiv.org/pdf/2408.07625
        """
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None

        # x : local_batch_size * current_site * 2
        x: torch.Tensor = torch.zeros([1, 1, 2], device=device, dtype=torch.int64)  # site=1, since the first is bos
        # (un)perturbed_log_probability : local_batch_size
        unperturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        perturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        for i in range(self.sites):
            local_batch_size: int = x.size(0)

            # xi: batch * sites=1 * 2
            xi: torch.Tensor = x[:, -1:]
            # xi4: batch * sites=1
            xi4: torch.Tensor = xi[:, :, 0] * 2 + xi[:, :, 1]
            # emb: batch * sites=1 * embedding
            emb: torch.Tensor = self.embedding(xi4, i)
            # post_transformers: batch * sites=1 * embedding
            post_transformers, cache = self.transformers(emb, cache, None)
            assert cache is not None
            # tail: batch * sites=1 * 8
            tail: torch.Tensor = self.tail(post_transformers)

            # the first 4 item are amplitude
            # delta_amplitude: batch * 2 * 2
            delta_amplitude: torch.Tensor = tail[:, :, :4].reshape([local_batch_size, 2, 2])
            # filter mask for amplitude
            delta_amplitude: torch.Tensor = delta_amplitude + torch.where(self.mask(x[:, 1:]), 0, -torch.inf)
            # normalize amplitude
            delta_amplitude: torch.Tensor = self.normalize_amplitude(delta_amplitude)

            # delta unperturbed prob for all batch and 4 adds
            l: torch.Tensor = (2 * delta_amplitude).reshape([local_batch_size, 4])
            # and add to get the current unperturbed prob
            l: torch.Tensor = unperturbed_probability.view([-1, 1]) + l
            # get perturbed prob
            L: torch.Tensor = l - (-torch.rand_like(l).log()).log()
            # get max perturbed prob
            Z: torch.Tensor = L.max(dim=-1).values.reshape([-1, 1])
            # evaluate the conditioned prob
            L: torch.Tensor = -torch.log(torch.exp(-perturbed_probability.view([-1, 1])) - torch.exp(-Z) + torch.exp(-L))

            # calculate appended configurations for 4 adds
            # local_batch_size * current_site * 2 + local_batch_size * 1 * 2
            x0: torch.Tensor = torch.cat([x, torch.tensor([[0, 0]], device=device).expand(local_batch_size, -1, -1)], dim=1)
            x1: torch.Tensor = torch.cat([x, torch.tensor([[0, 1]], device=device).expand(local_batch_size, -1, -1)], dim=1)
            x2: torch.Tensor = torch.cat([x, torch.tensor([[1, 0]], device=device).expand(local_batch_size, -1, -1)], dim=1)
            x3: torch.Tensor = torch.cat([x, torch.tensor([[1, 1]], device=device).expand(local_batch_size, -1, -1)], dim=1)

            # cat all configurations to get x : new_local_batch_size * current_size * 2
            # (un)perturbed prob : new_local_batch_size
            x = torch.cat([x0, x1, x2, x3])
            unperturbed_probability = l.permute(1, 0).reshape([-1])
            perturbed_probability = L.permute(1, 0).reshape([-1])
            cache = [(kv[0].repeat(4, 1, 1, 1), kv[1].repeat(4, 1, 1, 1)) for kv in cache]
            # kv cache: batch * heads_num * site * heads_dim, so just repeat first dimension

            # filter new data, only used largest batch_size ones
            selected = perturbed_probability.sort(descending=True).indices[:batch_size]
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]
            cache = [(kv[0][selected], kv[1][selected]) for kv in cache]

            # if prob = 0, filter it forcely
            selected = perturbed_probability != -torch.inf
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]
            cache = [(kv[0][selected], kv[1][selected]) for kv in cache]

        # apply ordering
        x = torch.index_select(x[:, 1:], 1, self.ordering)
        # flatten site part of x
        x = x.reshape([x.size(0), self.double_sites])
        # it should return configurations, amplitudes, probabilities and multiplicities
        # but it is unique generator, so last two field is none
        return x, self(x), None, None
