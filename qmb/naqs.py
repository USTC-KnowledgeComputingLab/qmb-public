# This file implements naqs from https://arxiv.org/pdf/2109.12606

import torch


class FakeLinear(torch.nn.Module):
    """
    Fake linear layer where dim_in = 0, avoid pytorch warning in initialization
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        assert dim_in == 0
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.zeros([dim_out]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, zero = x.shape
        return self.bias.view([1, -1]).expand([batch, -1])


def Linear(dim_in: int, dim_out: int) -> torch.nn.Module:
    # avoid torch warning when initialize a linear layer with dim_in = 0
    if dim_in == 0:
        return FakeLinear(dim_in, dim_out)
    else:
        return torch.nn.Linear(dim_in, dim_out)


class MLP(torch.nn.Module):
    """
    This module implements multiple layers MLP with given dim_input, dim_output and hidden_size
    """

    def __init__(self, dim_input: int, dim_output: int, hidden_size: tuple[int, ...]) -> None:
        super().__init__()
        self.dim_input: int = dim_input
        self.dim_output: int = dim_output
        self.hidden_size: tuple[int, ...] = hidden_size
        self.depth: int = len(hidden_size)

        dimensions: list[int] = [dim_input] + list(hidden_size) + [dim_output]
        linears: list[torch.nn.Module] = [Linear(i, j) for i, j in zip(dimensions[:-1], dimensions[1:])]
        modules: list[torch.nn.Module] = [module for linear in linears for module in (linear, torch.nn.SiLU())][:-1]
        self.model: torch.nn.Module = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class WaveFunction(torch.nn.Module):
    """
    This module implements naqs from https://arxiv.org/pdf/2109.12606
    """

    def __init__(
            self,
            *,
            double_sites: int,  # qubits number, qubits are grouped by two for each site in naqs, so name it as double sites
            physical_dim: int,  # is always 2 for naqs
            is_complex: bool,  # is always true for naqs
            spin_up: int,  # spin up number
            spin_down: int,  # spin down number
            hidden_size: tuple[int, ...],  # hidden size for MLP
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
        self.hidden_size: tuple[int, ...] = hidden_size

        # The amplitude and phase network for each site
        # each of them accept qubits before them and output vector with dimension of 4 as the configuration of two qubits on the current site.
        self.amplitude: torch.nn.ModuleList = torch.nn.ModuleList([MLP(i * 2, 4, self.hidden_size) for i in range(self.sites)])
        self.phase: torch.nn.ModuleList = torch.nn.ModuleList([MLP(i * 2, 4, self.hidden_size) for i in range(self.sites)])

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
        # x : batch_size * 2 * 2
        # param :  batch_size
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

        x_float: torch.Tensor = x.to(dtype=dtype)
        arange: torch.Tensor = torch.arange(batch_size, device=device)
        total_amplitude: torch.Tensor = torch.zeros([batch_size], device=device, dtype=dtype)
        total_phase: torch.Tensor = torch.zeros([batch_size], device=device, dtype=dtype)
        for i, amplitude_phase_m in enumerate(zip(self.amplitude, self.phase)):
            amplitude_m, phase_m = amplitude_phase_m
            # delta_amplitude/phase : batch * 2 * 2
            # delta amplitude and phase for the configurations at new site.
            delta_amplitude: torch.Tensor = amplitude_m(x_float[:, :i].reshape([batch_size, 2 * i])).reshape([batch_size, 2, 2])
            delta_phase: torch.Tensor = phase_m(x_float[:, :i].reshape([batch_size, 2 * i])).reshape([batch_size, 2, 2])
            # filter mask for amplitude
            delta_amplitude: torch.Tensor = delta_amplitude + torch.where(self.mask(x[:, :i]), 0, -torch.inf)
            # normalize amplitude
            delta_amplitude: torch.Tensor = self.normalize_amplitude(delta_amplitude)
            # delta_amplitude/phase : batch
            delta_amplitude: torch.Tensor = delta_amplitude[arange, x[:, i, 0], x[:, i, 1]]
            delta_phase: torch.Tensor = delta_phase[arange, x[:, i, 0], x[:, i, 1]]
            # calculate total amplitude and phase
            total_amplitude = total_amplitude + delta_amplitude
            total_phase = total_phase + delta_phase
        return torch.view_as_complex(torch.stack([total_amplitude, total_phase], dim=-1)).exp()

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

        # x : local_batch_size * current_site * 2
        x: torch.Tensor = torch.empty([1, 0, 2], device=device, dtype=torch.int64)
        # (un)perturbed_log_probability : local_batch_size
        unperturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        perturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        for i, amplitude_m in enumerate(self.amplitude):
            local_batch_size: int = x.shape[0]
            x_float: torch.Tensor = x.to(dtype=dtype)
            # delta_amplitude : batch * 2 * 2
            # delta amplitude for the configurations at new site.
            delta_amplitude: torch.Tensor = amplitude_m(x_float.reshape([local_batch_size, 2 * i])).reshape([local_batch_size, 2, 2])
            # filter mask for amplitude
            delta_amplitude: torch.Tensor = delta_amplitude + torch.where(self.mask(x), 0, -torch.inf)
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

            # filter new data, only used largest batch_size ones
            selected = perturbed_probability.sort(descending=True).indices[:batch_size]
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]

            # if prob = 0, filter it forcely
            selected = perturbed_probability != -torch.inf
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]

        # apply ordering
        x = torch.index_select(x, 1, self.ordering)
        # flatten site part of x
        x = x.reshape([x.size(0), self.double_sites])
        # it should return configurations, amplitudes, probabilities and multiplicities
        # but it is unique generator, so last two field is none
        return x, self(x), None, None


class WaveFunctionNormal(torch.nn.Module):
    """
    This module implements naqs from https://arxiv.org/pdf/2109.12606
    No subspace restrictor.
    """

    def __init__(
            self,
            *,
            sites: int,  # qubits number
            physical_dim: int,  # is always 2 for naqs
            is_complex: bool,  # is always true for naqs
            hidden_size: tuple[int, ...],  # hidden size for MLP
            ordering: int | list[int],  # ordering of sites +1 for normal order, -1 for reversed order, or the order list directly
    ) -> None:
        super().__init__()
        self.sites: int = sites
        assert physical_dim == 2
        assert is_complex == True
        self.hidden_size: tuple[int, ...] = hidden_size

        # The amplitude and phase network for each site
        # each of them accept qubits before them and output vector with dimension of 2 as the configuration of the qubit on the current site.
        self.amplitude: torch.nn.ModuleList = torch.nn.ModuleList([MLP(i, 2, self.hidden_size) for i in range(self.sites)])
        self.phase: torch.nn.ModuleList = torch.nn.ModuleList([MLP(i, 2, self.hidden_size) for i in range(self.sites)])

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
    def normalize_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize uncompleted log amplitude.
        """
        # x : batch_size * 2
        # param :  batch_size
        param = (2 * x).exp().sum(dim=[-1]).log() / 2
        x = x - param.unsqueeze(-1)
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
        # x : batch_size * sites
        x = x.reshape([batch_size, self.sites])
        # apply ordering
        x = torch.index_select(x, 1, self.ordering_reversed)

        x_float: torch.Tensor = x.to(dtype=dtype)
        arange: torch.Tensor = torch.arange(batch_size, device=device)
        total_amplitude: torch.Tensor = torch.zeros([batch_size], device=device, dtype=dtype)
        total_phase: torch.Tensor = torch.zeros([batch_size], device=device, dtype=dtype)
        for i, amplitude_phase_m in enumerate(zip(self.amplitude, self.phase)):
            amplitude_m, phase_m = amplitude_phase_m
            # delta_amplitude/phase : batch * 2
            # delta amplitude and phase for the configurations at new site.
            delta_amplitude: torch.Tensor = amplitude_m(x_float[:, :i].reshape([batch_size, i])).reshape([batch_size, 2])
            delta_phase: torch.Tensor = phase_m(x_float[:, :i].reshape([batch_size, i])).reshape([batch_size, 2])
            # normalize amplitude
            delta_amplitude: torch.Tensor = self.normalize_amplitude(delta_amplitude)
            # delta_amplitude/phase : batch
            delta_amplitude: torch.Tensor = delta_amplitude[arange, x[:, i]]
            delta_phase: torch.Tensor = delta_phase[arange, x[:, i]]
            # calculate total amplitude and phase
            total_amplitude = total_amplitude + delta_amplitude
            total_phase = total_phase + delta_phase
        return torch.view_as_complex(torch.stack([total_amplitude, total_phase], dim=-1)).exp()

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

        # x : local_batch_size * current_site
        x: torch.Tensor = torch.empty([1, 0], device=device, dtype=torch.int64)
        # (un)perturbed_log_probability : local_batch_size
        unperturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        perturbed_probability: torch.Tensor = torch.tensor([0], dtype=dtype, device=device)
        for i, amplitude_m in enumerate(self.amplitude):
            local_batch_size: int = x.shape[0]
            x_float: torch.Tensor = x.to(dtype=dtype)
            # delta_amplitude : batch * 2
            # delta amplitude for the configurations at new site.
            delta_amplitude: torch.Tensor = amplitude_m(x_float.reshape([local_batch_size, i])).reshape([local_batch_size, 2])
            # normalize amplitude
            delta_amplitude: torch.Tensor = self.normalize_amplitude(delta_amplitude)

            # delta unperturbed prob for all batch and 2 adds
            l: torch.Tensor = (2 * delta_amplitude).reshape([local_batch_size, 2])
            # and add to get the current unperturbed prob
            l: torch.Tensor = unperturbed_probability.view([-1, 1]) + l
            # get perturbed prob
            L: torch.Tensor = l - (-torch.rand_like(l).log()).log()
            # get max perturbed prob
            Z: torch.Tensor = L.max(dim=-1).values.reshape([-1, 1])
            # evaluate the conditioned prob
            L: torch.Tensor = -torch.log(torch.exp(-perturbed_probability.view([-1, 1])) - torch.exp(-Z) + torch.exp(-L))

            # calculate appended configurations for 2 adds
            # local_batch_size * current_site + local_batch_size * 1
            x0: torch.Tensor = torch.cat([x, torch.tensor([0], device=device).expand(local_batch_size, -1)], dim=1)
            x1: torch.Tensor = torch.cat([x, torch.tensor([1], device=device).expand(local_batch_size, -1)], dim=1)

            # cat all configurations to get x : new_local_batch_size * current_size * 2
            # (un)perturbed prob : new_local_batch_size
            x = torch.cat([x0, x1])
            unperturbed_probability = l.permute(1, 0).reshape([-1])
            perturbed_probability = L.permute(1, 0).reshape([-1])

            # filter new data, only used largest batch_size ones
            selected = perturbed_probability.sort(descending=True).indices[:batch_size]
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]

            # if prob = 0, filter it forcely
            selected = perturbed_probability != -torch.inf
            x = x[selected]
            unperturbed_probability = unperturbed_probability[selected]
            perturbed_probability = perturbed_probability[selected]

        # apply ordering
        x = torch.index_select(x, 1, self.ordering)
        # flatten site part of x
        x = x.reshape([x.size(0), self.sites])
        # it should return configurations, amplitudes, probabilities and multiplicities
        # but it is unique generator, so last two field is none
        return x, self(x), None, None
