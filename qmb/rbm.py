"""
This file implements the RBM network, which is used in https://arxiv.org/pdf/2208.06708.
"""

import torch
from .bitspack import pack_int, unpack_int


class RBM(torch.nn.Module):

    def __init__(self, visible_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        self.a = torch.nn.Parameter(torch.randn(visible_dim))  # Visible bias
        self.b = torch.nn.Parameter(torch.randn(hidden_dim))  # Hidden bias
        self.W = torch.nn.Parameter(torch.randn(visible_dim, hidden_dim))  # Weight matrix

    @torch.jit.export
    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Energy function E(v, h) = - (a^T v + b^T h + v^T W h)
        """
        return -(torch.einsum("v,bv->b", self.a, v) + torch.einsum("h,bh->b", self.b, h) + torch.einsum("vh,bv,bh->b", self.W, v, h))

    @torch.jit.export
    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Free energy F(v) = -log sum_h exp(-E(v, h))
        """
        wx_b = v @ self.W + self.b  # Shape: b * h
        hidden_contrib = torch.sum(torch.nn.functional.softplus(wx_b), dim=1)  # Shape: b
        return -v @ self.a - hidden_contrib

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unnormalized log-probability (negative free energy) for batch input
        """
        return -self.free_energy(x)

    @torch.jit.export
    def sample_h_by_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        Sample hidden units h given visible units v (Bernoulli sampling)
        """
        activation = v @ self.W + self.b  # Shape: b * h
        p_h_given_v = torch.sigmoid(activation)
        return torch.bernoulli(p_h_given_v)

    @torch.jit.export
    def sample_v_by_h(self, h: torch.Tensor) -> torch.Tensor:
        """
        Sample visible units v given hidden units h (Bernoulli sampling)
        """
        activation = h @ self.W.t() + self.a # Shape: b * v
        p_v_given_h = torch.sigmoid(activation)
        return torch.bernoulli(p_v_given_h)

    @torch.jit.ignore
    def gibbs_sample(self, v: torch.Tensor, k: int=1) -> torch.Tensor:
        """
        Gibbs sampling: Alternate between sampling h and v for k steps
        """
        for _ in range(k):
            h = self.sample_h_by_v(v)
            v = self.sample_v_by_h(h)
        return v

class WaveFunctionElectronUpDown(torch.nn.Module):
    """
    The wave function for the RBM network.
    This module maintains the conservation of particle number of spin-up and spin-down electrons.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
            self,
            *,
            double_sites: int,  # Number of qubits, where each pair of qubits represents a site in the MLP model
            physical_dim: int,  # Dimension of the physical space, which is always 2 for MLP
            is_complex: bool,  # Indicates whether the wave function is complex-valued, which is always true for MLP
            spin_up: int,  # Number of spin-up electrons
            spin_down: int,  # Number of spin-down electrons
            hidden_dim: int,  # Number of hidden unit in RBM
    ) -> None:
        super().__init__()
        assert double_sites % 2 == 0
        self.double_sites: int = double_sites
        self.sites: int = double_sites // 2
        assert physical_dim == 2
        assert is_complex == True  # pylint: disable=singleton-comparison
        self.spin_up: int = spin_up
        self.spin_down: int = spin_down

        self.rbm = RBM(double_sites, hidden_dim)

        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype
        
        batch_size: int = x.shape[0]
        x = unpack_int(x, size=1, last_dim=self.double_sites).view([batch_size, self.sites * 2])

        neg_log_prob = self.rbm(x.to(dtype=dtype))
        log_amplitude = - neg_log_prob / 2
        log_amplitude = log_amplitude - log_amplitude.mean()
        amplitude = log_amplitude.exp().double()
        return torch.view_as_complex(torch.stack([amplitude, torch.zeros_like(amplitude)], dim=-1))
    
    @torch.jit.ignore
    def generate_unique(self, batch_size: int, block_num: int = -1) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        assert block_num == 1
        v = torch.rand([batch_size, self.double_sites], device=device, dtype=dtype)
        v = self.rbm.gibbs_sample(v, k=1).to(dtype=torch.uint8)
        x = pack_int(v, size=1)
        x = torch.unique(x, dim=0)
        return x, self(x), None, None


import torch
import torch.nn as nn
import torch.nn.functional as F

class BitArrayToFloatCNN(nn.Module):
    def __init__(self, input_length=60):
        super(BitArrayToFloatCNN, self).__init__()
        
        # Calculate padding to maintain dimensions after convolution
        # Using kernel_size=3, stride=1, padding=1 maintains dimensions
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Adaptive pooling to handle variable input lengths
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)  # Output 2 float numbers
        
    def forward(self, x):
        # Input shape: (batch_size, input_length)
        # Add channel dimension: (batch_size, 1, input_length)
        x = x.unsqueeze(1)
        
        # Apply convolutions with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global average pooling
        x = self.pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation for final layer (regression output)
        
        return x

class CNN(torch.nn.Module):
    def __init__(
            self, dim
    ):
        super().__init__()
        self.dim = dim
        self.model = BitArrayToFloatCNN(dim)

    def forward(self, x):
        x = self.model(x).double()
        return torch.view_as_complex(x)
from torch.distributions import Categorical
class CnnWaveFunction(torch.nn.Module):
    """
    The wave function for the RBM network.
    This module maintains the conservation of particle number of spin-up and spin-down electrons.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
            self,
            *,
            double_sites: int,  # Number of qubits, where each pair of qubits represents a site in the MLP model
            physical_dim: int,  # Dimension of the physical space, which is always 2 for MLP
            is_complex: bool,  # Indicates whether the wave function is complex-valued, which is always true for MLP
            spin_up: int,  # Number of spin-up electrons
            spin_down: int,  # Number of spin-down electrons
    ) -> None:
        super().__init__()
        assert double_sites % 2 == 0
        self.double_sites: int = double_sites
        self.sites: int = double_sites // 2
        assert physical_dim == 2
        assert is_complex == True  # pylint: disable=singleton-comparison
        self.spin_up: int = spin_up
        self.spin_down: int = spin_down

        self.rbm = CNN(double_sites)

        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype
        
        batch_size: int = x.shape[0]
        x = unpack_int(x, size=1, last_dim=self.double_sites).view([batch_size, self.sites * 2])

        neg_log_prob = self.rbm(x.to(dtype=dtype))
        log_amplitude = - neg_log_prob / 2
        log_amplitude = log_amplitude - log_amplitude.mean()
        amplitude = log_amplitude.exp().double()
        return torch.view_as_complex(torch.stack([amplitude, torch.zeros_like(amplitude)], dim=-1))
    
    @torch.jit.ignore
    def generate_unique(self, batch_size: int, block_num: int = -1) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        device: torch.device = self.dummy_param.device
        dtype: torch.dtype = self.dummy_param.dtype

        assert block_num == 1
        v = torch.randint(0, 2, [batch_size, self.double_sites], device=device, dtype=torch.uint8)
        x = v.to(dtype=torch.uint8)
        y = self(pack_int(x, size=1))
        w = y.real ** 2
        w = w / w.sum()
        distribution = Categorical(w)
        samples = distribution.sample((batch_size,))
        x = x[samples]
        x = pack_int(v, size=1)
        x = torch.unique(x, dim=0)
        y = self(x)
        return x, y, None, None
