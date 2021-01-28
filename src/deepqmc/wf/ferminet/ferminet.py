import logging

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from deepqmc import Molecule
from deepqmc.physics import pairwise_diffs, pairwise_distance
from deepqmc.plugins import PLUGINS
from deepqmc.torchext import sloglindet, triu_flat
from deepqmc.wf import WaveFunction

from deepqmc.wf.paulinet.molorb import MolecularOrbital
from deepqmc.wf.paulinet.gto import GTOBasis
from deepqmc.wf.paulinet.pyscfext import pyscf_from_mol
from deepqmc.physics import pairwise_diffs, pairwise_distance
from deepqmc.sampling import LangevinSampler, sample_wf

# new packages
import inspect
from typing import Tuple

'''
useful resources
type hinting in pytorch: https://pytorch.org/docs/stable/jit_language_reference.html
'''

__version__ = '0.2.0'
__all__ = ['FermiNet']

log = logging.getLogger(__name__)


class FermiNet(WaveFunction):
    r""" Implements the FermiNet wave function Ansatz based on [pfau2020ab]

    Derived from :class:`WaveFunction`. This constructor provides a low-level interface.

    .. math:

    Usage:
        mol = Molecule.from_name('Be')
        net = FermiNet(mol)

    Args:
        mol (:class:`~deepqmc.Molecule`): molecule whose wave function is represented
        n_layers (int): number of hidden layers
        nf_hidden_single (int): number of nodes in hidden layers of single streams
        nf_hidden_pairwise (int): number of nodes in hidden layers of pairwise streams
        n_determinants (int): number of determinants
    """

    def __init__(self,
                 mol,

                 n_layers: int = 2,
                 nf_hidden_single: int = 32,
                 nf_hidden_pairwise: int = 8,
                 n_determinants: int = 4,

                 return_log: bool = True,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super().__init__(mol)
        self.dtype = dtype
        self.device = device

        # system
        n_elec = int(mol.charges)
        self.r_atoms = mol.coords.to(device=device, dtype=dtype)
        self.n_elec = int(n_elec)
        self.n_pairwise = int(n_elec**2 - n_elec)
        self.n_atoms = len(self.mol)

        # wave function mode set and structure
        '''
        s_in: 3d + distance for each atom
        p_in: 3d + distance
        s_mixed_in: outputs s_in plus 2 blocks of p_in AND 2 blocks of s_in
        '''
        self.return_log = return_log
        self.n_layers = n_layers
        s_in = 4 * self.n_atoms
        p_in = 4
        s_hidden = nf_hidden_single
        self.s_hidden = s_hidden
        p_hidden = nf_hidden_pairwise
        self.p_hidden = p_hidden
        s_mixed_in = s_in + p_in * 2
        s_mixed = nf_hidden_single + nf_hidden_pairwise * 2
        self.n_determinants = n_determinants

        self.mix_in = Mixer(s_in, p_in, n_elec, self.n_up)
        self.split_in = LinearSplit(2 * s_in, s_hidden)
        self.stream_s0 = LinearSingle(s_mixed_in + 1, s_hidden, n_elec)
        self.stream_p0 = LinearPairwise(p_in + 1, p_hidden, self.n_pairwise)
        self.m0 = Mixer(s_hidden, p_hidden, n_elec, self.n_up)

        '''
        initialisation of the envelopes
        + 1 is to add a bias term
        linear and linearsplit contain bias and no bias, respectively
        '''
        self.single_intermediate = \
            torch.nn.ModuleList([LinearSingle(s_mixed + 1, s_hidden, n_elec) for _ in range(n_layers)])
        self.pairwise_intermediate = \
            torch.nn.ModuleList([LinearPairwise(p_hidden + 1, p_hidden, self.n_pairwise) for _ in range(n_layers)])
        self.single_splits = \
            torch.nn.ModuleList([LinearSplit(2 * s_hidden, s_hidden) for _ in range(n_layers)])
        self.mix_intermediate = Mixer(s_hidden, p_hidden, n_elec, self.n_up)

        self.env_up_linear = EnvelopeLinear(s_mixed + 2 * s_hidden + 1, self.n_up, n_determinants)
        self.env_up_sigma = EnvelopeSigma(self.n_up, n_determinants, self.n_atoms)
        self.env_up_pi = EnvelopePi(self.n_up, n_determinants, self.n_atoms)

        self.env_down_linear = EnvelopeLinear(s_mixed + 2 * s_hidden + 1, self.n_down, n_determinants)
        self.env_down_sigma = EnvelopeSigma(self.n_down, n_determinants, self.n_atoms)
        self.env_down_pi = EnvelopePi(self.n_down, n_determinants, self.n_atoms)

        self.logabssumdet = LogAbsSumDet(n_determinants, self.n_up, self.n_down, device, dtype) # og
        # self.sloglindet_layer = SLogLinDet(n_determinants) # deepqmc method

        # there's a better way of doing this
        self.pretraining = False

    def forward(self, samples: torch.Tensor) -> Tuple:
        if samples.device != self.device:
            samples = samples.to(device=self.device)

        n_samples = int(samples.shape[0])

        single_input_residual = torch.zeros((n_samples, self.n_elec, self.s_hidden),
                                            device=self.device, dtype=self.dtype)
        pairwise_input_residual = torch.zeros((n_samples, self.n_pairwise, self.p_hidden),
                                              device=self.device, dtype=self.dtype)

        ae_vectors = compute_ae_vectors(self.r_atoms, samples)

        # the inputs
        single, pairwise = compute_inputs(samples, n_samples, ae_vectors, self.n_atoms, self.n_elec)
        single_mixed, single_split = self.mix_in(single, pairwise)
        single_split = self.split_in(single_split)

        # first layer
        single = self.stream_s0(single_mixed, single_split, single_input_residual)
        pairwise = self.stream_p0(pairwise, pairwise_input_residual)
        single_mixed, single_split = self.m0(single, pairwise)

        # intermediate layers
        for ss, ps, ls in zip(self.single_intermediate, self.pairwise_intermediate, self.single_splits):
            single_split = ls(single_split)
            single = ss(single_mixed, single_split, single)
            pairwise = ps(pairwise, pairwise)
            single_mixed, single_split = self.mix_intermediate(single, pairwise)
        single_mixed = torch.cat((single_mixed, single_split.repeat(1, self.n_elec, 1)), dim=2)

        # envelopes
        ae_vectors_up, ae_vectors_down = ae_vectors.split([self.n_up, self.n_down], dim=1)
        data_up, data_down = single_mixed.split([self.n_up, self.n_down], dim=1)

        factor_up = self.env_up_linear(data_up)
        factor_down = self.env_down_linear(data_down)

        exponent_up = self.env_up_sigma(ae_vectors_up)
        exponent_down = self.env_down_sigma(ae_vectors_down)

        det_up = self.env_up_pi(factor_up, exponent_up)
        det_down = self.env_down_pi(factor_down, exponent_down)

        if self.pretraining:
            return det_up, det_down

        # og
        log_psi, sign = self.logabssumdet(det_up, det_down)

        # deepqmc method
        # sign, log_psi = self.sloglindet_layer(det_up, det_down)

        return (log_psi.squeeze(), sign.squeeze()) if self.return_log else (log_psi, )


def compute_inputs(r_electrons: torch.Tensor,
                   n_samples: int,
                   ae_vectors: torch.Tensor,
                   n_atoms: int,
                   n_elec: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # r_atoms: (n_atoms, 3)
    # r_electrons: (n_samples, n_elec, 3)
    # ae_vectors: (n_samples, n_elec, n_atoms, 3)
    ae_distances = torch.norm(ae_vectors, dim=-1, keepdim=True)
    single_inputs = torch.cat((ae_vectors, ae_distances), dim=-1)
    single_inputs = single_inputs.view((-1, n_elec, 4 * n_atoms))

    re1 = r_electrons.unsqueeze(2)
    re2 = re1.permute((0, 2, 1, 3))
    ee_vectors = re1 - re2

    mask = torch.eye(n_elec, dtype=torch.bool)
    mask = ~mask.unsqueeze(0).unsqueeze(3).repeat((n_samples, 1, 1, 3))

    ee_vectors = ee_vectors[mask]
    ee_vectors = ee_vectors.view((-1, int(n_elec ** 2 - n_elec), 3))
    ee_distances = torch.norm(ee_vectors, dim=-1, keepdim=True)

    pairwise_inputs = torch.cat((ee_vectors, ee_distances), dim=-1)

    return single_inputs, pairwise_inputs


def compute_ae_vectors(r_atoms: torch.Tensor, r_electrons: torch.Tensor) -> torch.Tensor:
    # ae_vectors (n_samples, n_elec, n_atoms, 3)
    r_atoms = r_atoms.unsqueeze(1)
    r_electrons = r_electrons.unsqueeze(2)
    ae_vectors = r_electrons - r_atoms
    return ae_vectors


class LinearSingle(nn.Module):
    r""" Implements a LinearSingle layer from []

    Usage:
        ss = LinearSingle(s_mixed + 1, s_hidden, n_elec)

    Args:
        in_dim (int): the mixed variables dimension plus one bias
        out_dim (int): number of hidden nodes
        n_elec (int): number of electrons
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 n_elec: int,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(LinearSingle, self).__init__()

        # dimension
        self.n_elec = n_elec

        # initialization
        w = torch.empty(in_dim - 1, out_dim)
        b = torch.empty(1, out_dim)
        nn.init.orthogonal_(w)
        nn.init.normal_(b)
        w = torch.cat((w, b), dim=0)
        self.w = nn.Parameter(w.to(device=device, dtype=dtype))

    def forward(self, data: torch.Tensor, data_split: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        bias = torch.ones((data.shape[0], self.n_elec, 1), device=data.device, dtype=data.dtype)
        data_w_bias = torch.cat((data, bias), dim=-1)
        pre_activation = data_w_bias @ self.w + data_split
        output = pre_activation.tanh() + residual
        return output


class LinearPairwise(nn.Module):
    r""" Implements a LinearPairwise layer from []

        Usage:
            pp = LinearPairwise(p_hidden + 1, p_hidden, n_pairwise)

        Args:
            in_dim (int): the mixed variables dimension plus one bias
            out_dim (int): number of hidden nodes
            n_elec (int): number of electrons
        """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 n_elec: int,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(LinearPairwise, self).__init__()

        # dimension
        self.n_elec = n_elec

        # initialization
        w = torch.empty(in_dim - 1, out_dim)
        b = torch.empty(1, out_dim)
        nn.init.orthogonal_(w)
        nn.init.normal_(b)
        w = torch.cat((w, b), dim=0)
        self.w = nn.Parameter(w.to(device=device, dtype=dtype))

    def forward(self, data: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        bias = torch.ones((data.shape[0], self.n_elec, 1), device=data.device, dtype=data.dtype)
        data_w_bias = torch.cat((data, bias), dim=-1)
        pre_activation = data_w_bias @ self.w
        output = pre_activation.tanh() + residual
        return output


class LinearSplit(nn.Module):
    r""" Implements a LinearSplit layer from []

    Usage:
        ls = LinearSplit(2 * s_hidden, s_hidden)

    Args:
        in_dim (int): the mixed variables dimension plus one bias
        out_dim (int): number of hidden nodes
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(LinearSplit, self).__init__()
        self.device = device
        self.dtype = dtype

        w = torch.empty(in_dim, out_dim)
        nn.init.orthogonal_(w)
        self.w = nn.Parameter(w.to(device=device, dtype=dtype))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = data @ self.w
        return output


class Mixer(nn.Module):
    r""" Implements a Mixer layer from [], a permutation equivariant function on the hidden variables from the single
    and pairwise streams

    Usage:
        mix = Mixer(s_hidden, p_hidden, n_elec, n_up)

    Args:
        n_single_features (int): the mixed variables dimension plus one bias
        n_pairwise_features (int): number of hidden nodes
        n_elec (int): number of electrons
        n_up (int): number of electrons with spin up
    """
    def __init__(self,
                 n_single_features: int,
                 n_pairwise_features: int,
                 n_elec: int,
                 n_up: int,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(Mixer, self).__init__()

        n_spin_down = n_elec - n_up
        n_pairwise = n_elec**2

        self.n_elec = n_elec
        self.n_spin_up = float(n_up)
        self.n_spin_down = float(n_spin_down)

        tmp1 = torch.ones((1, n_up, n_single_features), dtype=torch.bool, device=device)
        tmp2 = torch.zeros((1, n_spin_down, n_single_features), dtype=torch.bool, device=device)
        self.spin_up_mask = torch.cat((tmp1, tmp2), dim=1).to(device=device, dtype=dtype)
        self.spin_down_mask = (~torch.cat((tmp1, tmp2), dim=1)).to(device=device, dtype=dtype)

        self.pairwise_spin_up_mask, self.pairwise_spin_down_mask = \
            generate_pairwise_masks(n_elec, n_pairwise, n_up, n_pairwise_features)
        self.pairwise_spin_up_mask = self.pairwise_spin_up_mask.to(device=device, dtype=dtype)
        self.pairwise_spin_down_mask = self.pairwise_spin_down_mask.to(device=device, dtype=dtype)

    def forward(self, single: torch.Tensor, pairwise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # single (n_samples, n_elec, n_single_features)
        # pairwise (n_samples, n_elec, n_pairwise_features)
        # spin_up_mask = self.spin_up_mask.repeat((n_samples, 1, 1))
        # spin_down_mask = self.spin_down_mask.repeat((n_samples, 1, 1))
        # --- Single summations
        # up
        sum_spin_up = self.spin_up_mask * single
        sum_spin_up = sum_spin_up.sum(1, keepdim=True) / self.n_spin_up

        # down
        sum_spin_down = self.spin_down_mask * single
        sum_spin_down = sum_spin_down.sum(1, keepdim=True) / self.n_spin_down

        # --- Pairwise summations
        sum_pairwise = pairwise.unsqueeze(1).repeat((1, self.n_elec, 1, 1))

        # up
        sum_pairwise_up = self.pairwise_spin_up_mask * sum_pairwise
        sum_pairwise_up = sum_pairwise_up.sum(2) / self.n_spin_up

        # down
        sum_pairwise_down = self.pairwise_spin_down_mask * sum_pairwise
        sum_pairwise_down = sum_pairwise_down.sum(2) / self.n_spin_down

        features = torch.cat((single, sum_pairwise_up, sum_pairwise_down), dim=2)
        split_features = torch.cat((sum_spin_up, sum_spin_down), dim=2)
        return features, split_features


def generate_pairwise_masks(n_elec: int, n_pairwise: int, n_up: int, n_pairwise_features: int) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    r""" Generates masks for summing over the pairwise terms in a spin dependent way

    Usage:
        mask_up, mask_down = generate_pairwise_masks(n_elec, n_pairwise, n_up, n_pairwise_features)

    Args:
        n_pairwise (int): number of pairwise terms, n_elec**2 - n_elec
        n_elec (int): number of electrons
        n_up (int): number of electrons with spin up
        n_pairwise_features (int): number hidden nodes in pairwise streams

    Returns:
        spin_up_mask (tc.Tensor): mask on contributing terms to spin up pairwise terms
        spin_down_mask (tc.Tensor): mask on contributing terms to spin down pairwise terms
    """

    eye_mask = ~np.eye(n_elec, dtype=np.bool)
    ups = np.ones(n_elec, dtype=np.bool)
    ups[n_up:] = False
    downs = ~ups

    spin_up_mask = []
    spin_down_mask = []
    mask = np.zeros((n_elec, n_elec), dtype=np.bool)

    for electron in range(n_elec):

        mask_up = np.copy(mask)
        mask_up[electron, :] = ups
        mask_up = mask_up[eye_mask].reshape(-1)

        mask_down = np.copy(mask)
        mask_down[electron, :] = downs
        mask_down = mask_down[eye_mask].reshape(-1)

        spin_up_mask.append(mask_up)
        spin_down_mask.append(mask_down)

    spin_up_mask = torch.tensor(spin_up_mask, dtype=torch.bool)
    # (n_samples, n_elec, n_elec, n_pairwise_features)
    spin_up_mask = spin_up_mask.view((1, n_elec, n_pairwise-n_elec, 1))
    spin_up_mask = spin_up_mask.repeat((1, 1, 1, n_pairwise_features))

    spin_down_mask = torch.tensor(spin_down_mask, dtype=torch.bool)
    spin_down_mask = spin_down_mask.view((1, n_elec, n_pairwise-n_elec, 1))
    spin_down_mask = spin_down_mask.repeat((1, 1, 1, n_pairwise_features))

    return spin_up_mask, spin_down_mask


class EnvelopeLinear(nn.Module):
    r""" Implements the linear layer in the Envelopes from []

    Usage:
        env_up_linear = EnvelopeLinear(s_mixed + 2 * s_hidden + 1, n_up, n_determinants)

    Args:
        n_hidden (int): number of hidden nodes at the input dimension of the layer
        n_spin_det (int): number of spins in this determinant
        n_determinants (int): number of determinants
    """
    def __init__(self,
                 n_hidden: int,
                 n_spin_det: int,
                 n_determinants: int,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(EnvelopeLinear, self).__init__()

        # dimension
        self.n_spin_det = n_spin_det
        self.n_determinants = n_determinants

        # initialize
        w = torch.empty((n_hidden - 1, n_determinants * n_spin_det))
        b = torch.empty((1, n_determinants * n_spin_det))
        nn.init.orthogonal_(w)
        nn.init.normal_(b)
        w = torch.cat((w, b), dim=0)
        self.w = nn.Parameter(w.to(device=device, dtype=dtype), requires_grad=True)

    def forward(self, data):
        bias = torch.ones((data.shape[0], self.n_spin_det, 1), device=data.device, dtype=data.dtype)
        data_w_bias = torch.cat((data, bias), dim=-1)
        data_w_bias = data_w_bias.reshape((-1, data_w_bias.shape[-1]))
        output = data_w_bias @ self.w  # (n*j, k*i)
        return output.view((-1, self.n_spin_det, self.n_determinants, self.n_spin_det))


def orthogonal_init_tensor(shape: Tuple, name: str) -> torch.Tensor:
    r"""
    This is required because the sublayers of the einsum layers need to be initialised independently
    hack much yes
    """
    if name == 'sigma': # kim33
        kdim = []
        for k in range(shape[0]):
            idim = []
            for i in range(shape[1]):
                mdim = []
                for m in range(shape[2]):
                    t = torch.empty(3, 3)
                    nn.init.orthogonal_(t)
                    mdim.append(t)
                idim.append(torch.stack(mdim))
            kdim.append(torch.stack(idim))
        tensor = torch.stack(kdim)

    elif name == 'pi':  # kims
        kdim = []
        for k in range(shape[0]):
            idim = []
            for i in range(shape[1]):
                t = torch.empty(shape[2])
                nn.init.normal_(t)
                idim.append(t)
            kdim.append(torch.stack(idim))
        tensor = torch.stack(kdim)

    else:
        print('Layer not a valid choice for orthogonal initialisation.')

    return tensor


class EnvelopeSigma(nn.Module):
    r""" Implements the sigma layer in the Envelopes from []

    Usage:
        env_up_sigma = EnvelopeSigma(n_up, n_determinants, n_atoms)

    Args:
        n_spin_det (int): number of spins in this determinant
        n_determinants (int): number of determinants
        n_atoms (int): number of atoms in system
    """
    def __init__(self,
                 n_spin_det: int,
                 n_determinants: int,
                 n_atoms: int,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(EnvelopeSigma, self).__init__()

        sigma_shape = (n_determinants, n_spin_det, n_atoms, 3, 3)
        sigma = orthogonal_init_tensor(sigma_shape, 'sigma')
        self.sigma = nn.Parameter(sigma.to(device=device, dtype=dtype), requires_grad=True)

    def forward(self, ae_vectors: torch.Tensor) -> torch.Tensor:
        pre_activation = torch.einsum('njmv, kimvc -> njkimc', ae_vectors, self.sigma)
        exponential = torch.exp(-torch.norm(pre_activation, dim=-1))
        return exponential


class EnvelopePi(nn.Module):
    r""" Implements the pi layer in the Envelopes from []

    Usage:
        env_up_pi = EnvelopePi(n_up, n_determinants, n_atoms)

    Args:
        n_spin_det (int): number of spins in this determinant
        n_determinants (int): number of determinants
        n_atoms (int): number of atoms in system
    """
    def __init__(self,
                 n_spin_det: int,
                 n_determinants: int,
                 n_atoms: int,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(EnvelopePi, self).__init__()

        # dimension
        self.n_determinants = n_determinants
        self.n_spin_det = n_spin_det
        self.n_atoms = n_atoms

        pi_shape = (n_determinants, n_spin_det, n_atoms)
        pi = orthogonal_init_tensor(pi_shape, 'pi')
        self.pi = nn.Parameter(pi.to(device=device, dtype=dtype), requires_grad=True)

    def forward(self, factor: torch.Tensor, exponential: torch.Tensor) -> torch.Tensor:
        exp = torch.einsum('njkim, kim -> njki', exponential, self.pi).contiguous()
        output = factor * exp
        output = output.permute((0, 2, 3, 1)).contiguous()
        return output


class SLogLinDet(nn.Module):
    r""" Constructs a custom stable log determinant from [HermannNC20]_

    Usage:
        sloglindet_layer = SLogLinDet(n_determinants)

    Args:
        n_determinants (int): number of determinants
    """
    def __init__(self,
                 n_determinants,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(SLogLinDet, self).__init__()
        w = torch.empty((n_determinants,))
        w.fill_(1. / n_determinants)
        self.w = nn.Parameter(w.to(device=device, dtype=dtype), requires_grad=True)

    def forward(self, det_up: torch.Tensor, det_down: torch.Tensor) -> Tuple:
        sign, log_psi = sloglindet(self.w, det_up, det_down)
        sign = sign.detach()
        return sign, log_psi


class LogAbsSumDet(nn.Module):
    r""" Implements a custom stable log determinant from [pfau2020ab]

    Usage:
        logabssumdet = LogAbsSumDet(n_determinants, n_up, n_down)

    Args:
        n_determinants (int): number of determinants
        n_up (int): number spin up electrons
        n_down (int): number spin down electrons
    """
    def __init__(self,
                 n_determinants: int,
                 n_up: int,
                 n_spin_down: int,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(LogAbsSumDet, self).__init__()

        self.device = device

        w = torch.empty((n_determinants, 1))
        w.fill_(1. / n_determinants)
        self.w = nn.Parameter(w.to(device=device, dtype=dtype), requires_grad=True)
        self.detgradgrad_masks = generate_detgradgrad_masks(n_determinants, n_up, n_spin_down, device)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_psi, sign, self._activation, self._sensitivity = \
            LogAbsSumDetFunction.apply(a, b, self.w, self.detgradgrad_masks)
        return log_psi, sign


class LogAbsSumDetFunction(Function):
    r""" Implements the forward pass of LogAbsSumDet

    Usage:
        LogAbsSumDetFunction.apply(a, b, w, detgradgrad_masks)

    Args:
        a (torch.Tensor): spin up determinants (n_samples, n_determinants, n_up, n_up)
        b (torch.Tensor): spin down determinants (n_samples, n_determinants, n_down, n_down)
        w (torch.nn.Parameter): layer parameters (n_determinants, 1)
        masks (tc.Tensor): masks for the second order derivatives of the determinants

    Returns:
        log_psi (torch.Tensor): log amplitudes of the configurations
        sign_unshifted_sum (torch.Tensor): the sign of the amplitudes
        activations (torch.Tensor): activations of layer (for kfac)
        sensitivities (torch.Tensor): sensitivities of layer (for kfac)
    """
    @staticmethod
    def forward(ctx, a, b, w, masks):
        sign_a, logdet_a = torch.slogdet(a)
        sign_b, logdet_b = torch.slogdet(b)

        x = logdet_a + logdet_b
        xmax = torch.max(x, dim=-1, keepdim=True)[0]

        unshifted_exp = sign_a * sign_b * torch.exp(x)
        unshifted_exp_w = unshifted_exp @ w
        sign_unshifted_sum = torch.sign(unshifted_exp_w)

        exponent = x - xmax
        shifted_exp = sign_a * sign_b * torch.exp(exponent)
        shifted_sum = shifted_exp @ w
        log_psi = shifted_sum.abs().log() + xmax

        activations = shifted_exp
        sensitivities = 1. / shifted_sum
        ctx.masks = masks
        ctx.save_for_backward(a,
                              b,
                              w,
                              shifted_sum,
                              sign_unshifted_sum,
                              sign_a,
                              sign_b,
                              logdet_a,
                              logdet_b,
                              x,
                              xmax,
                              log_psi)

        return log_psi, sign_unshifted_sum, activations, sensitivities

    @staticmethod
    def backward(ctx, dy, _, __, ___):
        a, b, w, shifted_sum, sign_unshifted_sum, sign_a, sign_b, logdet_a, logdet_b, x, xmax, log_psi = ctx.saved_tensors
        cache = (shifted_sum, sign_unshifted_sum, sign_a, sign_b, logdet_a, logdet_b, x, xmax, log_psi, ctx.masks)
        da, db, dw = LogAbsSumDetFirst.apply(a, b, w, dy, cache)
        return da, db, dw, None


class LogAbsSumDetFirst(Function):
    r""" Implements the backward pass of LogAbsSumDet

    Usage:
        da, db, dw = LogAbsSumDetFirst.apply(a, b, w, dy, cache)

    Args:
        ctx (Tuple): self of class for saving precomputed values
        a (torch.Tensor): spin up determinants (n_samples, n_determinants, n_up, n_up)
        b (torch.Tensor): spin down determinants (n_samples, n_determinants, n_down, n_down)
        w (torch.nn.Parameter): layer parameters (n_determinants, 1)
        dy (tc.Tensor): contains backpropagated errors
        cache (Tuple): contains precomputed values for the backpass

    Returns:
        da (torch.Tensor): backpropagated errors of a
        db (torch.Tensor): backpropagated errors of b
        dw (torch.Tensor): backpropagated errors of w
    """

    @staticmethod
    def forward(ctx, a, b, w, dy, cache):
        shifted_sum, sign_unshifted_sum, sign_a, sign_b, logdet_a, logdet_b, x, xmax, log_psi, masks = cache

        ddeta, detacache = detgrad(a)
        ddetb, detbcache = detgrad(b)

        w_ = w  # cache the original w tensor, why does this work? wizardry
        w = w.unsqueeze(0).squeeze(-1)
        dfddeta = w * sign_unshifted_sum * sign_b * torch.exp(logdet_b - log_psi)
        dfddetb = w * sign_unshifted_sum * sign_a * torch.exp(logdet_a - log_psi)

        dfddeta = dfddeta.unsqueeze(-1).unsqueeze(-1)
        dfddetb = dfddetb.unsqueeze(-1).unsqueeze(-1)

        da = dfddeta * ddeta
        db = dfddetb * ddetb
        dw = (sign_a * sign_b * torch.exp(x - xmax) / shifted_sum).unsqueeze(-1)
        dy = dy.unsqueeze(-1)
        # alternate math
        # dw = (sign_unshifted_sum * sign_a * sign_b * torch.exp(x - log_psi)).unsqueeze(-1)
        # print(dy.shape, dw.shape)
        dyab = dy.unsqueeze(-1)

        ctx.masks = masks
        ctx.detacache = detacache
        ctx.detbcache = detbcache
        ctx.save_for_backward(a,
                              b,
                              w_, # no idea why this works sorry not sorry
                              ddeta,
                              ddetb,
                              da,
                              db,
                              dw,
                              dfddeta,
                              dfddetb,
                              sign_unshifted_sum,
                              log_psi)
        # sum over the batch for dw as the output is directly used for the gradients of the parameters
        return dyab * da, dyab * db, (dy * dw).sum(0, keepdim=True)

    @staticmethod
    def backward(ctx, a_dash, b_dash, w_dash):
        assert ctx.needs_input_grad[-1] == False  # We dont need grads for the cache
        a, \
        b, \
        w, \
        ddeta, \
        ddetb, \
        da, \
        db, \
        dw, \
        dfddeta, \
        dfddetb, \
        sign_unshifted_sum, \
        log_psi \
            = ctx.saved_tensors
        mask_a, not_mask_a, mask_b, not_mask_b = ctx.masks
        detacache = ctx.detacache
        detbcache = ctx.detbcache
        # [print(t.device) for t in ctx.saved_tensors]
        # [print(t.device) for t in ctx.masks]
        # [print(t.device) for t in detacache]
        # [print(t.device) for t in detbcache]
        # print(a_dash.device, b_dash.device, w_dash.device)

        w = w.unsqueeze(0).unsqueeze(-1)  # (1, nk, 1, 1)
        dw = dw.unsqueeze(-1)
        w_dash = w_dash.unsqueeze(-1)
        sign_unshifted_sum = sign_unshifted_sum.unsqueeze(-1).unsqueeze(-1)
        log_psi = log_psi.unsqueeze(-1).unsqueeze(-1)

        dfddeta_w = dfddeta / w  # (n, nk, 1, 1)
        dfddetb_w = dfddetb / w  # (n, nk, 1, 1)

        ddeta_sum = matrix_sum(a_dash * ddeta)
        da_sum = matrix_sum(da * a_dash)
        ddetb_sum = matrix_sum(b_dash * ddetb)
        db_sum = matrix_sum(db * b_dash)
        a_sum = k_sum(dfddeta * ddeta_sum)
        b_sum = k_sum(dfddetb * ddetb_sum)

        # Compute second deriviate of f wrt to w
        d2w = -dw * k_sum(w_dash * dw)

        # compute deriviate of df/da wrt to w
        dadw = -dw * k_sum(da_sum)
        dadw = dadw + dfddeta_w * ddeta_sum  # i=j

        # compute derivative of df/db wrt to w
        dbdw = -dw * k_sum(db_sum)
        dbdw = dbdw + dfddetb_w * ddetb_sum  # i=j

        # Compute second derivative of f wrt to a
        d2a = -da * a_sum
        d2a = d2a + dfddeta * detgradgrad(a_dash, detacache, mask_a, not_mask_a)  # i=j
        # Compute derivative of df/db wrt to a
        dbda = -da * b_sum
        dbda = dbda + ddeta * sign_unshifted_sum * torch.exp(-log_psi) * w * ddetb_sum  # i=j
        # Compute derivative of df/dw wrt to a
        dwda = w_dash * -da * k_sum(dw)
        dwda = dwda + w_dash * da / w  # i=j

        # Compute second derivative of f wrt to b
        d2b = -db * b_sum
        d2b = d2b + dfddetb * detgradgrad(b_dash, detbcache, mask_b, not_mask_b)  # i=j
        # Compute derivative of df/da wrt to b
        dadb = -db * a_sum
        dadb = dadb + ddetb * sign_unshifted_sum * torch.exp(-log_psi) * w * ddeta_sum  # i=j
        # Compute derivative of df/dw wrt to b
        dwdb = w_dash * -db * k_sum(dw)
        dwdb = dwdb + w_dash * db / w  # i=j

        dady = (da).sum((-3, -2, -1)).unsqueeze(-1)
        dbdy = (db).sum((-3, -2, -1)).unsqueeze(-1)
        dwdy = (dw).sum(1).squeeze(1)
        ddy = dady + dbdy + dwdy

        return (d2a + dbda + dwda), \
               (d2b + dadb + dwdb), \
               (d2w + dadw + dbdw).sum(0, keepdim=True).squeeze(-1).squeeze(-2), \
               ddy, None


def generate_detgradgrad_masks(n_determinants, n_up, n_spin_down, device):
    # used in generate_p
    mask_a_tmp = generate_mask((1, n_determinants, n_up, n_up, n_up), n_up)
    mask_a = torch.from_numpy(mask_a_tmp).to(device)
    not_mask_a = torch.from_numpy(~mask_a_tmp).to(device)

    mask_b_tmp = generate_mask((1, n_determinants, n_spin_down, n_spin_down, n_spin_down), n_spin_down)
    mask_b = torch.from_numpy(mask_b_tmp).to(device)
    not_mask_b = torch.from_numpy(~mask_b_tmp).to(device)
    return mask_a, not_mask_a, mask_b, not_mask_b


def generate_mask(shape, n_dim):
    mask = np.ones(shape, dtype=np.bool)
    for i in range(n_dim):
        for j in range(n_dim):
            mask[..., i, i, j] = False
            mask[..., j, i, j] = False
    return mask


def detgrad(tensor):
    r""" Computes sensitivities of the determinant

    is much faster to compute on the cpu, afaik no fast gpu algorithm exists

    Usage:
        grad_det, (s, u, v_t, sign) = detgrad(a)

    Args:
        ctx (Tuple): self of class for saving precomputed values
        a (torch.Tensor): spin up determinants (n_samples, n_determinants, n_up, n_up)
        b (torch.Tensor): spin down determinants (n_samples, n_determinants, n_down, n_down)
        w (torch.nn.Parameter): layer parameters (n_determinants, 1)
        dy (tc.Tensor): contains backpropagated errors
        cache (Tuple): contains precomputed values for the backpass

    Returns:
        grad_det (torch.Tensor): sensitivities of the determinant
        s (torch.Tensor): singular values of svd
        u (torch.Tensor): left matrix of svd
        v_t (torch.Tensor): right matrix of svd
        sign (torch.Tensor): sign of the determinant sensitivities
    """

    device = tensor.device
    dtype = tensor.dtype

    tensor = tensor.cpu()
    u, s, v = tensor.svd()
    u, s, v = u.type(dtype).to(device, non_blocking=True), \
              s.type(dtype).to(device, non_blocking=True), \
              v.type(dtype).to(device, non_blocking=True)
    v_t = v.transpose(-2, -1)
    gamma = generate_gamma(s)
    sign = (u.det() * v.det())[..., None, None]
    grad_det = sign * ((u * gamma) @ v_t)
    return grad_det, (s, u, v_t, sign)


def generate_gamma(s):
    # product all eigenvalues other than index i
    n_egvs = s.shape[2]
    gamma = [s[:, :, :i].prod(-1) * s[:, :, i + 1:].prod(-1) for i in range(n_egvs - 1)]
    gamma.append(s[:, :, :-1].prod(-1)) # above loop does not work for last term. append last term
    gamma = torch.stack(gamma, dim=2)
    gamma = gamma.unsqueeze(2)
    return gamma


def generate_p(s, mask, not_mask):
    '''
    the mask here deletes entries relating to k \neq i, j, and the not_mask allows the product to be taken
    by replacing these 0s with 1s
    '''
    n_samples, n_k, n_dim = s.shape
    new_shape = (1, 1, 1, n_dim, n_dim)
    s = s[..., None, None]
    s = s.repeat(new_shape)
    s = s * mask + not_mask * torch.ones_like(s)
    s_prod = s.prod(-3)
    return s_prod


def detgradgrad(C_dash, cache, mask, not_mask):
    r""" Computes second order sensitivities of the determinant

    is much faster to compute on the cpu, afaik no fast gpu algorithm exists

    Usage:
        gradgrad_det = detgradgrad(C_dash, cache, mask, not_mask)

    Args:
        C_dash (torch.Tensor): reverse mode sensitivities
        cache (Tuple): contains precomputed values for the backpass
        mask (torch.Tensor): see generate_p
        not_mask (torch.Tensor): see generate_p


    Returns:
        grad_det (torch.Tensor): sensitivities of the determinant
        s (torch.Tensor): singular values of svd
        u (torch.Tensor): left matrix of svd
        v_t (torch.Tensor): right matrix of svd
        sign (torch.Tensor): sign of the determinant sensitivities
    """

    s, u, v_t, sign = cache
    M = v_t @ C_dash.transpose(-2, -1) @ u
    p = generate_p(s, mask, not_mask)
    sgn = torch.sign(sign)
    dim = M.shape[-1]
    m_jj = diag_part(M, dim).sum(-1, keepdim=True)  # n k d
    # off diagonal elements
    p_zero_diag = set_zero_diag(p, dim)  # n k d d
    xi = -M * p_zero_diag  # n k d d
    # the diagonal element of m is multiplied by all egvs in the row, remembering p is symmetric
    xi_diag = (m_jj * p_zero_diag).sum(-2, keepdim=True)
    xi = set_diag_with_vector(xi, xi_diag, dim)
    return sgn * u @ xi @ v_t


def k_sum(x):
    return x.sum(1, keepdim=True)


def matrix_sum(x):
    return x.sum((-2, -1), keepdim=True)


def set_zero_diag(tensor, dim):
    tensor = tensor * (1. - torch.eye(dim, device=tensor.device, dtype=tensor.dtype))
    return tensor


def set_diag_with_vector(tensor, diagonal, dim):
    tensor = tensor * (1. - torch.eye(dim, device=tensor.device, dtype=tensor.dtype)) \
             + diagonal * torch.eye(dim, device=tensor.device, dtype=tensor.dtype)
    return tensor


def set_diag_with_matrix(tensor, diagonal, dim):
    tensor = tensor * (1. - torch.eye(dim, device=tensor.device, dtype=tensor.dtype)) + diagonal
    return tensor


def diag_part(tensor, dim):
    tensor = tensor * torch.eye(dim, device=tensor.device, dtype=tensor.dtype)
    return tensor






