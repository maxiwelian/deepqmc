import logging

import numpy as np
import torch
from torch import nn

from deepqmc import Molecule
from deepqmc.physics import pairwise_diffs, pairwise_distance
from deepqmc.plugins import PLUGINS
from deepqmc.torchext import sloglindet, triu_flat
from deepqmc.wf import WaveFunction

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
    r"""
    Implements FermiNet

    # should really set torch.set_default_dtype() at the start of the script
    """
    def __init__(self,
                 mol,

                 n_layers: int,
                 n_electrons: int,
                 nf_hidden_single: int,
                 nf_hidden_pairwise: int,
                 n_determinants: int,

                 return_log: bool = True,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):

        super().__init__(mol)
        self.dtype = dtype
        self.device = device

        # add attributes??

        # system
        self.r_atoms = r_atoms
        self.n_electrons = int(n_electrons)
        self.n_pairwise = int(n_electrons ** 2)
        self.n_atoms = len(self.mol)

        # wave function mode set and structure
        self.return_log = return_log
        self.n_layers = n_layers
        s_in = 4 * self.n_atoms
        p_in = 4
        s_hidden = nf_hidden_single
        self.s_hidden = s_hidden
        p_hidden = nf_hidden_pairwise
        self.p_hidden = p_hidden
        s_mixed_in = 4 * self.n_atoms * 3 + 4 * 2
        s_mixed = nf_hidden_single * 3 + nf_hidden_pairwise * 2

        self.mix_in = Mixer(s_in, p_in, n_electrons, self.n_up)
        self.stream_s0 = Linear(s_mixed_in + 1, s_hidden, n_electrons)
        self.stream_p0 = Linear(p_in + 1, p_hidden, self.n_pairwise)
        self.m0 = Mixer(s_hidden, p_hidden, n_electrons, self.n_up)

        '''
        initialisation of the envelopes
        + 1 is to add a bias term
        '''
        self.single_intermediate = \
            torch.nn.ModuleList([Linear(s_mixed + 1, s_hidden, n_electrons) for _ in range(n_layers)])
        self.pairwise_intermediate = \
            torch.nn.ModuleList([Linear(p_hidden + 1, p_hidden, self.n_pairwise) for _ in range(n_layers)])
        self.mix_intermediate = Mixer(s_hidden, p_hidden, n_electrons, self.n_up)

        self.env_up_linear = EnvelopeLinear(s_mixed + 1, self.n_up, n_determinants)
        self.env_up_sigma = EnvelopeSigma(self.n_up, n_determinants, self.n_atoms)
        self.env_up_pi = EnvelopePi(self.n_up, n_determinants, self.n_atoms)

        self.env_down_linear = EnvelopeLinear(s_mixed + 1, self.n_down, n_determinants)
        self.env_down_sigma = EnvelopeSigma(self.n_down, n_determinants, self.n_atoms)
        self.env_down_pi = EnvelopePi(self.n_down, n_determinants, self.n_atoms)

    def forward(self, samples: torch.Tensor) -> Tuple:
        device = samples.device
        dtype = samples.dtype

        n_samples = int(samples.shape[0])

        diagonal_pairwise_input = torch.zeros((n_samples, self.n_electrons, 4), device=device, dtype=dtype)
        single_input_residual = torch.zeros((n_samples, self.n_electrons, self.s_hidden), device=device, dtype=dtype)
        pairwise_input_residual = torch.zeros((n_samples, self.n_pairwise, self.p_hidden), device=device, dtype=dtype)

        ae_vectors = compute_ae_vectors(self.r_atoms, samples)

        # the inputs
        single, pairwise = compute_inputs(samples, n_samples, ae_vectors, self.n_atoms, self.n_electrons)
        pairwise = torch.cat((pairwise, diagonal_pairwise_input), dim=1)
        single_mixed = self.mix_in(single, pairwise)

        # first layer
        single = self.stream_s0(single_mixed, single_input_residual)
        pairwise = self.stream_p0(pairwise, pairwise_input_residual)
        single_mixed = self.m0(single, pairwise)

        # intermediate layers
        for ss, ps in zip(self.single_intermediate, self.pairwise_intermediate):
            single = ss(single_mixed, single)
            pairwise = ps(pairwise, pairwise)
            single_mixed = self.mix_intermediate(single, pairwise)

        # envelopes
        ae_vectors_up, ae_vectors_down = ae_vectors.split([self.n_up, self.n_down], dim=1)
        data_up, data_down = single_mixed.split([self.n_up, self.n_down], dim=1)

        factor_up = self.env_up_linear(data_up)
        factor_down = self.env_down_linear(data_down)

        exponent_up = self.env_up_sigma(ae_vectors_up)
        exponent_down = self.env_down_sigma(ae_vectors_down)

        det_up = self.env_up_pi(factor_up, exponent_up)
        det_down = self.env_down_pi(factor_down, exponent_down)

        # og
        # log_psi, sign = self.logabssumdet(det_up, det_down)

        # deepqmc method
        det_up = det_up.flatten(start_dim=-4, end_dim=-3).contiguous()
        det_down = det_down.flatten(start_dim=-4, end_dim=-3).contiguous()
        conf_coeff = det_up.new_ones(1)  # WEIGHTS
        # not singular because of baseline??
        sign, psi = sloglindet(conf_coeff, det_up, det_down)
        sign = sign.detach()

        return (psi, sign) if self.return_log else (psi, )


class Linear(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 n_elec: int):
        super(Linear, self).__init__()

        # dimension
        self.n_elec = n_elec

        # initialization
        w = torch.empty(in_dim - 1, out_dim)
        b = torch.empty(1, out_dim)
        nn.init.orthogonal_(w)
        nn.init.normal_(b)
        w = torch.cat((w, b), dim=0)
        self.w = nn.Parameter(w)

    def forward(self, data: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        bias = torch.ones((data.shape[0], self.ne, 1), device=data.device, dtype=data.dtype)
        data_w_bias = torch.cat((data, bias), dim=-1)
        pre_activation = data_w_bias @ self.w
        output = pre_activation.tanh() + residual
        return output


class EnvelopeLinear(nn.Module):
    def __init__(self,
                 n_hidden: int,
                 n_spin_det: int,
                 n_determinants: int):
        super(EnvelopeLinear, self).__init__()

        # dimension
        self.n_spin_det = n_spin_det

        # initialize
        w = torch.empty((n_hidden - 1, n_determinants * n_spin_det))
        b = torch.empty((1, n_determinants * n_spin_det))
        nn.init.orthogonal_(w)
        nn.init.normal_(b)
        w = torch.cat((w, b), dim=0)
        self.w = nn.Parameter(w, requires_grad=True)

    def forward(self, data):
        bias = torch.ones((data.shape[0], self.n_spin_det, 1), device=data.device, dtype=data.dtype)
        data_w_bias = torch.cat((data, bias), dim=-1)
        output = torch.einsum('njf, kifs -> njkis', data_w_bias, self.w)
        return output


class EnvelopeSigma(nn.Module):
    def __init__(self,
                 n_spin_det: int,
                 n_determinants: int,
                 n_atoms: int):
        super(EnvelopeSigma, self).__init__()

        sigma_shape = (n_determinants, n_spin_det, n_atoms, 3, 3)
        sigma = orthogonal_init_tensor(sigma_shape, 'sigma')
        self.sigma = nn.Parameter(sigma, requires_grad=True)

    def forward(self, ae_vectors: torch.Tensor) -> torch.Tensor:
        pre_activation = torch.einsum('njmv, kimvc -> njkimc', ae_vectors, self.sigma)
        exponential = torch.exp(-torch.norm(pre_activation, dim=-1))
        return exponential


class EnvelopePi(nn.Module):
    def __init__(self,
                 n_spin_det: int,
                 n_determinants: int,
                 n_atoms: int):
        super(EnvelopePi, self).__init__()

        # dimension
        self.n_determinants = n_determinants
        self.n_spin_det = n_spin_det
        self.n_atoms = n_atoms

        pi_shape = (n_determinants, n_spin_det, n_atoms, 1)
        pi = orthogonal_init_tensor(pi_shape, 'pi')
        self.pi = nn.Parameter(pi, requires_grad=True)

    def forward(self, factor: torch.Tensor, exponential: torch.Tensor) -> torch.Tensor:
        exp = torch.einsum('njkim, kims -> njkis', exponential, self.pi).contiguous()
        output = factor * exp
        output = output.permute((0, 2, 3, 1, 4)).contiguous()
        return output


class Mixer(nn.Module):
    def __init__(self,
                 n_single_features: int,
                 n_pairwise_features: int,
                 n_electrons: int,
                 n_up: int,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(Mixer, self).__init__()
        '''
        The type/device is needed here because the conversion of subtensors is not captures in the same as as
        Parameter(s) at the top level. i.e. calling model.to('cuda') does not change the masks below
        '''

        n_down = n_electrons - n_up
        n_pairwise = n_electrons**2

        self.n_electrons = n_electrons
        self.n_up = float(n_up)
        self.n_down = float(n_down)

        tmp1 = torch.ones((1, n_up, n_single_features), dtype=torch.bool)
        tmp2 = torch.zeros((1, n_down, n_single_features), dtype=torch.bool)
        self.spin_up_mask = torch.cat((tmp1, tmp2), dim=1).type(dtype).to(device)
        self.spin_down_mask = (~torch.cat((tmp1, tmp2), dim=1)).type(dtype).to(device)

        self.pairwise_spin_up_mask, self.pairwise_spin_down_mask = \
            generate_pairwise_masks(n_electrons, n_pairwise, n_up, n_pairwise_features)
        self.pairwise_spin_up_mask = self.pairwise_spin_up_mask.type(dtype).to(device)
        self.pairwise_spin_down_mask = self.pairwise_spin_down_mask.type(dtype).to(device)

    def forward(self, single: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        # single (n_samples, n_electrons, n_single_features)
        # pairwise (n_samples, n_electrons, n_pairwise_features)
        # spin_up_mask = self.spin_up_mask.repeat((n_samples, 1, 1))
        # spin_down_mask = self.spin_down_mask.repeat((n_samples, 1, 1))

        # --- Single summations
        # up
        sum_spin_up = self.spin_up_mask * single
        sum_spin_up = sum_spin_up.sum(1, keepdim=True) / self.n_up
        sum_spin_up = sum_spin_up.repeat((1, self.n_electrons, 1))

        # down
        sum_spin_down = self.spin_down_mask * single
        sum_spin_down = sum_spin_down.sum(1, keepdim=True) / self.n_down
        sum_spin_down = sum_spin_down.repeat((1, self.n_electrons, 1))

        # --- Pairwise summations
        sum_pairwise = pairwise.unsqueeze(1).repeat((1, self.n_electrons, 1, 1))

        # up
        sum_pairwise_up = self.pairwise_spin_up_mask * sum_pairwise
        sum_pairwise_up = sum_pairwise_up.sum(2) / self.n_up

        # down
        sum_pairwise_down = self.pairwise_spin_down_mask * sum_pairwise
        sum_pairwise_down = sum_pairwise_down.sum(2) / self.n_down

        features = torch.cat((single, sum_spin_up, sum_spin_down, sum_pairwise_up, sum_pairwise_down), dim=2)
        return features


def compute_inputs(r_electrons: torch.Tensor,
                   n_samples: int,
                   ae_vectors: torch.Tensor,
                   n_atoms: int,
                   n_electrons: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # r_atoms: (n_atoms, 3)
    # r_electrons: (n_samples, n_electrons, 3)
    # ae_vectors: (n_samples, n_electrons, n_atoms, 3)
    ae_distances = torch.norm(ae_vectors, dim=-1, keepdim=True)
    single_inputs = torch.cat((ae_vectors, ae_distances), dim=-1)
    single_inputs = single_inputs.view((-1, n_electrons, 4 * n_atoms))

    re1 = r_electrons.unsqueeze(2)
    re2 = re1.permute((0, 2, 1, 3))
    ee_vectors = re1 - re2

    mask = torch.eye(n_electrons, dtype=torch.bool)
    mask = ~mask.unsqueeze(0).unsqueeze(3).repeat((n_samples, 1, 1, 3))

    ee_vectors = ee_vectors[mask]
    ee_vectors = ee_vectors.view((-1, int(n_electrons ** 2 - n_electrons), 3))
    ee_distances = torch.norm(ee_vectors, dim=-1, keepdim=True)

    pairwise_inputs = torch.cat((ee_vectors, ee_distances), dim=-1)

    return single_inputs, pairwise_inputs


def compute_ae_vectors(r_atoms: torch.Tensor, r_electrons: torch.Tensor) -> torch.Tensor:
    # ae_vectors (n_samples, n_electrons, n_atoms, 3)
    r_atoms = r_atoms.unsqueeze(1)
    r_electrons = r_electrons.unsqueeze(2)
    ae_vectors = r_electrons - r_atoms
    return ae_vectors


def generate_pairwise_masks(n_electrons: int,
                            n_pairwise: int,
                            n_up: int,
                            n_pairwise_features: int) -> Tuple[torch.Tensor, torch.Tensor]:
    eye_mask = ~np.eye(n_electrons, dtype=np.bool)
    ups = np.ones(n_electrons, dtype=np.bool)
    ups[n_up:] = False
    downs = ~ups

    spin_up_mask = []
    spin_down_mask = []
    mask = np.zeros((n_electrons, n_electrons), dtype=np.bool)

    for electron in range(n_electrons):
        e_mask_up = np.zeros((n_electrons,), dtype=np.bool)
        e_mask_down = np.zeros((n_electrons,), dtype=np.bool)

        mask_up = np.copy(mask)
        mask_up[electron, :] = ups
        mask_up = mask_up[eye_mask].reshape(-1)
        if electron < n_up:
            e_mask_up[electron] = True
        spin_up_mask.append(np.concatenate((mask_up, e_mask_up), axis=0))

        mask_down = np.copy(mask)
        mask_down[electron, :] = downs
        mask_down = mask_down[eye_mask].reshape(-1)
        if electron >= n_up:
            e_mask_down[electron] = True
        spin_down_mask.append(np.concatenate((mask_down, e_mask_down), axis=0))

    spin_up_mask = torch.tensor(spin_up_mask, dtype=torch.bool)
    # (n_samples, n_electrons, n_electrons, n_pairwise_features)
    spin_up_mask = spin_up_mask.view((1, n_electrons, n_pairwise, 1))
    spin_up_mask = spin_up_mask.repeat((1, 1, 1, n_pairwise_features))

    spin_down_mask = torch.tensor(spin_down_mask, dtype=torch.bool)
    spin_down_mask = spin_down_mask.view((1, n_electrons, n_pairwise, 1))
    spin_down_mask = spin_down_mask.repeat((1, 1, 1, n_pairwise_features))

    return spin_up_mask, spin_down_mask


def orthogonal_init_tensor(shape: Tuple, name: str) -> torch.Tensor:
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
                t = torch.empty(shape[2], 1)
                nn.init.normal_(t)
                idim.append(t)
            kdim.append(torch.stack(idim))
        tensor = torch.stack(kdim)

    elif name == 'w':  # kifs
        kdim = []
        for k in range(shape[0]):
            idim = []
            for i in range(shape[1]):
                t = torch.empty(shape[2], shape[3])
                b = torch.empty(1, shape[3])
                nn.init.orthogonal_(t)
                nn.init.normal_(b)
                t = torch.cat((t, b), dim=0)
                idim.append(t)
            kdim.append(torch.stack(idim))
        tensor = torch.stack(kdim)

    else:
        print('Layer not a valid choice for orthogonal initialisation.')

    return tensor


