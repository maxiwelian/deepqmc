
from pyscf import gto

from typing import Tuple
import pickle as pk
import sys, os
import numpy as np
from torch.autograd import grad
import torch
from deepqmc.wf import WaveFunction

'''
useful resources
type hinting in pytorch: https://pytorch.org/docs/stable/jit_language_reference.html
'''

# class MolecularOrbital(nn.Module):

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def reader(path: str):
    mol = gto.Mole()
    with open(path, 'rb') as f:
        data = pk.load(f)
    mol.atom = data["mol"]
    mol.unit = "Bohr"
    mol.basis = data["basis"]
    mol.verbose = 4
    mol.spin = data["spin"]
    # mol.charge = 1
    mol.build()
    number_of_electrons = mol.tot_electrons()
    number_of_atoms = mol.natm
    ST = data["super_twist"]
    print('atom: ', mol.atom)
    # mol
    return ST, mol


class Pretrainer():
    def __init__(self,
                 n_pretrain_iterations: int,
                 n_determinants: int,
                 n_electrons: int,
                 n_spin_up: int,
                 n_spin_down: int,
                 pretrain_path: str):

        try:
            self.super_twist, self.mol = reader(pretrain_path)
            self.moT = torch.from_numpy(self.super_twist.T.astype(np.float32))
        except:
            print('pretrain data does not exist...')

        self.n_spin_up = n_spin_up
        self.n_spin_down = n_spin_down

        self.n_electrons = n_electrons
        self.n_determinants = n_determinants
        self.n_iterations = n_pretrain_iterations

    def compute_orbital_probability(self, samples: torch.Tensor) -> torch.Tensor:
        up_dets, down_dets = self.wave_function(samples)

        spin_ups = up_dets ** 2
        spin_downs = down_dets ** 2

        p_up = torch.diagonal(spin_ups, dim1=-2, dim2=-1).prod(-1)
        p_down = torch.diagonal(spin_downs, dim1=-2, dim2=-1).prod(-1)
        # p_up = spin_ups.prod(1).prod(1)
        # p_down = spin_downs.prod(1).prod(1)

        probabilities = p_up * p_down

        return probabilities

    def pyscf_call(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.cpu().numpy()
        ao_values = self.mol.eval_gto("GTOval_cart", samples)
        return torch.from_numpy(ao_values.astype(np.float32))

    def wave_function(self, coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coord = coord.view((-1, 3))

        number_spin_down = self.n_spin_down
        number_spin_up = self.n_electrons - number_spin_down

        ao_values = self.pyscf_call(coord)
        ao_values = ao_values.view((int(len(ao_values) / self.n_electrons), self.n_electrons, len(ao_values[0])))

        spin_up = torch.stack([(self.moT[orb_number, :] * ao_values[:, el_number, :]).sum(-1)
             for orb_number in range(number_spin_up) for el_number in
             range(number_spin_up)], dim=1).view((-1, number_spin_up, number_spin_up))

        spin_down = torch.stack([(self.moT[orb_number, :] * ao_values[:, el_number, :]).sum(-1)
                            for orb_number in range(number_spin_down) for el_number in
                            range(number_spin_up, self.n_electrons)], dim=1).view((-1, number_spin_down, number_spin_down))

        return spin_up, spin_down

    def compute_grads(self, model: WaveFunction, samples: torch.Tensor) -> list:
        up_dets, down_dets = self.wave_function(samples)
        up_dets = tile_labels(up_dets, self.n_determinants).to(model.device)
        down_dets = tile_labels(down_dets, self.n_determinants).to(model.device)
        model_up_dets, model_down_dets = model(samples)[-2:]
        loss = mse_error(up_dets, model_up_dets)
        loss += mse_error(down_dets, model_down_dets)
        model.zero_grad()
        loss.backward()  # in order for hook to work must call backward
        grads = [w.grad.data for w in list(model.parameters())[:-1]]
        return grads


def mse_error(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    return ((targets - outputs)**2).mean(0).sum()


def tile_labels(label: torch.Tensor, n_k: int) -> torch.Tensor:
    x = label.unsqueeze(dim=1).repeat((1, n_k, 1, 1))
    return x


