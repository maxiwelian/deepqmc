import torch
from torch.distributions import Normal
import numpy as np
from torch import nn
from typing import Tuple
from tqdm.auto import trange

from deepqmc import Molecule
from deepqmc.wf import WaveFunction
from deepqmc.wf.paulinet.molorb import MolecularOrbital
from deepqmc.wf.paulinet.gto import GTOBasis
from deepqmc.wf.paulinet.pyscfext import pyscf_from_mol
from deepqmc.physics import pairwise_diffs, local_energy


import torch
from deepqmc import Molecule
from deepqmc.wf.paulinet.pyscfext import eval_ao_normed, pyscf_from_mol

# mol = Molecule.from_name('H2O')
# mf, _ = pyscf_from_mol(mol, '6-31g')
# rs = torch.randn(100, 10, 3).double()
# mo = (eval_ao_normed(mf.mol, rs.flatten(end_dim=1).numpy()).reshape(100, 10, -1) @ mf.mo_coeff[:, :5])


class Pretrainer(nn.Module):
    r""" Implements the FermiNet wave function Ansatz pretraining based on [pfau2020ab]

    Provides tools for pretraining the Ansatz.

    .. math:

    Usage:
        wf = FermiNet(mol, n_layers, nf_hidden_single, nf_hidden_pairwise, n_determinants).cuda()
        pretrainer = Pretrainer(mol).cuda()
        pretrainer.pretrain(wf)

    Args:
        mol (:class:`~deepqmc.Molecule`): molecule whose wave function is represented
        basis (str): basis for the molecular orbitals

    """

    def __init__(self,
                 mol,
                 basis: str = '6-311g',
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(Pretrainer, self).__init__()
        self.device = device
        self.dtype = dtype

        self.atom_positions = [x.cpu().numpy() for x in mol.coords.split(1, dim=0)]
        self.ne_atoms = [int(i) for i in mol.charges]

        self.mol = mol
        self.n_elec = int(mol.charges)
        self.n_up = (self.n_elec + mol.spin) // 2
        self.n_down = (self.n_elec - mol.spin) // 2
        self.n_atoms = len(self.mol)
        self.n_orbitals = max(self.n_up, self.n_down)  # if n_orbital is none return max of up or down
        # cas and workdir set to None

        self.mf, _ = pyscf_from_mol(mol, basis)

    def compute_orbital_probability(self, samples: torch.Tensor) -> torch.Tensor:
        up_dets, down_dets = self.hf_orbitals(samples)

        spin_ups = up_dets ** 2
        spin_downs = down_dets ** 2

        p_up = torch.diagonal(spin_ups, dim1=-2, dim2=-1).prod(-1)
        p_down = torch.diagonal(spin_downs, dim1=-2, dim2=-1).prod(-1)

        probabilities = p_up * p_down

        return probabilities.detach()

    def hf_orbitals(self, samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_samples = samples.shape[0]
        # mol = Molecule.from_name('H2O')
        # mf, _ = pyscf_from_mol(mol, '6-31g')
        # rs = torch.randn(100, 10, 3).double()
        # mo = (eval_ao_normed(mf.mol, rs.flatten(end_dim=1).numpy()).reshape(100, 10, -1) @ mf.mo_coeff[:, :5])
        #
        samples = samples.flatten(end_dim=1).cpu().numpy()
        determinants = (eval_ao_normed(self.mf.mol, samples).reshape(n_samples, self.n_elec, -1)
                        @ self.mf.mo_coeff[:, :self.n_orbitals])
        determinants = torch.from_numpy(determinants).to(device=self.device, dtype=self.dtype)
        up_dets, down_dets = determinants.split([self.n_up, self.n_down], dim=1)
        up_dets, down_dets = up_dets[:, :, :up_dets.shape[1]], down_dets[:, :, :down_dets.shape[1]]

        return up_dets, down_dets

    def pretrain(self,
                 wf: WaveFunction,
                 n_samples: int = 1024,
                 n_steps: int = 1000,
                 lr: float = 1e-4):

        sampler = MetropolisHastingsPretrain()
        opt = torch.optim.Adam(list(wf.parameters())[:-1], lr=lr)
        steps = trange(
            0,  # init_step = 0
            n_steps,
            initial=0,
            total=n_steps,
            desc='pretraining',
            disable=None,
        )

        samples = initialize_samples(self.ne_atoms, self.atom_positions, n_samples).to(device=self.device, dtype=self.dtype)

        for step in steps:
            Es_loc, _, _ = local_energy(
                samples,
                wf.sample(False),
                create_graph=False,
                keep_graph=False,
            )

            samples = sampler(wf, self, samples)

            up_dets, down_dets = self.hf_orbitals(samples)
            up_dets = tile_labels(up_dets, wf.n_determinants)
            down_dets = tile_labels(down_dets, wf.n_determinants)

            wf.pretraining = True
            model_up_dets, model_down_dets = wf(samples)
            wf.pretraining = False

            loss = mse_error(up_dets, model_up_dets)
            loss += mse_error(down_dets, model_down_dets)
            opt.zero_grad()
            loss.backward()  # in order for hook to work must call backward
            opt.step()

            steps.set_postfix(E=f'{Es_loc.mean():.6f}')

            # print('iteration: ', step, ' energy: ', Es_loc.mean())


class Pretrainer_dep(nn.Module):
    r""" Implements the FermiNet wave function Ansatz pretraining based on [pfau2020ab]

    Provides tools for pretraining the Ansatz.

    .. math:

    Usage:
        wf = FermiNet(mol, n_layers, nf_hidden_single, nf_hidden_pairwise, n_determinants).cuda()
        pretrainer = Pretrainer(mol).cuda()
        pretrainer.pretrain(wf)

    Args:
        mol (:class:`~deepqmc.Molecule`): molecule whose wave function is represented
        basis (str): basis for the molecular orbitals

    """

    def __init__(self,
                 mol,
                 basis: str = '6-311g',
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(Pretrainer_dep, self).__init__()
        self.device = device
        self.dtype = dtype

        self.atom_positions = [x.cpu().numpy() for x in mol.coords.split(1, dim=0)]
        self.ne_atoms = [int(i) for i in mol.charges]

        self.mol = mol
        self.n_elec = int(mol.charges)
        self.n_up = (self.n_elec + mol.spin) // 2
        self.n_down = (self.n_elec - mol.spin) // 2
        self.n_atoms = len(self.mol)
        self.n_orbitals = max(self.n_up, self.n_down)  # if n_orbital is none return max of up or down
        # cas and workdir set to None
        mf, mc = pyscf_from_mol(mol, basis, None, None)
        basis = GTOBasis.from_pyscf(mf.mol)  # basis from molecule from name
        mol = Molecule(
            mf.mol.atom_coords(),
            mf.mol.atom_charges(),
            mf.mol.charge,
            mf.mol.spin,
        )
        self.mo = MolecularOrbital(  # create the molecular orbital
            mol,
            basis,
            self.n_orbitals,
            cusp_correction=False)
        self.mo.init_from_pyscf(mf, freeze_mos=True)
        self.coords = mol.coords.unsqueeze(0).to(device=device, dtype=dtype)

    def compute_orbital_probability(self, samples: torch.Tensor) -> torch.Tensor:
        up_dets, down_dets = self.hf_orbitals(samples)

        spin_ups = up_dets ** 2
        spin_downs = down_dets ** 2

        p_up = torch.diagonal(spin_ups, dim1=-2, dim2=-1).prod(-1)
        p_down = torch.diagonal(spin_downs, dim1=-2, dim2=-1).prod(-1)

        probabilities = p_up * p_down

        return probabilities.detach()

    def hf_orbitals(self, samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        samples_hf = samples.view(-1, 1, 3).repeat(1, self.n_atoms, 1)
        diffs_nuc = pairwise_diffs(torch.cat([self.coords, samples_hf]), self.coords).squeeze(1)
        determinants = self.mo(diffs_nuc).unsqueeze(1).view(-1, self.n_elec, self.n_orbitals)
        up_dets, down_dets = determinants.split([self.n_up, self.n_down], dim=1)
        up_dets, down_dets = up_dets[:, :, :up_dets.shape[1]], down_dets[:, :, :down_dets.shape[1]]
        return up_dets, down_dets

    def pretrain(self,
                 wf: WaveFunction,
                 n_samples: int = 1024,
                 n_steps: int = 1000):

        sampler = MetropolisHastingsPretrain()
        opt = torch.optim.Adam(list(wf.parameters())[:-3], lr=0.0001)
        steps = trange(
            0,  # init_step = 0
            n_steps,
            initial=0,
            total=n_steps,
            desc='pretraining',
            disable=None,
        )

        samples = initialize_samples(self.ne_atoms, self.atom_positions, n_samples).to(device=self.device, dtype=self.dtype)

        for step in steps:
            Es_loc, _, _ = local_energy(
                samples,
                wf.sample(False),
                create_graph=False,
                keep_graph=False,
            )

            samples = sampler(wf, self, samples)

            up_dets, down_dets = self.hf_orbitals(samples)
            up_dets = tile_labels(up_dets, wf.n_determinants)
            down_dets = tile_labels(down_dets, wf.n_determinants)

            wf.pretraining = True
            model_up_dets, model_down_dets = wf(samples)
            wf.pretraining = False

            loss = mse_error(up_dets, model_up_dets)
            loss += mse_error(down_dets, model_down_dets)
            wf.zero_grad()
            loss.backward()  # in order for hook to work must call backward
            opt.step()
            print('iteration: ', step, ' energy: ', Es_loc.mean())


def mse_error(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    return ((targets - outputs)**2).mean(0).sum()


def tile_labels(label: torch.Tensor, n_k: int) -> torch.Tensor:
    x = label.unsqueeze(dim=1).repeat((1, n_k, 1, 1))
    return x


class RandomWalker():
    r""" Creates normal sampler with std of sigma

    Used to suggest new updates to the positions of the walkers

    Usage:
        distr = RandomWalker(sigma.to(device=device, dtype=dtype))

    Args:
        sigma (float): step size of the walkers

    """
    def __init__(self, sigma):
        self.step_gaussian = Normal(0.0, sigma)

    def resample(self, prev_sample) -> torch.Tensor:
        return prev_sample + self.step_gaussian.sample(prev_sample.shape)


class Uniform(nn.Module):
    r""" Creates a uniform sampler between 0 and 1

    Used to determine whether moves accepted or rejected

    Usage:
        alpha_distr = Uniform(torch.tensor(0.).to(device=device, dtype=dtype), torch.tensor(1.).to(device=device, dtype=dtype))

    """
    def __init__(self, low=0, high=1):
        super(Uniform, self).__init__()
        self.low = torch.tensor(low) if type(low) != torch.Tensor else low
        self.high = torch.tensor(high) if type(high) != torch.Tensor else high

    def forward(self, batch_size: int = 1):
        return self.low + torch.rand(batch_size, device=self.low.device) * (self.high - self.low)

    def sample(self, batch_size: int = 1):
        return self(batch_size)


class ToProb(nn.Module):
    def forward(self, amps: torch.Tensor) -> torch.Tensor:
        return torch.exp(amps) ** 2


def initialize_samples(ne_atoms, atom_positions, n_samples):
    r""" Initialises samples for pretraining

        Usage:
            samples = initialize_samples(ne_atoms, atom_positions, n_samples).to(device=self.device, dtype=self.dtype)

        Args:
            ne_atoms (list int): number of electrons assigned to each nucleus
            atom_positions (list np.array): positions of the nuclei
            n_samples (int): number of walkers

        Returns:
            samples (np.array): walker positions (n_samples, n_elec, 3)

        """
    ups = []
    downs = []
    for ne_atom, atom_position in zip(ne_atoms, atom_positions):
        for e in range(ne_atom):
            if e % 2 == 0:  # fill up the orbitals alternating up down
                curr_sample_up = np.random.normal(loc=atom_position, scale=1., size=(n_samples, 1, 3))
                ups.append(curr_sample_up)
            else:
                curr_sample_down = np.random.normal(loc=atom_position, scale=1., size=(n_samples, 1, 3))
                downs.append(curr_sample_down)
    ups = np.concatenate(ups, axis=1)
    downs = np.concatenate(downs, axis=1)
    curr_sample = np.concatenate([ups, downs], axis=1)  # stack the ups first to be consistent with model
    return torch.from_numpy(curr_sample)


class MetropolisHastingsPretrain(nn.Module):
    r""" Implements MetropolisHastings sampling based on [pfau2020ab]

    Samples congigurations based on the amplitudes of both the Hartree Fock orbitals and the wave function Ansatz

    .. math:

    Usage:
        sampler = MetropolisHastingsPretrain()

    Args:
        sigma (float): step size for the walkers (std of the proposed moves)
        correlation_length (int): number of steps between sampling each update of the walker positions
        target_acceptance (float): the target acceptance of the steps

    Returns:
        curr_sample (torch.Tensor): walker configurations (n_samples, n_elec, 3)

    """
    def __init__(self,
                 sigma: float = 0.02,
                 correlation_length: int = 10,
                 target_acceptance: float = 0.5,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super(MetropolisHastingsPretrain, self).__init__()
        self.device = device
        self.dtype = dtype

        self.sigma = sigma
        self.correlation_length = correlation_length

        self.distr = RandomWalker(sigma)
        self.alpha_distr = Uniform(torch.tensor(0.).to(device=device, dtype=dtype), torch.tensor(1.).to(device=device, dtype=dtype))
        self.to_prob = ToProb()

        self.acceptance = 0.0
        self.target_acceptance = target_acceptance

        print('initialized pretraining sampler')

    def forward(self, model, pretrainer, curr_sample):
        n_samples = curr_sample.shape[0]

        # --- split the walkers and sample half from the hf_orbitals and half from the wave function
        sams = curr_sample.split([n_samples // 2, n_samples // 2])
        curr_sample_model, curr_sample_hf = sams[0].squeeze(), sams[1].squeeze()
        shape = curr_sample_model.shape

        curr_log_amp = model(curr_sample_model)[0]
        curr_prob_model = self.to_prob(curr_log_amp)
        curr_prob_hf = pretrainer.compute_orbital_probability(curr_sample_hf)

        acceptance_total_mod = 0.
        acceptance_total_hf = 0.
        for _ in range(self.correlation_length):
            # --- next sample
            new_sample_model = curr_sample_model + torch.normal(0.0, self.sigma, size=shape, device=self.device, dtype=self.dtype)
            new_log_amp = model(new_sample_model)[0]
            new_prob_model = self.to_prob(new_log_amp)

            new_sample_hf = curr_sample_hf + torch.normal(0.0, self.sigma, size=shape, device=self.device)
            new_prob_hf = pretrainer.compute_orbital_probability(new_sample_hf).to(self.device)

            # --- update sample
            alpha_model = new_prob_model / curr_prob_model
            alpha_hf = new_prob_hf / curr_prob_hf

            # --- generate masks
            mask_model = alpha_model > torch.rand(shape[0], device=self.device, dtype=self.dtype)
            mask_hf = alpha_hf > torch.rand(shape[0], device=self.device, dtype=self.dtype)

            curr_sample_model = torch.where(mask_model.unsqueeze(-1).unsqueeze(-1), new_sample_model, curr_sample_model)
            curr_prob_model = torch.where(mask_model, new_prob_model, curr_prob_model)

            curr_sample_hf = torch.where(mask_hf.unsqueeze(-1).unsqueeze(-1), new_sample_hf, curr_sample_hf)
            curr_prob_hf = torch.where(mask_hf, new_prob_hf, curr_prob_hf)

            acceptance_total_mod += mask_model.type(self.dtype).mean()
            acceptance_total_hf += mask_hf.type(self.dtype).mean()

        curr_sample = torch.cat([curr_sample_model, curr_sample_hf], dim=0)
        # --- randomly permute so some walkers in the next run sample from different distribution than in this run
        idxs = torch.randperm(len(curr_sample))
        curr_sample = curr_sample[idxs]
        return curr_sample

    def adjust_sampling_steps(self, acceptance):
        if acceptance < 0.5:
            self.sigma += 0.001
        else:
            self.sigma -= 0.001
