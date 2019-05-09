from itertools import cycle

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from .physics import local_energy
from .sampling import samples_from
from .utils import NULL_DEBUG, state_dict_copy


def loss_local_energy(Es_loc, weights, E_ref=None, p=1):
    ws, w_mean = (weights, weights.mean()) if weights is not None else (1.0, 1.0)
    E0 = E_ref if E_ref is not None else (Es_loc * ws).mean() / w_mean
    return (ws * (Es_loc - E0).abs() ** p).mean() / w_mean


def fit_wfnet_multi(wfnet, loss_funcs, opts, gen_factory, gen_kwargs, writers):
    for loss_func, opt, kwargs, writer in zip(loss_funcs, opts, gen_kwargs, writers):
        with writer:
            fit_wfnet(wfnet, loss_func, opt, gen_factory(**kwargs), writer=writer)


def fit_wfnet(
    wfnet,
    loss_func,
    opt,
    sample_gen,
    correlated_sampling=True,
    clip_grad=None,
    writer=None,
    start=0,
    debug=NULL_DEBUG,
    scheduler=None,
    epoch_size=100,
):
    for step, (rs, psi0s) in enumerate(sample_gen, start=start):
        d = debug[step]
        d['psi0s'], d['rs'] = psi0s, rs
        Es_loc, psis = d['Es_loc'], d['psis'] = local_energy(
            rs, wfnet, create_graph=True
        )
        valid = ~(torch.isnan(Es_loc) | torch.isinf(Es_loc))
        Es_loc = Es_loc[valid]
        if correlated_sampling:
            weights = (psis ** 2 / psi0s ** 2)[valid]
        else:
            weights = None
        loss = loss_func(Es_loc, weights)
        if writer:
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('E_loc/mean', Es_loc.mean(), step)
            writer.add_scalar('E_loc/var', Es_loc.var(), step)
            for label, value in wfnet.tracked_parameters():
                writer.add_scalar(f'param/{label}', value, step)
        loss.backward()
        if clip_grad:
            clip_grad_norm_(wfnet.parameters(), clip_grad)
        opt.step()
        opt.zero_grad()
        d['state_dict'] = state_dict_copy(wfnet)
        if scheduler and (step + 1) % epoch_size == 0:
            scheduler.step()


def wfnet_fit_driver(
    sampler,
    *,
    samplings,
    n_epochs,
    n_sampling_steps,
    batch_size=10_000,
    n_discard=50,
    range_sampling=range,
    range_training=range,
):
    for _ in samplings:
        rs, psis, _ = samples_from(
            sampler, range_sampling(n_sampling_steps), n_discard=n_discard
        )
        samples_ds = TensorDataset(rs.flatten(end_dim=1), psis.flatten(end_dim=1))
        rs_dl = DataLoader(samples_ds, batch_size=batch_size, shuffle=True)
        n_steps = n_epochs * len(rs_dl)
        for _, (rs, psis) in zip(range_training(n_steps), cycle(rs_dl)):
            yield rs, psis


def wfnet_fit_driver_simple(sampler, *, samplings, n_sampling_steps):
    for _ in samplings:
        rs, psis, _ = samples_from(sampler, range(n_sampling_steps))
        samples_ds = TensorDataset(rs.flatten(end_dim=1), psis.flatten(end_dim=1))
        yield from DataLoader(samples_ds, batch_size=len(samples_ds), shuffle=True)
