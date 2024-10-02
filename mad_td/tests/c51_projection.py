from collections import namedtuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def test_c51_projection():
    vmin = -5.3
    vmax = 5.3
    num_bins = 151
    atoms = np.linspace(vmin, vmax, num_bins)

    mu = -2
    sigma = 3

    z = jax.nn.softmax(-((atoms - mu) ** 2) / (2 * sigma**2))

    r = 1
    gamma = 0.9

    plt.plot(atoms, z)
    print(z.dot(atoms))
    z_target = jax.nn.softmax(-((atoms - (mu + r)) ** 2) / (2 * (sigma * gamma) ** 2))
    plt.plot(atoms, z_target)
    print(z_target.dot(atoms))

    next_atoms = r + gamma * atoms
    # projection
    delta_z = atoms[1] - atoms[0]
    tz = next_atoms.clip(vmin, vmax)

    b = (tz - vmin) / delta_z
    l = np.floor(b).clip(0, len(atoms) - 1)
    u = np.ceil(b).clip(0, len(atoms) - 1)
    # (l == u).float() handles the case where bj is exactly an integer
    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
    d_m_l = (u + (l == u) - b) * z
    d_m_u = (b - l) * z
    target_pmfs = np.zeros_like(z)
    for i in range(target_pmfs.shape[0]):
        target_pmfs[int(l[i])] += d_m_l[i]
        target_pmfs[int(u[i])] += d_m_u[i]
    plt.plot(atoms, target_pmfs)
    print(target_pmfs.dot(atoms))

    HP = namedtuple("HP", ["vmin", "vmax", "gamma", "num_quantile"])
    hyperparams = HP(vmin, vmax, gamma, num_bins)

    rew_jax = jnp.array([[r]])  # (1,1)
    val_jax = jnp.array(z)  # (151,)
    jax_tar = jax_c51(hyperparams, rew_jax, val_jax)
    print(jax_tar)
    plt.plot(atoms, jax_tar)
    print(jax_tar.dot(atoms), np.sum(z), jnp.sum(jax_tar))
    plt.show()


def project_distribution(atoms, v_min, v_max, reward, gamma, values):
    import jax
    import jax.numpy as jnp

    n_atoms = atoms.shape[0]
    next_atoms = reward + gamma * atoms
    delta_z = atoms[1] - atoms[0]

    b = (next_atoms - v_min) / delta_z
    l = jnp.clip(jnp.floor(b), a_min=0, a_max=n_atoms - 1)
    u = jnp.clip(jnp.ceil(b), a_min=0, a_max=n_atoms - 1)
    # (l == u).astype(jnp.float) handles the case where bj is exactly an integer
    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
    d_m_l = (u + (l == u).astype(jnp.float32) - b) * values
    d_m_u = (b - l) * values
    td_target = jnp.zeros_like(values)

    def project_to_bins(i, val):
        val = val.at[l[i].astype(jnp.int32)].add(d_m_l[i])
        val = val.at[u[i].astype(jnp.int32)].add(d_m_u[i])
        return val

    td_target = jax.lax.fori_loop(0, td_target.shape[0], project_to_bins, td_target)
    return td_target


def jax_c51(hyperparams, reward, value):
    import jax
    import jax.numpy as jnp

    # projection
    atoms = jnp.linspace(hyperparams.vmin, hyperparams.vmax, hyperparams.num_quantile)
    n_atoms = atoms.shape[0]
    next_atoms = reward[0] + hyperparams.gamma * atoms
    delta_z = atoms[1] - atoms[0]

    b = (next_atoms - hyperparams.vmin) / delta_z
    l = jnp.clip(jnp.floor(b), a_min=0, a_max=n_atoms - 1)
    u = jnp.clip(jnp.ceil(b), a_min=0, a_max=n_atoms - 1)
    # (l == u).astype(jnp.float) handles the case where bj is exactly an integer
    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
    d_m_l = (u + (l == u).astype(jnp.float32) - b) * value
    d_m_u = (b - l) * value
    td_target = jnp.zeros_like(value)

    def project_to_bins(i, val):
        val = val.at[l[i].astype(jnp.int32)].add(d_m_l[i])
        val = val.at[u[i].astype(jnp.int32)].add(d_m_u[i])
        return val

    td_target = jax.lax.fori_loop(0, td_target.shape[0], project_to_bins, td_target)
    return td_target
