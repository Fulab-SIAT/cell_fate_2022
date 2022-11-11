# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import pandas as pd  # Or any other
# […]

# Own modules
import numpy as np
import matplotlib.pyplot as plt
import sciplot as splt
from scipy.integrate import odeint
from joblib import Parallel, delayed
from tqdm import tqdm

splt.whitegrid()


class ToggleBasic:
    def __init__(self):
        self.k_t = 18
        self.k_l = 26
        self.n_t = 2.0
        self.n_l = 3.6
        self.gr = 1.
        self.tau_p_l = 0.06  # 34.733 / 794.8
        self.tau_p_t = 1 / 650
        self.alphal = 5 * 670 * self.gr / (1 + (self.gr / 0.8) ** 4.5)
        self.alphat = 10 * 670 * self.gr / (1 + (self.gr / 0.8) ** 4.5)

        self.atc_conc = 0.
        self.iptg_conc = 0.
        self.k_atc = 1
        self.k_iptg = 1.
        self.m = 1.
        self.n = 1.

        self.sst_laci_conc = None
        self.sst_tetr_conc = None
        self.bistable

    def change_gr(self, gr):
        self.gr = gr
        self.alphal = 5 * 670 * self.gr / (1 + (self.gr / 0.8) ** 4.5)
        self.alphat = 10 * 670 * self.gr / (1 + (self.gr / 0.8) ** 4.5)

    def h_l(self, laci):
        return self.tau_p_l + (1. - self.tau_p_l) / (1. + (laci / self.k_l) ** self.n_l)

    def h_t(self, tetr):
        return self.tau_p_t + (1. - self.tau_p_t) / (1. + (tetr / self.k_t) ** self.n_t)

    def null_cline_tetr(self, laci_tot):
        laci = laci_tot * (1. + self.iptg_conc / self.k_iptg * laci_tot) ** -self.n
        return self.alphal * self.h_l(laci) / self.gr

    def null_cline_laci(self, tetr_tot):
        tetr = tetr_tot * (1. + self.atc_conc / self.k_atc * tetr_tot) ** -self.m
        return self.alphat * self.h_t(tetr) / self.gr

    def fild_flow(self, laci_tetr):
        laci_tot, tetr_tot = laci_tetr
        tetr = tetr_tot * (1. + self.atc_conc / self.k_atc * tetr_tot) ** -self.m
        laci = laci_tot * (1. + self.iptg_conc / self.k_iptg * laci_tot) ** -self.n
        dev_laci = self.alphat * self.h_t(tetr) - self.gr * laci_tot
        dev_tetr = self.alphal * self.h_l(laci) - self.gr * tetr_tot
        return [dev_laci, dev_tetr]

    def dev_laci(self, laci):
        free_laci = laci * (1. + self.iptg_conc / self.k_iptg * laci) ** -self.n
        sst_tetr = self.null_cline_tetr(free_laci)
        return self.alphat * self.h_t(sst_tetr) - self.gr * laci

    def potential_laci(self, U, laci):
        return - self.dev_laci(laci)

    def calu_laci_potential(self, laci_conc_list):
        laci_pot = odeint(self.potential_laci, 0, laci_conc_list)
        return laci_pot.flatten()

    def solve_sst(self, laci_conc_list):
        sign_dev_laci = np.sign(self.dev_laci(laci_conc_list))
        root = np.diff(sign_dev_laci)
        sst_index = np.nonzero(root)[0]
        self.sst_laci_conc = laci_conc_list[sst_index]
        self.sst_tetr_conc = self.null_cline_tetr(self.sst_laci_conc)
        self.sst_state = root[sst_index]
        if len(sst_index) == 3:
            self.bistable = True
        else:
            self.bistable = False


    def quasipotential(self, t_end, laci_tetr, num=1000):
        def dev_potential(pars, t):
            dev_laci_tetr = self.fild_flow(pars[:2])
            dp = np.sqrt(np.sum(np.array(dev_laci_tetr) ** 2))
            return [dev_laci_tetr[0],
                    dev_laci_tetr[1],
                    dp]

        # p0 = np.sqrt(np.sum(np.array(self.fild_flow(laci_tetr)) ** 2))
        pars0 = laci_tetr + [0.]
        t_list = np.linspace(0, t_end, num=num)
        laci_tetr_p_t = odeint(dev_potential, pars0, t_list)
        return laci_tetr_p_t


def titrate_gr(gr=1.0, laci_range=np.linspace(0, 20), tetr_range=np.linspace(0, 20), **kwargs):
    toggle = ToggleBasic()
    toggle.change_gr(gr)

    if 'fdcg_alphat' in kwargs:
        toggle.tau_p_t = 1. / kwargs['fdcg_alphat']
    if 'fdcg_alphal' in kwargs:
        toggle.tau_p_l = 1. / kwargs['fdcg_alphal']
    if 'alpha_trc' in kwargs:
        toggle.alphal = kwargs['alpha_trc']
    if 'alpha_ltet' in kwargs:
        toggle.alphat = kwargs['alpha_ltet']

    reslts = [toggle.null_cline_laci(tetr_range), toggle.null_cline_tetr(laci_range)]
    return reslts


def cal_quasipotential(t_end, laci_tetr, **kwargs):
    tg = ToggleBasic()
    if 'growth_rate' in kwargs:
        tg.change_gr(kwargs['growth_rate'])
    resluts = tg.quasipotential(t_end, laci_tetr)
    return resluts[-1:, 2]


# %% Test quasi potential landscape
# laci_range = np.linspace(0, 10000, num=100)
laci_range = np.logspace(0, 4, num=500)

tetr_range = np.logspace(1, 4, num=500)
init_laci, init_tetr = np.meshgrid(laci_range, tetr_range)
init_laci = init_laci.flatten()
init_tetr = init_tetr.flatten()

gr_list = [2.3, 1.65, 1.2, 1.0, 0.6, 0.4, 0.2]
grid_p_list = []
for gr in gr_list:
    quasi_p = Parallel(n_jobs=64)(delayed(cal_quasipotential)(100, [init_laci[i], init_tetr[i]], gr=gr)
                                  for i in tqdm(range(len(init_laci))))
    final_p = np.array(quasi_p)
    grid_p = final_p.reshape((len(laci_range), len(tetr_range)))
    grid_p = np.transpose(grid_p)
    grid_p_list.append(grid_p)

# %%
fig1, ax1 = plt.subplots(1, 6, figsize=(12 * 6, 10))
for i, ax in enumerate(ax1):
    pos = ax.imshow(grid_p_list[i], interpolation='nearest', cmap='coolwarm', vmin=0, vmax=10000, origin='lower')
    ct = ax.contour(grid_p_list[i], levels=[100, 500, 1200], origin='lower', linewidths=3, cmap='Set3')
    ax.clabel(ct, inline=True, fontsize=15, fmt='%.f')
    cb = fig1.colorbar(pos, ax=ax)
    # cb_contor = fig1.colorbar(ct, ax=ax, orientation='horizontal')
    # cb_contor.ax.set_position()
    ax.grid(False)
    ticklocy = lambda x: 500 / (np.log2(40000) - np.log2(1)) * (x - np.log2(1))
    # ax.xaxis.set_ticks(np.linspace(0, 500, num=6))
    ypow = np.linspace(2, 14, num=7)
    ax.yaxis.set_ticklabels(['$2^{%s}$' % str(int(i)) for i in ypow])
    ax.yaxis.set_ticks([ticklocy(i) for i in ypow])
    ticklocx = lambda x: 500 / (np.log2(40000) - np.log2(10)) * (x - np.log2(10))
    xpow = np.linspace(4, 14, num=6)
    ax.xaxis.set_ticklabels(['$2^{%s}$' % str(int(i)) % i for i in xpow])
    ax.xaxis.set_ticks([ticklocx(i) for i in xpow])
    cb.set_label("$P$")
    ax.set_ylabel('LacI conc. (a.u.)')
    ax.set_xlabel('TetR conc. (a.u.)')

fig1.show()
fig1.savefig(r'./data/quasi_potential_along_gr.png')
# %%
"""
This part shows a symmetric toggle and its null-clines
"""
laci_range = np.linspace(0, 10000, num=10000)
tetr_range = np.linspace(0, 10000, num=10000)

toggle = ToggleBasic()
tetr_nucl = toggle.null_cline_tetr(laci_range)
laci_nucl = toggle.null_cline_laci(tetr_range)

gr_list = [2.3, 0.2]
resluts = [titrate_gr(gr, laci_range, tetr_range) for gr in gr_list]
lt = ['-', '--', ':']

fig1, ax1 = plt.subplots()
for i, reslut in enumerate(resluts):
    ax1.plot(reslut[0], tetr_range, lt[i] + 'g', label=f'$\lambda = {gr_list[i]}$')
    ax1.plot(laci_range, reslut[1], lt[i] + 'r')
ax1.set_xlabel('LacI conc.')
ax1.set_ylabel('TetR conc.')
ax1.set_xlim(1, 10000)
ax1.set_ylim(1, 10000)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(False)
ax1.legend()
fig1.show()

# %%
laci_range = np.linspace(0, 1000, num=1000)
tetr_range = np.linspace(0, 1000, num=1000)

gr_list = [1.2, 0.5, 0.2]
resluts = [titrate_gr(gr, laci_range, tetr_range, pt=30) for gr in gr_list]
lt = ['-', '--', ':']

fig2, ax2 = plt.subplots()
for i, reslut in enumerate(resluts):
    ax2.plot(reslut[0], tetr_range, lt[i] + 'g', label=f'$\lambda = {gr_list[i]}$')
    ax2.plot(laci_range, reslut[1], lt[i] + 'r')
ax2.set_xlabel('LacI conc.')
ax2.set_ylabel('TetR conc.')
ax2.set_xlim(10, 5000)
ax2.set_ylim(0, 5000)
ax2.grid(False)
ax2.legend()
fig2.show()

# %% titrate pl
laci_range = np.linspace(0, 100, num=1000)
tetr_range = np.linspace(0, 100, num=1000)

pl_list = [15, 25, 45]
resluts = [titrate_gr(laci_range=laci_range, tetr_range=tetr_range, pl=pl) for pl in pl_list]
lt = ['-', '--', ':']

fig3, ax3 = plt.subplots()
for i, reslut in enumerate(resluts):
    ax3.plot(reslut[0], tetr_range, lt[i] + 'g')
    ax3.plot(laci_range, reslut[1], lt[i] + 'r', label=f'$p_L = {pl_list[i]}$')
ax3.set_xlabel('LacI conc.')
ax3.set_ylabel('TetR conc.')
ax3.set_xlim(0, 50)
ax3.set_ylim(0, 50)
ax3.grid(False)
ax3.legend()
ax3.set_title('Titrate $p_L$ strength')
fig3.show()

# %% titrate pt
laci_range = np.linspace(0, 100, num=1000)
tetr_range = np.linspace(0, 100, num=1000)

pt_list = [15, 25, 45]
resluts = [titrate_gr(laci_range=laci_range, tetr_range=tetr_range, pt=pt) for pt in pt_list]
lt = ['-', '--', ':']

fig4, ax4 = plt.subplots()
for i, reslut in enumerate(resluts):
    ax4.plot(reslut[0], tetr_range, lt[i] + 'g')
    ax4.plot(laci_range, reslut[1], lt[i] + 'r', label=f'$p_T = {pt_list[i]}$')
ax4.set_xlabel('LacI conc.')
ax4.set_ylabel('TetR conc.')
ax4.set_xlim(0, 50)
ax4.set_ylim(0, 40)
ax4.grid(False)

ax4.set_title('Titrate $p_t$ strength')
ax4.legend()
fig4.show()

# %% titrate fold change of p_l
laci_range = np.linspace(0, 100, num=1000)
tetr_range = np.linspace(0, 100, num=1000)

fc_pl_list = [1000, 50, 10]
resluts = [titrate_gr(laci_range=laci_range, tetr_range=tetr_range, fdcg_pl=fc_pl) for fc_pl in fc_pl_list]
lt = ['-', '--', ':']

fig4, ax4 = plt.subplots()
for i, reslut in enumerate(resluts):
    ax4.plot(reslut[0], tetr_range, lt[i] + 'g')
    ax4.plot(laci_range, reslut[1], lt[i] + 'r', label=f'$1/ \\tau = {fc_pl_list[i]}$')
ax4.set_xlabel('LacI conc.')
ax4.set_ylabel('TetR conc.')
ax4.set_xlim(0, 50)
ax4.set_ylim(0, 40)
ax4.set_title('Titrate fold change of $p_L$')
ax4.grid(False)
ax4.legend()
fig4.show()

# %% titrate fold change of p_t
laci_range = np.linspace(0, 100, num=1000)
tetr_range = np.linspace(0, 100, num=1000)

fc_pt_list = [1000, 50, 10]
resluts = [titrate_gr(laci_range=laci_range, tetr_range=tetr_range, fdcg_pt=fc_pt) for fc_pt in fc_pt_list]
lt = ['-', '--', ':']

fig4, ax4 = plt.subplots()
for i, reslut in enumerate(resluts):
    ax4.plot(reslut[0], tetr_range, lt[i] + 'g')
    ax4.plot(laci_range, reslut[1], lt[i] + 'r', label=f'$1/ \\tau = {fc_pt_list[i]}$')
ax4.set_xlabel('LacI conc.')
ax4.set_ylabel('TetR conc.')
ax4.set_xlim(0, 50)
ax4.set_ylim(0, 40)
ax4.set_title('Titrate fold change of $p_T$')
ax4.grid(False)
ax4.legend()
fig4.show()

# %%
laci_range = np.linspace(0, 100, num=1000)
tetr_range = np.linspace(0, 100, num=1000)

gr_list = [1.2, 0.5, 0.2]
resluts = [titrate_gr(gr, laci_range, tetr_range, pt=45, fdcg_pl=10) for gr in gr_list]
lt = ['-', '--', ':']

fig2, ax2 = plt.subplots()
for i, reslut in enumerate(resluts):
    ax2.plot(reslut[0], tetr_range, lt[i] + 'g', label=f'$\lambda = {gr_list[i]}$')
    ax2.plot(laci_range, reslut[1], lt[i] + 'r')
ax2.set_xlabel('LacI conc.')
ax2.set_ylabel('TetR conc.')
ax2.set_xlim(0, 400)
ax2.set_ylim(0, 400)
ax2.grid(False)
ax2.legend()
fig2.show()
