# -*- coding: utf-8 -*-

"""
 This module simulates wet experiments those can be performed in lab.
 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

import pandas as pd
import numpy as np  # Or any other
from toggle_dynamic import ToggleBasic, steady_state, cal_laci_poetntial, assign_vars, bistability_titrate_k
from joblib import Parallel, delayed, dump, load
from tqdm import tqdm
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import sciplot as splt
from typing import List

splt.whitegrid()

# plot the steady states along the growth rate, bifurcation plot
data_expression = pd.read_csv(r'./data/single_arm_lacI_TetR_expression_level.csv')
data_alpha = pd.read_csv(r'./data/lacI_TetR_expression_level_vs_gr.csv')
al_t_list, al_l_list = data_expression['LacI'].values, data_expression['TetR'].values
growth_rate_point = data_alpha[data_alpha['Type'] == 'LacI']['Growth_rate'].values
growth_rate_sim = np.linspace(0.15, 1.8, num=500)
laci_list_sim = np.linspace(0, 250, num=1000)
phase_portrait = pd.read_csv('./data/M5_L3_phase_diagram.csv')
green_ratio_exp = phase_portrait['LacI'] / (phase_portrait['TetR'] + phase_portrait['LacI'])
green_ratio_gr = phase_portrait['Growth_rate']
sim_index = slice(0, len(growth_rate_sim))
growth_rate = np.hstack([growth_rate_sim, green_ratio_gr])
# ret = []
#
# for i in range(len(growth_rate)):
#     ret.append(steady_state(growth_rate[i], k_l=16, k_t=3.8, tau_p_ltet=0.001))
#
# ret = Parallel(n_jobs=-1)(delayed(steady_state)(growth_rate[i], k_l=8, k_t=3.1, tau_p_ltet=0.001, tau_p_trc=0.05)
#                           for i in tqdm(range(len(growth_rate))))
# %%
# factor = 0.80
# kg = 1.
#
# toggle_pars = dict(k_t=3.552 * factor, k_l=3.39 * factor * kg, tau_p_ltet=0.001, tau_p_trc=0.015, n_l=1.4)

factor = 0.98
kg = 1
toggle_pars = dict(k_t=4.14 * factor, k_l=5.71 * factor * kg, tau_p_ltet=0.0016, tau_p_trc=0.015, n_l=1.4)
ret = Parallel(n_jobs=-1)(delayed(steady_state)(growth_rate[i], **toggle_pars)
                          for i in tqdm(range(len(growth_rate))))
# print(ret)
red_state_tetr = np.array([sst[0][0] for sst in ret])
red_state_laci = np.array([sst[1][0] for sst in ret])

green_state_tetr = np.array([sst[0][-1] for sst in ret])
green_state_laci = np.array([sst[1][-1] for sst in ret])

bistable_mask = np.array([sst[2] for sst in ret])
red_unstable = []
green_unstable = []
for ss in ret:
    try:
        red_usb = ss[0][1]
        green_usb = ss[1][1]
        red_unstable.append(red_usb)
        green_unstable.append(green_usb)
    except IndexError:
        red_unstable.append(np.nan)
        green_unstable.append(np.nan)
red_unstable = np.array(red_unstable)
green_unstable = np.array(green_unstable)

max_red = 1 / 12
max_green = 1 / 4
# normalize
red_state_tetr = red_state_tetr / max_red
red_state_laci = red_state_laci / max_green

green_state_tetr = green_state_tetr / max_red
green_state_laci = green_state_laci / max_green

red_unstable = red_unstable / max_red
green_unstable = green_unstable / max_green

exp_red = phase_portrait['TetR']
exp_green = phase_portrait['LacI']

exp_red = exp_red
exp_green = exp_green
exp_green_ratio = exp_green / (exp_green + exp_red)
exp_red_ratio = 1. - exp_green_ratio

markers = ['o', 'v', "^", "p", 's', '*', '+', 'x']
fig2, ax2 = plt.subplots(1, 1)
# plot modeling steady results of growth rates collected from exp.
for a, i in enumerate(range(len(green_state_tetr) - len(growth_rate_point), len(green_state_tetr))):
    ax2.scatter(green_state_tetr[i], green_state_laci[i], color='#43A047', marker=markers[a], s=500)
    ax2.scatter(red_state_tetr[i], red_state_laci[i], color='#EF5350', marker=markers[a], s=500)
for a, i in enumerate(range(len(green_state_tetr) - len(growth_rate_point), len(green_state_tetr))):
    ax2.scatter(red_unstable[i], green_unstable[i], color='#FFC107', alpha=0.2, marker=markers[a], s=500)

i_max = len(green_state_tetr) - len(growth_rate_point) - 1
ax2.plot(green_state_tetr[sim_index], green_state_laci[sim_index], color='#43A047', ls='--', alpha=0.3)
ax2.plot(red_state_tetr[sim_index], red_state_laci[sim_index], color='#EF5350', ls='--', alpha=0.3)
ax2.plot(red_unstable[sim_index], green_unstable[sim_index], color='#FFC107', ls='--', alpha=0.3)

ax2.scatter(exp_red, exp_green)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(False)
# ax2.set_xlim(.1, 1000)
# ax2.set_ylim(10, 1000)
ax2.set_xlabel('TetR Conc.')
ax2.set_ylabel('LacI Conc.')
fig2.show()

fig5, ax5 = plt.subplots(1, 1)
red_unstable_ratio = (red_unstable / (red_unstable + green_unstable))
red_state_tetr_ratio = (red_state_tetr / (red_state_laci + red_state_tetr))
green_state_tetr_ratio = (green_state_tetr / (green_state_laci + green_state_tetr))
ax5.scatter(growth_rate[sim_index], red_unstable_ratio[sim_index],
            color='y')
ax5.scatter(growth_rate[sim_index], red_state_tetr_ratio[sim_index],
            color='r')
ax5.scatter(growth_rate[sim_index], green_state_tetr_ratio[sim_index],
            color='g')

ax5.scatter(green_ratio_gr, exp_red_ratio, marker='o', facecolors='none', edgecolors='r', lw=4)
ax5.set_yscale('linear')
ax5.set_xlabel('$\lambda$')
ax5.set_ylabel('red ratio')
ax5.set_ylim(-0.2, 1.2)
ax5.grid(False)
fig5.show()
export_data = pd.DataFrame(data=dict(growth_rate=growth_rate[sim_index],
                                     red_unstable_ratio=red_unstable_ratio[sim_index],
                                     red_state_tetr_ratio=red_state_tetr_ratio[sim_index],
                                     green_state_tetr_ratio=green_state_tetr_ratio[sim_index]))
export_data.to_csv(r'./data/bifurcation_gr_vs_Rratio_1.csv')
# %%
# factor = 0.8
kg = 0.476
# toggle_pars = dict(k_t=3.552 * factor, k_l=3.39 * factor * kg, tau_p_ltet=0.001, tau_p_trc=0.015, n_l=1.4)
factor = 1.0
# kg = 1.
toggle_pars = dict(k_t=3.0 * factor, k_l=4.5 * factor * kg, tau_p_ltet=0.003, tau_p_trc=0.015, n_l=1.4)

ret = Parallel(n_jobs=-1)(delayed(steady_state)(growth_rate[i], **toggle_pars)
                          for i in tqdm(range(len(growth_rate))))

# print(ret)
red_state_tetr = np.array([sst[0][0] for sst in ret])
red_state_laci = np.array([sst[1][0] for sst in ret])

green_state_tetr = np.array([sst[0][-1] for sst in ret])
green_state_laci = np.array([sst[1][-1] for sst in ret])

bistable_mask = np.array([sst[2] for sst in ret])
red_unstable = []
green_unstable = []
for ss in ret:
    try:
        red_usb = ss[0][1]
        green_usb = ss[1][1]
        red_unstable.append(red_usb)
        green_unstable.append(green_usb)
    except IndexError:
        red_unstable.append(np.nan)
        green_unstable.append(np.nan)
red_unstable = np.array(red_unstable)
green_unstable = np.array(green_unstable)

max_red = 1 / 12
max_green = 1 / 4
# normalize
red_state_tetr = red_state_tetr / max_red
red_state_laci = red_state_laci / max_green

green_state_tetr = green_state_tetr / max_red
green_state_laci = green_state_laci / max_green

red_unstable = red_unstable / max_red
green_unstable = green_unstable / max_green

exp_red = phase_portrait['TetR']
exp_green = phase_portrait['LacI']

exp_red = exp_red
exp_green = exp_green
exp_green_ratio = exp_green / (exp_green + exp_red)
exp_red_ratio = 1. - exp_green_ratio

markers = ['o', 'v', "^", "p", 's', '*', '+', 'x']
fig2, ax2 = plt.subplots(1, 1)
# plot modeling steady results of growth rates collected from exp.
for a, i in enumerate(range(len(green_state_tetr) - len(growth_rate_point), len(green_state_tetr))):
    ax2.scatter(green_state_tetr[i], green_state_laci[i], color='#43A047', marker=markers[a], s=500)
    ax2.scatter(red_state_tetr[i], red_state_laci[i], color='#EF5350', marker=markers[a], s=500)
for a, i in enumerate(range(len(green_state_tetr) - len(growth_rate_point), len(green_state_tetr))):
    ax2.scatter(red_unstable[i], green_unstable[i], color='#FFC107', alpha=0.2, marker=markers[a], s=500)

i_max = len(green_state_tetr) - len(growth_rate_point) - 1
ax2.plot(green_state_tetr[sim_index], green_state_laci[sim_index], color='#43A047', ls='--', alpha=0.3)
ax2.plot(red_state_tetr[sim_index], red_state_laci[sim_index], color='#EF5350', ls='--', alpha=0.3)
ax2.plot(red_unstable[sim_index], green_unstable[sim_index], color='#FFC107', ls='--', alpha=0.3)

ax2.scatter(exp_red, exp_green)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(False)
# ax2.set_xlim(.1, 1000)
# ax2.set_ylim(10, 1000)
ax2.set_xlabel('TetR Conc.')
ax2.set_ylabel('LacI Conc.')
fig2.show()

fig5, ax5 = plt.subplots(1, 1)
red_unstable_ratio = (red_unstable / (red_unstable + green_unstable))
red_state_tetr_ratio = (red_state_tetr / (red_state_laci + red_state_tetr))
green_state_tetr_ratio = (green_state_tetr / (green_state_laci + green_state_tetr))
ax5.scatter(growth_rate[sim_index], red_unstable_ratio[sim_index],
            color='y')
ax5.scatter(growth_rate[sim_index], red_state_tetr_ratio[sim_index],
            color='r')
ax5.scatter(growth_rate[sim_index], green_state_tetr_ratio[sim_index],
            color='g')

ax5.scatter(green_ratio_gr, exp_red_ratio, marker='o', facecolors='none', edgecolors='r', lw=4)
ax5.set_yscale('linear')
ax5.set_xlabel('$\lambda$')
ax5.set_ylabel('red ratio')
ax5.set_ylim(-0.2, 1.2)
ax5.grid(False)
fig5.show()
export_data = pd.DataFrame(data=dict(growth_rate=growth_rate[sim_index],
                                     red_unstable_ratio=red_unstable_ratio[sim_index],
                                     red_state_tetr_ratio=red_state_tetr_ratio[sim_index],
                                     green_state_tetr_ratio=green_state_tetr_ratio[sim_index]))
export_data.to_csv(r'./data/bifurcation_gr_vs_Rratio_2.csv')

# %%
# factor = 0.8
kg = 0.2
# toggle_pars = dict(k_t=3.552 * factor, k_l=3.39 * factor * kg, tau_p_ltet=0.001, tau_p_trc=0.015, n_l=1.4)

factor = 1.0
# kg = 1.
toggle_pars = dict(k_t=3.0 * factor, k_l=4.5 * factor * kg, tau_p_ltet=0.003, tau_p_trc=0.015, n_l=1.4)

ret = Parallel(n_jobs=-1)(delayed(steady_state)(growth_rate[i], **toggle_pars)
                          for i in tqdm(range(len(growth_rate))))

# print(ret)
red_state_tetr = np.array([sst[0][0] for sst in ret])
red_state_laci = np.array([sst[1][0] for sst in ret])

green_state_tetr = np.array([sst[0][-1] for sst in ret])
green_state_laci = np.array([sst[1][-1] for sst in ret])

bistable_mask = np.array([sst[2] for sst in ret])
red_unstable = []
green_unstable = []
for ss in ret:
    try:
        red_usb = ss[0][1]
        green_usb = ss[1][1]
        red_unstable.append(red_usb)
        green_unstable.append(green_usb)
    except IndexError:
        red_unstable.append(np.nan)
        green_unstable.append(np.nan)
red_unstable = np.array(red_unstable)
green_unstable = np.array(green_unstable)

max_red = 1 / 12
max_green = 1 / 4
# normalize
red_state_tetr = red_state_tetr / max_red
red_state_laci = red_state_laci / max_green

green_state_tetr = green_state_tetr / max_red
green_state_laci = green_state_laci / max_green

red_unstable = red_unstable / max_red
green_unstable = green_unstable / max_green

exp_red = phase_portrait['TetR']
exp_green = phase_portrait['LacI']

exp_red = exp_red
exp_green = exp_green
exp_green_ratio = exp_green / (exp_green + exp_red)
exp_red_ratio = 1. - exp_green_ratio

markers = ['o', 'v', "^", "p", 's', '*', '+', 'x']
fig2, ax2 = plt.subplots(1, 1)
# plot modeling steady results of growth rates collected from exp.
for a, i in enumerate(range(len(green_state_tetr) - len(growth_rate_point), len(green_state_tetr))):
    ax2.scatter(green_state_tetr[i], green_state_laci[i], color='#43A047', marker=markers[a], s=500)
    ax2.scatter(red_state_tetr[i], red_state_laci[i], color='#EF5350', marker=markers[a], s=500)
for a, i in enumerate(range(len(green_state_tetr) - len(growth_rate_point), len(green_state_tetr))):
    ax2.scatter(red_unstable[i], green_unstable[i], color='#FFC107', alpha=0.2, marker=markers[a], s=500)

i_max = len(green_state_tetr) - len(growth_rate_point) - 1
ax2.plot(green_state_tetr[sim_index], green_state_laci[sim_index], color='#43A047', ls='--', alpha=0.3)
ax2.plot(red_state_tetr[sim_index], red_state_laci[sim_index], color='#EF5350', ls='--', alpha=0.3)
ax2.plot(red_unstable[sim_index], green_unstable[sim_index], color='#FFC107', ls='--', alpha=0.3)

ax2.scatter(exp_red, exp_green)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(False)
# ax2.set_xlim(.1, 1000)
# ax2.set_ylim(10, 1000)
ax2.set_xlabel('TetR Conc.')
ax2.set_ylabel('LacI Conc.')
fig2.show()

fig5, ax5 = plt.subplots(1, 1)
red_unstable_ratio = (red_unstable / (red_unstable + green_unstable))
red_state_tetr_ratio = (red_state_tetr / (red_state_laci + red_state_tetr))
green_state_tetr_ratio = (green_state_tetr / (green_state_laci + green_state_tetr))
ax5.scatter(growth_rate[sim_index], red_unstable_ratio[sim_index],
            color='y')
ax5.scatter(growth_rate[sim_index], red_state_tetr_ratio[sim_index],
            color='r')
ax5.scatter(growth_rate[sim_index], green_state_tetr_ratio[sim_index],
            color='g')

ax5.scatter(green_ratio_gr, exp_red_ratio, marker='o', facecolors='none', edgecolors='r', lw=4)
ax5.set_yscale('linear')
ax5.set_xlabel('$\lambda$')
ax5.set_ylabel('red ratio')
ax5.set_ylim(-0.2, 1.2)
ax5.grid(False)
fig5.show()
export_data = pd.DataFrame(data=dict(growth_rate=growth_rate[sim_index],
                                     red_unstable_ratio=red_unstable_ratio[sim_index],
                                     red_state_tetr_ratio=red_state_tetr_ratio[sim_index],
                                     green_state_tetr_ratio=green_state_tetr_ratio[sim_index]))
export_data.to_csv(r'./data/bifurcation_gr_vs_Rratio_3.csv')

# %% test plasmid
from toggle_dynamic import gr_pars, hill_pp, assign_vars, ToggleBasic
from scipy.integrate import odeint
from functools import partial
from unilateral_expression_data_fitting import ColE1Plasmid, const_transcript, frc_act_ribo, protein_trans_rate, \
    phi_ribosome, gene_expression2


def plasmid_dydt(y, t, plasmid_obj: ColE1Plasmid, gr_func):
    growth_rate = gr_func(t)
    return plasmid_obj.dev_plasmid(growth_rate, *y)


def exp_eff_dydt(y, t, plasmid_obj: ColE1Plasmid, gr_func):
    exp_eff, plasmid_cp, r1 = y
    growth_rate = gr_func(t)
    transcript_affinity = const_transcript(growth_rate, k=0.25, n=3.2, tau=0.01)
    active_ribo_ratio = frc_act_ribo(growth_rate)
    alph = protein_trans_rate(growth_rate)
    phi_r = phi_ribosome(growth_rate)
    dev_g, dev_r1 = plasmid_obj.dev_plasmid(growth_rate, plasmid_cp, r1)
    exp_eff_1 = plasmid_cp * transcript_affinity * active_ribo_ratio * alph * phi_r * 2.87252353
    dev_exp_eff = exp_eff_1 - exp_eff
    return [dev_exp_eff, dev_g, dev_r1]


def toggle_dydt(y, t, toggle_obj: ToggleBasic, plasmid_obj: ColE1Plasmid, gr_func):
    green, red, green_exp_eff, red_exp_eff, plasmid_cp, r1 = y
    growth_rate = gr_func(t)
    transcript_affinity = const_transcript(growth_rate, k=0.25, n=3.2, tau=0.01)
    active_ribo_ratio = frc_act_ribo(growth_rate)
    alph = protein_trans_rate(growth_rate)
    phi_r = phi_ribosome(growth_rate)
    dev_g, dev_r1 = plasmid_obj.dev_plasmid(growth_rate, plasmid_cp, r1)
    red_exp_eff_1 = plasmid_cp * transcript_affinity * active_ribo_ratio * alph * phi_r * 2.87
    green_exp_eff_1 = plasmid_cp * transcript_affinity * active_ribo_ratio * alph * phi_r * 5.47
    dev_red_exp_eff = red_exp_eff_1 - red_exp_eff
    dev_green_exp_eff = green_exp_eff_1 - green_exp_eff
    toggle_obj.alpha_trc = red_exp_eff
    toggle_obj.alpha_ltet = green_exp_eff
    toggle_obj.gr = growth_rate
    dev_green, dev_red = toggle_obj.field_flow([green, red])
    return [dev_green, dev_red, dev_green_exp_eff, dev_red_exp_eff, dev_g, dev_r1]


def toggle2_dydt(y, t, toggle_obj: ToggleBasic, plasmid_obj: ColE1Plasmid, gr_func):
    green, red, green_exp_eff, red_exp_eff, phi_r, plasmid_cp, r1 = y
    growth_rate = gr_func(t)
    transcript_affinity = const_transcript(growth_rate, k=0.25, n=3.2, tau=0.01)
    active_ribo_ratio = frc_act_ribo(growth_rate)
    alph = protein_trans_rate(growth_rate)
    sigma = growth_rate / phi_r
    dev_phi_r = (sigma * 0.0362 / (1. - 0.2116/2.21 * sigma)) * phi_r - growth_rate * phi_r
    dev_g, dev_r1 = plasmid_obj.dev_plasmid(growth_rate, plasmid_cp, r1)

    red_exp_eff_1 = plasmid_cp * transcript_affinity * alph * active_ribo_ratio * phi_r * 2.87
    green_exp_eff_1 = plasmid_cp * transcript_affinity * alph * active_ribo_ratio * phi_r * 5.47
    # red_exp_eff_1 = plasmid_cp * transcript_affinity * growth_rate * 2.87
    # green_exp_eff_1 = plasmid_cp * transcript_affinity * growth_rate * 5.47
    dev_red_exp_eff = red_exp_eff_1 - red_exp_eff
    dev_green_exp_eff = green_exp_eff_1 - green_exp_eff

    toggle_obj.gr = growth_rate + 0.045
    toggle_obj.alpha_trc = red_exp_eff
    toggle_obj.alpha_ltet = green_exp_eff
    dev_green, dev_red = toggle_obj.field_flow([green, red])
    return [dev_green, dev_red, dev_green_exp_eff, dev_red_exp_eff, dev_phi_r, dev_g, dev_r1]


gr_init = hill_pp(gr_pars, 1e-3)
time_space = np.linspace(1e-3, 60, num=50000)

factor = 1.3
kg = 1.05
toggle_pars = dict(k_t=4.05 * factor, k_l=5.71 * factor * kg, tau_p_ltet=0.0015, tau_p_trc=0.03, n_l=1.4)
toggle = ToggleBasic(growth_rate=gr_init)
assign_vars(toggle, toggle_pars)
toggle.solve_sst(optimize=True)
red_init = toggle.sst_tetr_conc[-1]
green_init = toggle.sst_laci_conc[-1]

plasmid = ColE1Plasmid(growth_rate=gr_init, n=1.2, k_p=60)

_ = plasmid.g_sst
vars_init = [plasmid.g_sst, plasmid.r1]

plasmid_t = odeint(plasmid_dydt, vars_init, time_space, args=(plasmid, partial(hill_pp, gr_pars)))

exp_eff_init = [gene_expression2(gr_init), *vars_init]

exp_eff_t = odeint(exp_eff_dydt, exp_eff_init, time_space, args=(plasmid, partial(hill_pp, gr_pars)))

# toggle_init = [green_init, red_init,
#                gene_expression2(gr_init, rbs=5.47),
#                gene_expression2(gr_init, rbs=2.87),
#                *vars_init]
#
# toggle_t = odeint(toggle_dydt, toggle_init, time_space, args=(toggle, plasmid, partial(hill_pp, gr_pars)))

phi_r_init = phi_ribosome(gr_init)

toggle2_init = [green_init, red_init,
                gene_expression2(gr_init, rbs=5.47),
                gene_expression2(gr_init, rbs=2.87),
                phi_r_init,
                *vars_init]

toggle2_t = odeint(toggle2_dydt, toggle2_init, time_space, args=(toggle, plasmid, partial(hill_pp, gr_pars)))

fig4, ax4 = plt.subplots(4, 2, figsize=(20, 34))
ax = ax4.flatten()
ax[0].plot(time_space, exp_eff_t[:, 1])
ax[0].plot(time_space, plasmid_t[:, 0])
ax[0].set_xlabel('Time (h)')
ax[0].set_ylabel('Plasmid copy number')

ax[1].plot(hill_pp(gr_pars, time_space), exp_eff_t[:, 1])
ax[1].plot(np.linspace(0.1, 1.8), plasmid.get_g_sst(np.linspace(0.1, 1.8)), '--')
ax[1].set_xlabel('Growth rate')
ax[1].set_ylabel('Plasmid copy number')

eff = toggle2_t[:, -2] * partial(hill_pp, gr_pars)(time_space)
ax[2].plot(time_space, eff / eff.max())
ax[2].plot(time_space, exp_eff_t[:, 1] / exp_eff_t[:, 1].max(), '--')

ax[2].set_xlabel('Time (h)')
ax[2].set_ylabel('expression efficiency')

ax[3].plot(hill_pp(gr_pars, time_space), exp_eff_t[:, 0])

ax[3].plot(np.linspace(0.1, 1.8), gene_expression2(np.linspace(0.1, 1.8)), '--')
ax[3].set_xlabel('Growth rate')
ax[3].set_ylabel('expression efficiency')

ax[4].plot(time_space, toggle2_t[:, 0], '-g')
ax[4].plot(time_space, toggle2_t[:, 1], '--r')
ax[4].set_yscale('log')

ax[5].scatter(toggle2_t[:, 1][::200], toggle2_t[:, 0][::200])
ax[5].set_yscale('log')
ax[5].set_xscale('log')

ax[6].plot(time_space, toggle2_t[:, 4])
ax[6].plot(time_space, phi_ribosome(partial(hill_pp, gr_pars)(time_space)), '--')

ax[7].plot(time_space, partial(hill_pp, gr_pars)(time_space))
ax[7].plot(time_space, partial(hill_pp, gr_pars)(time_space)/toggle2_t[:, -3])


fig4.show()
