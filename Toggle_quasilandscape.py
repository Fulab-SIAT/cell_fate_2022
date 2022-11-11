# -*- coding: utf-8 -*-

"""
Calculates the quasi-potential landscape and visualization.

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
import matplotlib.colors as colors
from matplotlib import cm
from joblib import Parallel, delayed
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import sciplot as splt
from toggle_dynamic import cal_laci_poetntial
splt.whitegrid()

#%%
data_expression = pd.read_csv(r'./data/single_arm_lacI_TetR_expression_level.csv')
data_alpha = pd.read_csv(r'./data/lacI_TetR_expression_level_vs_gr.csv')
al_t_list, al_l_list = data_expression['LacI'].values, data_expression['TetR'].values
growth_rate_point = data_alpha[data_alpha['Type'] == 'LacI']['Growth_rate'].values
growth_rate_sim = np.linspace(0.1, 1.65, num=1000)
laci_list_sim = np.linspace(0, 500, num=10000)
phase_portrait = pd.read_csv('./data/M5_L3_phase_diagram.csv')
green_ratio_exp = phase_portrait['LacI'] / (phase_portrait['TetR'] + phase_portrait['LacI'])
green_ratio_gr = phase_portrait['Growth_rate']
sim_index = slice(0, len(growth_rate_sim))

growth_rate = np.hstack([growth_rate_sim, green_ratio_gr])
laci_list_sim = np.linspace(0, 500, num=10000)
growth_rate_sim = np.linspace(0.1, 1.65, num=1000)
toggle_pars = dict(k_t=1.2, k_l=1.6, tau_p_ltet=0.0015, tau_p_trc=0.005)
sim_index = slice(0, len(growth_rate_sim))

green_state_tetr = np.array([sst[0][-1] for sst in ret])
green_state_laci = np.array([sst[1][-1] for sst in ret])
red_state_tetr = np.array([sst[0][0] for sst in ret])
red_state_laci = np.array([sst[1][0] for sst in ret])

potential_laci = Parallel(n_jobs=-1)(delayed(cal_laci_poetntial)(laci_list=laci_list_sim,
                                                                     gr=growth_rate_sim[i], **toggle_pars)
                                         for i in tqdm(range(len(growth_rate_sim))))
min_potential = np.min(potential_laci)
potential_laci_space = np.vstack([pot.reshape(1, -1) - min_potential for pot in potential_laci])

fig7, ax7 = plt.subplots(1, 1)
poten_im = ax7.imshow(potential_laci_space,
                        origin='lower', extent=[laci_list_sim[0], laci_list_sim[-1],
                                                growth_rate_sim[0], growth_rate_sim[-1]],
                        norm=colors.LogNorm(vmin=1, vmax=2000), aspect='auto', cmap='summer'
                        )
cb = fig7.colorbar(poten_im, ax=ax7)
cb.set_label("$U$")
ax7.plot(green_state_laci[sim_index], growth_rate[sim_index], 'g--')
ax7.plot(red_state_laci[sim_index], growth_rate[sim_index], 'r--')
ax7.set_xlim(0, 80)
ax7.grid(False)
ax7.set_xlabel('LacI conc.')
ax7.set_ylabel('Growth rate ($h^{-1}$)')
fig7.show()


def find_index(s, p):
    space_shape = s[0].shape
    d = np.zeros(space_shape)
    for i, ss in enumerate(s):
        d += np.abs(ss - p[i])
    index = np.unravel_index(np.argmin(d, axis=None), space_shape)
    return index


lac_mesh, growth_mesh = np.meshgrid(laci_list_sim, growth_rate_sim)
# index_green = [find_index([lac_mesh, growth_mesh], [laci, growth_rate[sim_index][i]])
#                 for i, laci in tqdm(enumerate(green_state_laci[sim_index]))]
# index_red = [find_index([lac_mesh, growth_mesh], [laci, growth_rate[sim_index][i]])
#                 for i, laci in tqdm(enumerate(red_state_laci[sim_index]))]

index_green = Parallel(n_jobs=-1)(delayed(find_index)([lac_mesh, growth_mesh], [laci, growth_rate[sim_index][i]])
                                    for i, laci in tqdm(enumerate(green_state_laci[sim_index])))
index_red = Parallel(n_jobs=-1)(delayed(find_index)([lac_mesh, growth_mesh], [laci, growth_rate[sim_index][i]])
                                for i, laci in tqdm(enumerate(red_state_laci[sim_index])))
green_potential = np.array([potential_laci_space[index] for index in index_green])

red_potential = np.array([potential_laci_space[index] for index in index_red])

new_coolwarm = cm.get_cmap('coolwarm')
new_coolwarm = new_coolwarm(np.linspace(0, 1, 256))[60:-30]
new_coolwarm = ListedColormap(new_coolwarm)
vmax = 1400
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(24, 12))
surface = ax.plot_surface(lac_mesh, growth_mesh, potential_laci_space,
                            linewidth=0, antialiased=False, cmap=new_coolwarm,
                            rcount=100, ccount=100,
                            norm=colors.PowerNorm(vmin=0, vmax=vmax, gamma=18))
cb = fig.colorbar(surface, shrink=0.5, aspect=5, pad=0.1)
cb.set_label('$U$')
ax.plot(red_state_laci[sim_index], growth_rate[sim_index], red_potential, color='#E74C3C', ls=':')
ax.plot(green_state_laci[sim_index], growth_rate[sim_index], green_potential, color='#3CE74C', ls=':')
ax.tick_params(axis='z', pad=25)
ax.tick_params(axis='y', pad=25)

ax.set_xlabel('LacI conc. (a.u.)', labelpad=25)
ax.set_ylabel('Growth rate ($h^{-1}$)', labelpad=50)
ax.set_zlabel('U (a.u.)', labelpad=50)
ax.view_init(30, -90)
# ax.set_zlim(1, 2000)
# fig.savefig(r'./data/3d_potential_laci.svg')
fig.show()
