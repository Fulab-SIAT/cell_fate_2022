# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import pandas as pd
import numpy as np  # Or any other
import json
# […]
import time
# Own modules
from toggle_potential_landscape import TogglePotential, simulation_process
from toggle_dynamic import ToggleBasic, assign_vars
from pde_solver_cpy import crate_init_p, evolve_ADI  # evaluating the PDEs via C compiled codes.
import matplotlib.pyplot as plt

from multiprocessing import Process

# %%
jsonfile = open(r'./data/potenial_strains_parameters_LO3_2.json', 'r')
strains_pars = json.load(jsonfile)
jsonfile.close()


processList = []
for strainPars in strains_pars:

    toggle = ToggleBasic()
    strain = strainPars['strain']
    toggle_pars = strainPars['toggle_pars']
    growth_rate = strainPars['growth_rate']
    time_duration = np.log(2) / growth_rate * 60  # strainPars['time_duration']  # 40 generations
    t_step = time_duration / 5000  # strainPars['t_step']  # 5000 round evolution
    grid_num = strainPars['grid_num']
    x1 = np.linspace(0, strainPars['lacI_max'], endpoint=True, num=grid_num)
    x2 = np.linspace(0, strainPars['TetR_max'], endpoint=True, num=grid_num)
    D = strainPars['D']
    cov = np.array([D, D]).reshape(-1, 1) * np.eye(2)

    x1_step = x1[1] - x1[0]
    x2_step = x1[1] - x1[0]

    assign_vars(toggle, toggle_pars)
    toggle.growth_rate = growth_rate
    toggle.solve_sst(x1)
    if len(toggle.sst_tetr_conc) > 2:
        mean = np.array([toggle.sst_laci_conc[-2], toggle.sst_tetr_conc[-2]]).reshape(-1, 1)
    else:
        mean = np.array([toggle.sst_laci_conc[0], toggle.sst_tetr_conc[0]]).reshape(-1, 1)
    # mean = np.array([32.07, 0.53]).reshape(-1, 1)
    x1_len, x2_len = len(x1), len(x2)
    x1_v, x2_v = np.meshgrid(x1, x2, indexing='ij')
    p_0 = crate_init_p(x1_v.flatten(), x2_v.flatten(), cov, mean).reshape(x1_v.shape)

    p_0[:, 0] = 0.
    p_0[0, :] = 0.
    p_0[:, -1] = 0.
    p_0[-1, :] = 0

    time = np.arange(0, time_duration, t_step)

    fig2, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.plot(x1, toggle.null_cline_tetr(x1), '--r')
    ax.plot(toggle.null_cline_laci(x2), x2, '--g')
    ax.imshow(p_0.T, origin='lower', cmap='coolwarm', extent=(x1.min(), x1.max(), x2.min(), x2.max()))
    ax.set_xlim(x1[0], x1[-1])
    ax.set_ylim(x2[0], x2[-1])
    ax.set_xlabel('LacI Conc.')
    ax.set_ylabel('TetR Conc.')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, strainPars['lacI_max'])
    ax.set_ylim(1, strainPars['TetR_max'])
    ax.set_title(f'{strain}_{growth_rate}_{D}')
    fig2.show()

    potential_args = dict(laci_list=x1, tetr_list=x2, time_list=time, D_1=cov[0, 0] / 2, D_2=cov[1, 1] / 2, mean=mean,
                          cov=cov,
                          gr=growth_rate)

    process = Process(target=simulation_process,
                      args=(toggle_pars, potential_args,
                            f'H:\potential_data\{strain}_{growth_rate}_{D}.bin'))
                        # f'/media/fulab/fulab_zc_1/potential_data/{strain}_{growth_rate}_{D}.bin'
    process.start()
    processList.append(process)

    # toggle_potential = TogglePotential(toggle_pars=toggle_pars, landscape_pars=potential_args)
    # toggle_potential.start_evolution()
    # toggle_potential.save_rets(os.path.join('./data', f'{strain}_{growth_rate}_{D}.bin'))

for process in processList:
    process.join()


