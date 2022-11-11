# -*- coding: utf-8 -*-
# %%
"""
 This script is used to fit the exp data of Ptrc_vars promoters responding LacI and optimize the parameters of these
 promoters.

 @author: Pan M. CHU
 @Copyright: Copyright 2021, toggle mini
 @Email: pan_chu@outlook.com
"""
__author__ = "Pan M. CHU"
__email__ = 'pan_chu@outlook.com'

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Or any other
# […]


from unilateral_expression_data_fitting import laci_inhibit, inhibit_response, inhibit_response_fixn, \
    inhibit_response_cgk, \
    inhibit_respnse_normalize, inhibit_respnse_normalize_fixn, inhibit_response_nfix_curve_fit
from scipy.optimize import leastsq


def filter_func(str):
    column_index = int(str.split('-')[-1][1:])
    if column_index > 20:
        return False
    else:
        return True


# %%
"""
Set all parameters free
"""
fig1, ax1 = plt.subplots(1, 1)
in_sim = np.logspace(-1, 4, num=1000)

# data_ps = r'./data/20210619_sali_laci_response_summary.xlsx'
data_ps = r'./data/normalized_laci_response.xlsx'
data = pd.read_excel(data_ps)
# data = data[[filter_func(name) for name in data['Tube Name:']]]
sample_list = list(set(data['Strain']) - set(['INPUT', 'BAC']))
sample_list.sort()
# sample_list = ['L3', 'L4', 'WT']
#%%
all_rets = []
sim_data = in_sim.reshape(-1, 1)
for strain_na in sample_list:
    out_put = data[data['Strain'] == strain_na]['Output'].values
    input = data[data['Strain'] == strain_na]['Input'].values

    init_pars = [1.15, 10]

    opt_pars = leastsq(laci_inhibit, init_pars,
                       args=(input, out_put, inhibit_respnse_normalize))[0]
    rets = list(opt_pars) + [strain_na]
    all_rets.append(rets)
    print(f'{strain_na} -> pars: {opt_pars}')
    y_sim = inhibit_respnse_normalize(opt_pars, in_sim)
    sim_data = np.hstack((sim_data, y_sim.reshape(-1, 1)))
    ax1.scatter(input, out_put)
    ax1.plot(in_sim, y_sim, label=f'{strain_na} $\\rightarrow$ k:%.3f' % opt_pars[-1], alpha=0.2)
    ax1.grid(False)
    ax1.set_xscale('log')
    ax1.set_xlim(1, 10000)
    ax1.set_ylim(1e-5, 1)
    ax1.set_yscale('log')
    # ax1.set_xticks([1, 100], minor=False)
    # ax1.set_xticks([], minor=True)
    # ax1.set_yticks([100, 1000], minor=False)
    # ax1.set_yticks([], minor=True)

ax1.legend()
fig1.show()

# dataframe = pd.DataFrame(data=all_rets, columns=['y_min', 'y_max', 'n', 'Kd', 'Strain'])
dataframe = pd.DataFrame(data=all_rets, columns=['n', 'Kd', 'Strain'])

dataframe.to_csv(data_ps + '.fitting_paras_allfree.csv')
sim_data = pd.DataFrame(data=sim_data, columns=['input'] + sample_list)
sim_data.to_csv(os.path.join(data_ps + 'sim_data_bestfit.csv'))

# %%
"""
Set n fixed
"""
fig1, ax1 = plt.subplots(1, 1)
in_sim = np.logspace(-2, 4, num=1000)

# data = pd.read_excel(data_ps)
# data = data[[filter_func(name) for name in data['Tube Name:']]]
sample_list = list(set(data['Strain']) - set(['INPUT', 'BAC']))
sample_list.sort()

# sample_list = ['L3', 'L4', 'WT']
all_rets = []
sim_data = np.ndarray([])
opt_rets = []
sim_data = in_sim.reshape(-1, 1)
from scipy.optimize import curve_fit
from functools import partial
for strain_na in sample_list:
    out_put = data[data['Strain'] == strain_na]['Output'].values
    input = data[data['Strain'] == strain_na]['Input'].values
    init_pars = [5]
    n_t = 1.15
    opt_ret = leastsq(laci_inhibit, init_pars,
                      args=(input, out_put, inhibit_respnse_normalize_fixn, dict(n=n_t)), ftol=True)
    fit_ret = curve_fit(f=partial(inhibit_response_nfix_curve_fit, n=n_t),
                        xdata=input, ydata=np.log(out_put), p0=init_pars)
    opt_rets.append(fit_ret)
    opt_pars = opt_ret[0]
    rets = list(opt_pars) + [strain_na]
    all_rets.append(rets)
    print(f'{strain_na} -> pars: {opt_pars}')
    y_sim = inhibit_respnse_normalize_fixn(opt_pars, in_sim, n=n_t)
    sim_data = np.hstack((sim_data, y_sim.reshape(-1, 1)))
    ax1.scatter(input, out_put, s=400)
    # ax1.scatter(data[np.logical_and(data['Strain'] == strain_na,
    #                                 data['date'] == 21210616)
    #                  ]['Input'].values, data[np.logical_and(data['Strain'] == strain_na,
    #                                 data['date'] == 21210616)
    #                  ]['Output'].values, s=400, color='r')

    ax1.plot(in_sim, y_sim, label=f'{strain_na} $\\rightarrow$ k:%.3f' % opt_pars[-1], alpha=0.2)
    ax1.grid(False)
    ax1.set_xscale('log')
    ax1.set_xlim(1, 10000)
    ax1.set_ylim(1e-4, 1)
    ax1.set_yscale('log')

    ax1.set_xlabel('Input: Psali (a.u.)')
    ax1.set_ylabel('Output: Ptrc_vars (a.u.)')
    # ax1.set_xticks([1, 100], minor=False)
    # ax1.set_xticks([], minor=True)
    # ax1.set_yticks([100, 1000], minor=False)
    # ax1.set_yticks([], minor=True)

ax1.legend()
fig1.show()
#%%
# dataframe = pd.DataFrame(data=all_rets, columns=['y_min', 'y_max', 'Kd', 'Strain'])
dataframe = pd.DataFrame(data=all_rets, columns=['Kd', 'Strain'])
dataframe.to_csv(data_ps + '.fitting_paras_fixn.csv')
sim_data = pd.DataFrame(data=sim_data, columns=['input'] + sample_list)
sim_data.to_csv(os.path.join(data_ps + 'sim_data_fixn.csv'))

# %%
"""
Normailze the value
"""
fig1, ax1 = plt.subplots(1, 1)

sample_list = ['L3', 'L4', 'WT']
color_list = dict(L3='#41F0AE', L4='#B856D7', WT='#FF8080')
norm_sim = np.array([])
sim_input = sim_data['input']
norm_sim = np.hstack([sim_input]).reshape(-1, 1)
for strain_na in sample_list:
    strain_pars = dataframe[dataframe['Strain'] == strain_na]
    y_min, y_max = strain_pars['y_min'].values, strain_pars['y_max'].values
    out_put = data[data['Strain'] == strain_na]['Output'].values
    norm_output = (out_put - y_min) / y_max
    input = data[data['Strain'] == strain_na]['Input'].values
    sim_output = sim_data[strain_na].values
    norm_sim_output = (sim_output - y_min) / y_max
    norm_sim = np.hstack([norm_sim, norm_sim_output.reshape(-1, 1)])

    ax1.scatter(input, norm_output,
                s=400, c=color_list[strain_na])
    ax1.plot(sim_input, norm_sim_output,
             label=f'{strain_na}', alpha=0.2, c=color_list[strain_na])
    ax1.grid(False)
    ax1.set_xscale('log')
    ax1.set_xlim(1, 2000)

    # ax1.set_yscale('log')
    # ax1.set_ylim(1e-4, 1)

    ax1.set_xlabel('Input: Psali (a.u.)')
    ax1.set_ylabel('Normalized output: Ptrc_vars (a.u.)')
    ax1.set_xticks([100, 1000], minor=False)
    ax1.set_xticks([], minor=True)
    ax1.set_yticks([1e-4, 1e-2], minor=False)
    ax1.set_yticks([], minor=True)

ax1.legend()
fig1.show()
norm_sim = pd.DataFrame(data=norm_sim, columns=['input'] + sample_list)
norm_sim.to_csv(os.path.join(data_ps + 'norm_sim_data_fixn.csv'))

# sim_data = pd.DataFrame(data=sim_data, columns=['input'] + sample_list)
# sim_data.to_csv(os.path.join(data_ps + 'sim_data_fixn.csv'))
