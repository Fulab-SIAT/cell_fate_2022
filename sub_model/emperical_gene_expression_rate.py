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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Or any other
# […]
import sciplot as splt
from scipy.optimize import least_squares
from sub_model.gene_expression_rate import protein_trans_rate, phi_ribosome, frc_act_ribo, ribo_elongation_klumpp, \
    ribo_klumpp

splt.whitegrid()
from typing import Union


# Own modules
def mRNA_level(growth_rate: Union[list, float], args=None, type=None):
    """
    Wrapper of mRNA levels.

    Parameters
    --------
    growth_rate: list or float
        growth rate
    args: list
        arguments for describing the relationship between growth rate and mRNA level
    type: str.
        the args in defaults, laci or tetr.
    Returns
    --------------
    float or list:
    mRNA level dependent on growth rates.

    """
    if type == 'laci':
        # args = [9.80e+04,  1.8e0, 1.1e1, .49, .49, 2.35, 2.1]
        # args = [18.90e+04, 1.8e0, 1.1e1, .66, .98, 4.65, 1.9]
        args = [18.90e+04,  1.8e0, 1.1e1, .66, .98, 4.2, 1.9]
    elif type == 'tetr':
        # args = [9.95e+04, 9.08e+00, 3.32e+01, 5.43e-01, 3.3e-01, 3.01e+00, 1.8e00]
        # args = [9.95e+04, 9.08e+00, 3.32e+01, 5.43e-01, 3.3e-01, 3.01e+00, 1.8e00]
        args = [8.8e+04, 9.08e+00, 3.32e+01, 5.43e-01, 3.3e-01, 3.1e+00, 1.8e00]
    return hill_function(args, growth_rate)


def alpha_sst(growth_rate, pars=None):
    """
    The steady state of alpha_G,R.

    Parameters
    ---------------
    growth_rate: float or list
        cell growth rate h-1.

    Returns
    ---------------
    green_exp_eff_1, red_exp_eff_1: float
        LacI/GFP expression rate (alpha_G), TetR/RFP expression rate (alpha_R)
    """

    active_ribo_ratio = frc_act_ribo(growth_rate)
    alph = protein_trans_rate(growth_rate)
    if pars is None:
        transcript_affinity_red = mRNA_level(growth_rate, type='tetr') * 10 ** -4
        transcript_affinity_green = mRNA_level(growth_rate, type='laci') * 10 ** -4
        red_exp_eff_1 = transcript_affinity_red * alph * active_ribo_ratio
        green_exp_eff_1 = transcript_affinity_green * alph * active_ribo_ratio
        return green_exp_eff_1, red_exp_eff_1
    elif pars == 'tetr':
        transcript_affinity_red = mRNA_level(growth_rate, type=pars) * 10 ** -4
        red_exp_eff_1 = transcript_affinity_red * alph * active_ribo_ratio
        return red_exp_eff_1
    elif pars == 'laci':
        transcript_affinity_green = mRNA_level(growth_rate, type=pars) * 10 ** -4
        green_exp_eff_1 = transcript_affinity_green * alph * active_ribo_ratio
        return green_exp_eff_1


def hill_function(args, x):
    a, b, n, k1, k2, n1, n2 = args

    # return a / (1. + (x / b) ** n)  # Hill function format
    return a / (1. + (x / b) ** n) * 1. / ((1 + (x / k1) ** n1) * (1 + (k2 / x) ** n2))


def loss_func(args, func, y, x):
    return y - func(args, x)


# %%
if __name__ == '__main__':
    # %% import summary data

    summary_data = pd.read_excel(r'./sub_model/Experimental_data_summary_for_steday_state.xlsx',
                                 sheet_name='Summary Data')
    growth_rate = summary_data['Growth_rate']
    transcript_level = summary_data['mRNA_number_fraction']

    lac_data_filter = np.logical_and(summary_data['Strain'] == 'NH2.23',
                                     ~np.isnan(summary_data['mRNA_number_fraction']))
    tet_data_filter = np.logical_and(summary_data['Strain'] == 'NH2.24',
                                     ~np.isnan(summary_data['mRNA_number_fraction']))
    # %%
    # fit the experimental data
    optimize_laci = least_squares(loss_func, x0=[1.01e+05, 1.8e0, 1.1e1, .49, .49, 2.35, 2.1],
                                  args=(hill_function, transcript_level[lac_data_filter], growth_rate[lac_data_filter]))
    optimize_tetr = least_squares(loss_func, x0=[1.65e5, 1.4e+00, 1e1, .4, .4, 2, 2],
                                  args=(hill_function, transcript_level[tet_data_filter], growth_rate[tet_data_filter]))
    print(optimize_laci, optimize_tetr)
    sim_gr = np.linspace(0.16, 1.65)
    # sim_lac_mRNA = hill_fucction([1.07e+04, 1.61e+00, 4.53e+00], sim_gr)
    # sim_tetr_mRNA = hill_fucction([1.35e+04, 1.30e+00, 2.25e+00], sim_gr)
    # %%
    sim_lac_mRNA = hill_function([10.99e+04, 1.8e0, 1.1e1, .74, .71, 3.9, 1.6], sim_gr)

    sim_tetr_mRNA = hill_function([9.45e+04, 9.48e+00, 3.32e+01, 5.43e-01, 3.3e-01, 2.91e+00, 1.6e0], sim_gr)

    fig1, ax1 = plt.subplots(1, 1)
    ax1.scatter(growth_rate[lac_data_filter], transcript_level[lac_data_filter], c='#1d8e1d')  # green
    # ax1.plot(growth_rate[lac_data_filter], transcript_level[lac_data_filter], c='#1d8e1d')  # green
    ax1.plot(sim_gr, sim_lac_mRNA, '--', c='#1d8e1d')
    ax1.plot(sim_gr, sim_tetr_mRNA, '--', c='#ff6000')

    ax1.scatter(growth_rate[tet_data_filter], transcript_level[tet_data_filter], c='#ff6000')  # red
    # ax1.plot(growth_rate[tet_data_filter], transcript_level[tet_data_filter], c='#ff6000')  # red
    ax1.set_xlim((0, 2))
    splt.aspect_ratio(1)
    ax1.set_xlabel('Growth rate ($h^{-1}$)')
    ax1.set_ylabel('mRNA abundance \n(a.u., mRNA-seq)')

    fig1.show()

    # %% calculate alpha
    gr_std_index = np.argmin(np.abs(sim_gr - 1.))
    sim_lac_alpha = sim_lac_mRNA * phi_ribosome(sim_gr) * protein_trans_rate(sim_gr) * frc_act_ribo(sim_gr)
    sim_tetr_alpha = sim_tetr_mRNA * phi_ribosome(sim_gr) * protein_trans_rate(sim_gr) * frc_act_ribo(sim_gr)
    sim_lac_alpha_klumpp = sim_lac_mRNA * ribo_elongation_klumpp(sim_gr) * ribo_klumpp(sim_gr)
    sim_tetr_alpha_kiumpp = sim_tetr_mRNA * ribo_elongation_klumpp(sim_gr) * ribo_klumpp(sim_gr)
    # sim_lac_alpha_klumpp = sim_lac_mRNA * ribo_klumpp(sim_gr) * ribo_elongation_klumpp(sim_gr)
    # sim_tetr_alpha_kiumpp = sim_tetr_mRNA * ribo_klumpp(sim_gr) * ribo_elongation_klumpp(sim_gr)

    fig2, ax2 = plt.subplots(1, 1)
    # ax2.plot(sim_gr, sim_lac_alpha, '--', c='#1d8e1d')
    # ax2.plot(sim_gr, sim_tetr_alpha, '--', c='#ff6000')
    ax2.plot(sim_gr, sim_lac_alpha_klumpp / sim_lac_alpha_klumpp[gr_std_index], '*-', c='#1d8e1d')
    ax2.plot(sim_gr, sim_tetr_alpha_kiumpp / sim_tetr_alpha_kiumpp[gr_std_index], '*-', c='#ff6000')
    ax2.set_xlim((0, 2))
    splt.aspect_ratio(1)
    ax2.set_xlabel('Growth rate ($h^{-1}$)')
    ax2.set_ylabel('Expression rate \n($\\alpha_{G,R}$)')
    fig2.show()

    df_export = pd.DataFrame(data=dict(gr=sim_gr,
                                       alpha_g=sim_lac_alpha_klumpp / sim_lac_alpha_klumpp[gr_std_index],
                                       alpha_r=sim_tetr_alpha_kiumpp / sim_tetr_alpha_kiumpp[gr_std_index]))

    translation_level = summary_data['Protein expression']
    lac_p_data_filter = np.logical_and(summary_data['Strain'] == 'NH2.23',
                                       ~np.isnan(summary_data['Protein expression']))
    tet_p_data_filter = np.logical_and(summary_data['Strain'] == 'NH2.24',
                                       ~np.isnan(summary_data['Protein expression']))

    sim_norm_laci_exp = sim_lac_alpha_klumpp / sim_gr / (sim_lac_alpha_klumpp / sim_gr)[gr_std_index]
    sim_norm_tetr_exp = sim_tetr_alpha_kiumpp / sim_gr / (sim_tetr_alpha_kiumpp / sim_gr)[gr_std_index]

    fig3, ax3 = plt.subplots(1, 1)
    ax3.scatter(growth_rate[lac_p_data_filter][:-1], (translation_level[lac_p_data_filter][:-1] /
                                                      translation_level[lac_p_data_filter].iloc[3]), c='#1d8e1d')
    ax3.scatter(growth_rate[tet_p_data_filter][:-1],
                (translation_level[tet_p_data_filter][:-1] / translation_level[tet_p_data_filter].iloc[3]), c='#ff6000')

    ax3.plot(sim_gr, sim_norm_laci_exp, '--', c='#1d8e1d')
    ax3.plot(sim_gr, sim_norm_tetr_exp, '--', c='#ff6000')
    splt.aspect_ratio(1)
    ax3.set_xlabel('Growth rate (1/h)')
    ax3.set_ylabel('Expression level \n(a.u., Normalized)')
    fig3.show()
    # %%
    fig4, ax4 = plt.subplots(1, 1)
    ax4.plot(sim_gr, sim_norm_tetr_exp / sim_norm_laci_exp / (sim_norm_tetr_exp / sim_norm_laci_exp)[-1])

    ax4.set_xlim((0, 2))
    ax4.set_ylim(1 / 3, 3 / 1)
    ax4.set_yscale('log')
    splt.aspect_ratio(1, ax=ax4)

    fig4.show()
