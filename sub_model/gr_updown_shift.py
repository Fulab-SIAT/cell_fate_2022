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
from sciplot import whitegrid, aspect_ratio

whitegrid()
# […]

from toggle_dynamic import assign_vars, ToggleBasic
from sub_model.gene_expression_rate import plasmid_prime, const_transcript, frc_act_ribo, \
    protein_trans_rate, phi_ribosome, plasmid_sst
from sub_model.emperical_gene_expression_rate import mRNA_level, alpha_sst

from scipy.integrate import odeint
from functools import partial


def plasmid_dydt(g_n, t, gr_func):
    """
    derivation of plasmids.

    Parameters
    ------
    g_n: float
        plasmid copy number at time T.

    Returns
    --------
    g_n_dev: float
        derivation of plasmid copy number.

    """
    growth_rate = gr_func(t)
    g_n_dev = plasmid_prime(g_n, t, growth_rate=growth_rate)
    return g_n_dev


def exp_eff_dydt(y, t, gr_func):
    exp_eff, plasmid_cp, r1 = y
    growth_rate = gr_func(t)
    transcript_affinity = const_transcript(growth_rate, k=0.25, n=3.2, tau=0.01)
    active_ribo_ratio = frc_act_ribo(growth_rate)
    alph = protein_trans_rate(growth_rate)
    phi_r = phi_ribosome(growth_rate)
    dev_g = plasmid_prime(plasmid_cp, t, growth_rate=growth_rate)
    exp_eff_1 = plasmid_cp * transcript_affinity * active_ribo_ratio * alph * phi_r
    dev_exp_eff = exp_eff_1 - exp_eff
    return [dev_exp_eff, dev_g]


def toggle2_dydt(y, t, toggle_obj: ToggleBasic, gr_func) -> list:
    """

    Returns
    ------------------
    dev_list: list
        [dev_green, dev_red, dev_green_exp_eff, dev_red_exp_eff, dev_phi_r, dev_g]
    """
    green, red, green_exp_eff, red_exp_eff, phi_r, plasmid_cp = y
    growth_rate = gr_func(t)
    sigma = growth_rate / phi_r
    growth_rate_quasi_stdady = sigma * 0.05 / (1. - 0.09 * sigma)

    active_ribo_ratio = frc_act_ribo(growth_rate_quasi_stdady)
    alph = protein_trans_rate(growth_rate_quasi_stdady)

    dev_phi_r = growth_rate_quasi_stdady * phi_r - growth_rate * phi_r
    dev_g = plasmid_prime(plasmid_cp, t, growth_rate=growth_rate, args=[8, 15, 1.0])
    transcript_affinity_red = const_transcript(growth_rate_quasi_stdady, strain='Red')
    transcript_affinity_green = const_transcript(growth_rate_quasi_stdady, strain='Green')

    # red_exp_eff_1 = plasmid_cp * transcript_affinity_red * alph * active_ribo_ratio * phi_r
    # green_exp_eff_1 = plasmid_cp * transcript_affinity_green * alph * active_ribo_ratio * phi_r
    red_exp_eff_1 = plasmid_cp * transcript_affinity_red * alph * active_ribo_ratio / \
                    growth_rate_quasi_stdady * sigma * phi_r
    green_exp_eff_1 = plasmid_cp * transcript_affinity_green * alph * active_ribo_ratio / \
                      growth_rate_quasi_stdady * sigma * phi_r

    dev_red_exp_eff = red_exp_eff_1 - red_exp_eff
    dev_green_exp_eff = green_exp_eff_1 - green_exp_eff

    toggle_obj.gr = growth_rate
    toggle_obj.alpha_trc = red_exp_eff_1
    toggle_obj.alpha_ltet = green_exp_eff_1
    dev_green, dev_red = toggle_obj.field_flow([green, red])
    dev_list = [dev_green, dev_red, dev_green_exp_eff, dev_red_exp_eff, dev_phi_r, dev_g]
    return dev_list


def toggle_emperical_dydt(y, t, toggle_obj: ToggleBasic, gr_func) -> list:
    """

    Returns
    ------------------
    dev_list: list
        [dev_green, dev_red, dev_phi_r, dev_g]
    """
    green, red, phi_r = y
    growth_rate = gr_func(t)
    sigma = growth_rate / phi_r
    growth_rate_quasi_stdady = sigma * 0.05 / (1. - 0.09 * sigma)

    active_ribo_ratio = frc_act_ribo(growth_rate_quasi_stdady)
    kappa_t = protein_trans_rate(growth_rate_quasi_stdady)

    dev_phi_r = growth_rate_quasi_stdady * phi_r - growth_rate * phi_r
    transcript_affinity_red = mRNA_level(growth_rate_quasi_stdady, type='tetr') * 10 ** -4
    transcript_affinity_green = mRNA_level(growth_rate_quasi_stdady, type='laci') * 10 ** -4

    red_exp_eff = transcript_affinity_red * kappa_t * active_ribo_ratio / \
                    growth_rate_quasi_stdady * sigma * phi_r
    green_exp_eff = transcript_affinity_green * kappa_t * active_ribo_ratio / \
                      growth_rate_quasi_stdady * sigma * phi_r

    # dev_red_exp_eff = red_exp_eff_1 - red_exp_eff
    # dev_green_exp_eff = green_exp_eff_1 - green_exp_eff

    toggle_obj.gr = growth_rate
    toggle_obj.set_alpha_trc(red_exp_eff)
    toggle_obj.set_alpha_ltet(green_exp_eff)
    dev_green, dev_red = toggle_obj.field_flow([green, red])
    dev_list = [dev_green, dev_red, dev_phi_r]
    return dev_list


def expression_alpha_sst(growth_rate):
    # phi_r = phi_ribosome(growth_rate)
    plasmid_cp = plasmid_sst(growth_rate, args=[8, 15])
    active_ribo_ratio = frc_act_ribo(growth_rate)
    alph = protein_trans_rate(growth_rate)
    transcript_affinity_red = const_transcript(growth_rate, strain='Red')
    transcript_affinity_green = const_transcript(growth_rate, strain='Green')
    red_exp_eff_1 = plasmid_cp * transcript_affinity_red * alph * active_ribo_ratio
    green_exp_eff_1 = plasmid_cp * transcript_affinity_green * alph * active_ribo_ratio
    return green_exp_eff_1, red_exp_eff_1


def const_exp_dydt(y, t, gr_func) -> list:
    alpha, phi_r, plasmid_cp = y
    growth_rate = gr_func(t)
    sigma = growth_rate / phi_r
    growth_rate_quasi_stdady = sigma * 0.05 / (1. - 0.09 * sigma)
    active_ribo_ratio = frc_act_ribo(growth_rate_quasi_stdady)
    alph = protein_trans_rate(growth_rate_quasi_stdady)
    # dev_phi_r = (sigma * 0.0362 / (1. - 0.2116 / 2.21 * sigma)) * phi_r - growth_rate * phi_r
    dev_phi_r = (sigma * 0.05 / (1. - 0.09 * sigma)) * phi_r - growth_rate * phi_r
    # dev_g = plasmid_prime(plasmid_cp, t, growth_rate=growth_rate, args=[3.5, 15.1, 1.0])
    dev_g = plasmid_prime(plasmid_cp, t, growth_rate=growth_rate, args=[8, 15, 1.0])

    transcript_affinity_red = const_transcript(growth_rate_quasi_stdady, strain='Green')
    # red_exp_eff_1 = plasmid_cp * transcript_affinity_red * alph * active_ribo_ratio * phi_r * 2.87
    par_alpha = plasmid_cp * \
                transcript_affinity_red / (0.09 * growth_rate_quasi_stdady + 0.05) \
                * phi_r * alph * active_ribo_ratio \
                / growth_rate_quasi_stdady * phi_r * sigma - growth_rate * alpha
    dev_list = [par_alpha, dev_phi_r, dev_g]
    return dev_list


def sigma_lambda_sst(lambda_):
    return lambda_ / (0.09 * lambda_ + 0.05)


# Own modules
gr_pars = [-19.23669571, 3.033802, 2.10852488, 4,
           1.63754155, 0.16819914, 0.68346174, 5.3086755,
           0.8491872, 98.19158635, 45.42363766, 1.36716679]

up_down_shift_pars = [-19.23669571, 3.033802, 2.10852488, 4,
                      1.63754155, 0.16819914, 0.68346174, 5.3086755,
                      0.8491872, 2000.19158635, 45.42363766, 1.36716679]

m5_l2_up_down_shift_pars = [-19.23669571, 3.033802, 2.10852488, 4,
                            1.63754155, 0.16819914, 0.68346174, 5.3086755,
                            0.8491872, 98.19158635, 45.42363766, 1.36716679]


def hill_pp(pars, time):
    n1, k1, n2, k2, g1, g2, a, b, c, n3, k3, g3 = pars
    rate1 = (g1 * (time / k1) ** n1) / (1 + (time / k1) ** n1)
    rate2 = (g2 * (time / k2) ** n2) / (1 + (time / k2) ** n2)
    rate3 = a * np.exp(-(time - b) ** 2 / (2 * c ** 2))
    rate4 = (g3 * (time / k3) ** n3) / (1 + (time / k3) ** n3)
    return rate1 + rate2 + rate3 + rate4


# %%
if __name__ == '__main__':
    # %%
    ud_shift_data_ps = r'sub_model/growth_rate_down_up_shift_20210310.xlsx'

    m5_l3_pd_shift_data = pd.read_excel(ud_shift_data_ps, sheet_name='M5_L3_growth_rate')
    # clean data
    m5_l3_pd_shift_data = m5_l3_pd_shift_data[np.logical_and(~np.isnan(m5_l3_pd_shift_data.iloc[:, 1]),
                                                             ~np.isnan(m5_l3_pd_shift_data.iloc[:, 0]))]

    l2_pd_shift_data = pd.read_excel(ud_shift_data_ps, sheet_name='L2_growth_rate')
    # clean data
    l2_pd_shift_data = l2_pd_shift_data[np.logical_and(~np.isnan(l2_pd_shift_data.iloc[:, 1]),
                                                       ~np.isnan(l2_pd_shift_data.iloc[:, 0]))]

    m5_l3_up_down_shift_pars = [-19.23669571, 3.033802, 2.10852488, 4,
                                1.63754155, 0.2119914, 0.68346174, 5.3086755,
                                0.8491872, 5e2, 45.42363766, 1.36716679]

    time_list = np.linspace(0, 51, num=1000)

    sim_gr = hill_pp(time=time_list, pars=m5_l3_up_down_shift_pars)
    sim_l2_gr = hill_pp(time=time_list, pars=m5_l3_up_down_shift_pars)

    fig1, ax1 = plt.subplots(1, 1, figsize=(15, 8))
    ax1.scatter(m5_l3_pd_shift_data['Time (h)'], m5_l3_pd_shift_data['Growth rate (h-1)'], c='#898686')
    ax1.plot(time_list, sim_gr, c='#00a8ff')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Growth rate ($h^{-1}$)')
    fig1.show()

    # %%
    fig2, ax2 = plt.subplots(1, 1, figsize=(15, 8))
    ax2.scatter(l2_pd_shift_data['Time (h)'], l2_pd_shift_data['Growth rate (h-1)'], c='#898686')
    ax2.plot(time_list, sim_l2_gr, c='#00a8ff')
    ax2.set_xlabel('Time (h)')
    ax2.set_ylabel('Growth rate ($h^{-1}$)')
    fig2.show()


    # %%
    m5_l3_up_down_shift_pars = [-19.23669571, 3.033802, 2.10852488, 4,
                                1.63754155, 0.2119914, 0.68346174, 5.3086755,
                                0.8491872, 1e2, 45.42363766, 1.36716679]
    ud_shift_data_ps = r'sub_model/growth_rate_down_up_shift_20210310.xlsx'
    m5_l3_pd_fluorescent = pd.read_excel(ud_shift_data_ps, sheet_name='M5_L3_fluorescent')
    gr_pars = m5_l3_up_down_shift_pars
    gr_init = hill_pp(gr_pars, 1e-3)
    time_space = np.linspace(1e-3, 52, num=5000)
    factor = 4.2
    kg = .1
    toggle_pars = dict(k_t=3.552 * factor, k_l=4 * factor * kg, tau_p_ltet=0.002, tau_p_trc=0.035, n_l=1.4)
    # ============ Simulation Parameters ====================== #
    gene_alpha_green, gene_alpha_red = expression_alpha_sst(growth_rate=gr_init)
    toggle = ToggleBasic(growth_rate=gr_init)
    assign_vars(toggle, toggle_pars)
    # toggle.growth_rate = gr_init
    toggle.set_alpha_trc(gene_alpha_red)
    toggle.set_alpha_ltet(gene_alpha_green)
    toggle.solve_sst(optimize=True)
    red_init = toggle.sst_tetr_conc[-1]
    green_init = toggle.sst_laci_conc[-1]

    plasmid_g_sst = plasmid_sst(growth_rate=gr_init)

    # plasmid_t = odeint(plasmid_dydt, plasmid_g_sst, time_space, args=(partial(hill_pp, gr_pars),))

    # phi_r_init = phi_ribosome(gr_init)
    phi_r_init = 0.09 * gr_init + 0.05

    toggle2_init = [green_init, red_init,
                    gene_alpha_green,
                    gene_alpha_red,
                    phi_r_init,
                    plasmid_g_sst]

    toggle2_t = odeint(toggle2_dydt, toggle2_init, time_space, args=(toggle, partial(hill_pp, gr_pars)))
    const_exp_t = odeint(const_exp_dydt, [red_init, phi_r_init, plasmid_g_sst], time_space,
                         args=(partial(hill_pp, gr_pars),))

    #
    fig5, ax5 = plt.subplots(1, 1, figsize=(12, 12))
    ax5.plot(toggle2_t[:, 1] / np.max(toggle2_t[:, 1]), toggle2_t[:, 0] / np.max(toggle2_t[:, 0]),
             label='Sim.', color='#00a8ff')
    ax5.scatter(m5_l3_pd_fluorescent['Green'] / m5_l3_pd_fluorescent['Green'].max(),
                m5_l3_pd_fluorescent['Red'] / m5_l3_pd_fluorescent['Red'].max(),
                label='Exp.', color='#898686', alpha=.8)
    ax5.set_yscale('log')
    ax5.set_xscale('log')
    ax5.set_xlabel('Relative green intensity (a.u.)')
    ax5.set_ylabel('Relative red intensity (a.u.)')

    ax5.legend()
    fig5.show()

    #
    fig6, ax6 = plt.subplots(1, 1, figsize=(18, 12))
    ax6.plot(time_space, toggle2_t[:, 0] / np.max(toggle2_t[:, 0]), '-g', label='Sim. Green')
    ax6.plot(time_space, toggle2_t[:, 1] / toggle2_t[:, 1].max(), '--r', label='Sim. Red')
    # ax6.plot(time_space, toggle2_t[:, -1] / toggle2_t[:, -1].max(), label='Dynamic plasmid', color='#a6d4ff')
    # ax6.plot(time_space, toggle2_t[:, -2], label='Dynamic ribo')
    # ax6.plot(time_space, toggle2_t[:, 2] / toggle2_t[:, 2].max(), label='Dynamic green')

    ax6.scatter(m5_l3_pd_fluorescent['Time (h)'], m5_l3_pd_fluorescent['Red'] / m5_l3_pd_fluorescent['Red'].max(),
                label='Exp. Red', alpha=.8, color='#ff8080')
    ax6.scatter(m5_l3_pd_fluorescent['Time (h)'], m5_l3_pd_fluorescent['Green'] / m5_l3_pd_fluorescent['Green'].max(),
                label='Exp. Green', alpha=.8, color='#92d050')
    ax6.set_ylabel('Relative fluorescent intensity (a.u.)')
    ax6.set_xlabel('Time (h)')
    ax6.legend()
    fig6.show()

    # %%
    fig7, ax7 = plt.subplots(1, 1, figsize=(18, 12))
    ax7.plot(time_space, toggle2_t[:, -1], label='Dynamic', color='#a6d4ff')
    ax7.plot(time_space, plasmid_sst(growth_rate=partial(hill_pp, gr_pars)(time_space)),
             label='Steady State', color='#ffa400')
    ax7.set_xlabel('Time (h)')
    ax7.set_ylabel('Plasmid copy number')
    ax7.legend()
    fig7.show()


    # %%  change the parameters of toggle and plot their trajectory.
    gr_pars = m5_l3_up_down_shift_pars
    gr_init = hill_pp(gr_pars, 1e-3)
    time_space = np.linspace(1e-3, 52, num=500)
    factor = 4.2
    kg = .1
    toggle_pars_1 = dict(k_t=4.05 * factor, k_l=5.71 * factor * kg, tau_p_ltet=0.01, tau_p_trc=0.03, n_l=1.4)
    toggle_pars_2 = dict(k_t=4.05 * factor, k_l=5.71 * factor * kg * 5 / 10.5, tau_p_ltet=0.01, tau_p_trc=0.03, n_l=1.4)
    toggle_pars_3 = dict(k_t=4.05 * factor, k_l=5.71 * factor * kg * 2 / 10.5, tau_p_ltet=0.01, tau_p_trc=0.03, n_l=1.4)
    toggle_pars_list = [toggle_pars_1, toggle_pars_2, toggle_pars_3]
    toggle_sim = []
    for toggle_pars in toggle_pars_list:
        # ============ Simulation Parameters ====================== #
        gene_alpha_green, gene_alpha_red = expression_alpha_sst(growth_rate=gr_init)
        toggle = ToggleBasic(growth_rate=gr_init)
        assign_vars(toggle, toggle_pars)
        toggle.growth_rate = gr_init
        toggle.set_alpha_trc(gene_alpha_red)
        toggle.set_alpha_ltet(gene_alpha_green)
        toggle.solve_sst(optimize=True)
        red_init = toggle.sst_tetr_conc[-1]
        green_init = toggle.sst_laci_conc[-1]

        plasmid_g_sst = plasmid_sst(growth_rate=gr_init)

        # plasmid_t = odeint(plasmid_dydt, plasmid_g_sst, time_space, args=(partial(hill_pp, gr_pars),))

        phi_r_init = phi_ribosome(gr_init)

        toggle2_init = [green_init, red_init,
                        gene_alpha_green,
                        gene_alpha_red,
                        phi_r_init,
                        plasmid_g_sst]

        toggle_sim.append(odeint(toggle2_dydt, toggle2_init, time_space, args=(toggle, partial(hill_pp, gr_pars))))
    fig8, ax8 = plt.subplots(1, 3, figsize=(25, 9), sharey=True)
    for i, toggle_time in enumerate(toggle_sim):
        down_shift_index = time_space <= 42.
        up_shift_index = time_space > 42.

        ax8[i].plot(toggle_time[:, 1][down_shift_index] / np.max(toggle_time[:, 1]),
                    toggle_time[:, 0][down_shift_index] / np.max(toggle_time[:, 0]),
                    '--', color='#5c5cfc')
        ax8[i].plot(toggle_time[:, 1][up_shift_index] / np.max(toggle_time[:, 1]),
                    toggle_time[:, 0][up_shift_index] / np.max(toggle_time[:, 0]),
                    '--', color='#ff8900')
        ax8[i].set_xscale('log')
        ax8[i].set_yscale('log')
        ax8[i].set_xlim(1e-2, 1.5)
        ax8[i].set_ylim(1e-2, 1.5)
    fig8.show()

    # %% this part used to explore the gene expression dynamics
    growth_range = np.linspace(0.1, 1.8)
    ref_index = np.argmin(np.abs(growth_range - 1.0))
    sigma_vs_gr = sigma_lambda_sst(growth_range)
    plasmid_vs_gr = plasmid_sst(growth_range, args=[8, 15])
    phi_r_vs_gr = 0.09 * growth_range + 0.05
    trans_reg_vs_gr = const_transcript(growth_range, strain='Green') / phi_r_vs_gr

    m5_l3_up_down_shift_pars = [-19.23669571, 3.033802, 2.10852488, 4,
                                1.63754155, 0.2119914, 0.68346174, 5.3086755,
                                0.8491872, 5e2, 45.42363766, 1.36716679]

    gr_pars = m5_l3_up_down_shift_pars
    gr_init = hill_pp(gr_pars, 1e-3)
    time_space = np.linspace(1e-3, 52, num=5000)

    plasmid_g_sst = plasmid_sst(growth_rate=gr_init, args=[4, 15])

    # plasmid_t = odeint(plasmid_dydt, plasmid_g_sst, time_space, args=(partial(hill_pp, gr_pars),))

    # phi_r_init = phi_ribosome(gr_init)
    phi_r_init = 0.09 * gr_init + 0.05

    constitutive_init = [expression_alpha_sst(gr_init)[0] / gr_init,
                         phi_r_init,
                         plasmid_g_sst]

    constitutive_expression = odeint(const_exp_dydt, constitutive_init, time_space,
                                     args=(partial(hill_pp, gr_pars),))

    # fig9, axs9 = plt.subplots(1, 1)
    # ax9 = axs9
    # ax9.plot(growth_range, sigma_vs_gr/sigma_vs_gr[ref_index])
    # ax9.plot(growth_range, plasmid_vs_gr/plasmid_vs_gr[ref_index])
    # ax9.plot(growth_range, trans_reg_vs_gr/trans_reg_vs_gr[ref_index])
    #
    # fig9.show()

    fig11, ax11 = plt.subplots(1, 1)
    plasmid_eval = constitutive_expression[:, ]

    ax11.plot(time_space, constitutive_expression[:, 0] / constitutive_expression[0, 0])
    ax11.plot(time_space, constitutive_expression[:, 1] / constitutive_expression[0, 1], '--')

    fig11.show()


    # %% Fitting the growth rate for the Up-Down shift exp. in 20210630

    ud_shift_data_ps = r'sub_model/growth_rate_down_up_shift_20210630.xlsx'

    pd_shift_data = pd.read_excel(ud_shift_data_ps)
    # clean data
    pd_shift_data = pd_shift_data[np.logical_and(~np.isnan(pd_shift_data.iloc[:, 1]),
                                                 ~np.isnan(pd_shift_data.iloc[:, 0]))]

    up_down_shift_pars = [-19.23669571, 3.033802, 2.0, 4.1,
                          1.60, 0.24, 0.68346174, 5.3086755,
                          0.8491872, 8.5e1, 32.5, 1.36716679]

    time_list = np.linspace(0, 35, num=1000)

    sim_gr = hill_pp(time=time_list, pars=up_down_shift_pars)

    fig1, ax1 = plt.subplots(1, 1, figsize=(25, 8))
    down_shift_index = time_list <= 31.2
    up_shift_index = time_list > 31.2
    ax1.scatter(pd_shift_data['Time'], pd_shift_data['Growth_rate'], c='#898686')
    ax1.plot(time_list[down_shift_index], sim_gr[down_shift_index], color='#5c5cfc', lw=10)

    ax1.plot(time_list[up_shift_index], sim_gr[up_shift_index], color='#ff8900', lw=10)
    aspect_ratio(1 / 3.5)
    ax1.set_xlim(0, 35)
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Growth rate ($h^{-1}$)')
    fig1.show()

    # %% This code used for empirical expression level  (Up-Down shift exp. in 20210630)


    gr_pars = up_down_shift_pars
    gr_init = hill_pp(gr_pars, 1e-3)
    time_space = np.linspace(1e-3, 40, num=5000)

    factor = 9
    kg = 0.4

    toggle_pars_1 = dict(k_t=factor, k_l=factor * kg, tau_p_ltet=0.002, tau_p_trc=0.035, n_l=2, n_t=4)
    toggle_pars_2 = dict(k_t=factor, k_l=factor * kg, tau_p_ltet=0.002, tau_p_trc=0.035 / 0.021 * 0.0084, n_l=2, n_t=4)
    toggle_pars_3 = dict(k_t=factor, k_l=factor * kg, tau_p_ltet=0.002, tau_p_trc=0.035 / 0.021 * 0.0042, n_l=2, n_t=4)
    toggle_pars_list = [toggle_pars_1, toggle_pars_2, toggle_pars_3]
    toggle_sim = []
    for toggle_pars in toggle_pars_list:
        # gene_alpha_green, gene_alpha_red = alpha_sst(growth_rate=gr_init)
        toggle = ToggleBasic(growth_rate=gr_init)
        assign_vars(toggle, toggle_pars)
        toggle.growth_rate = gr_init
        # toggle.set_alpha_trc(gene_alpha_red)
        # toggle.set_alpha_ltet(gene_alpha_green)
        toggle.solve_sst(optimize=True)
        print('Toggle Bistability:', toggle.bistable)
        green_state_index = np.argmax(toggle.sst_laci_conc)
        red_init = toggle.sst_tetr_conc[green_state_index]
        green_init = toggle.sst_laci_conc[green_state_index]

        phi_r_init = phi_ribosome(gr_init)

        toggle_emp_init = [green_init, red_init,
                           phi_r_init]

        toggle_sim.append(
            odeint(toggle_emperical_dydt, toggle_emp_init, time_space, args=(toggle, partial(hill_pp, gr_pars))))

    fig11, ax11 = plt.subplots(1, 3, figsize=(25, 8), sharey=True)
    for i, toggle_time in enumerate(toggle_sim):
        down_shift_index = time_space <= 31.2
        up_shift_index = time_space > 31.2

        ax11[i].plot(time_space, toggle_time[:, 0], 'g')
        ax11[i].plot(time_space, toggle_time[:, 1], 'r')

    fig11.show()

    # %%
    fig12, ax12 = plt.subplots(1, 3, figsize=(25, 8), sharey=True)
    for i, toggle_time in enumerate(toggle_sim):
        down_shift_index = time_space <= 31.2
        up_shift_index = time_space > 31.2

        ax12[i].plot(toggle_time[:, 1][down_shift_index],
                     toggle_time[:, 0][down_shift_index],
                     '.-', color='#5c5cfc', lw=10)
        ax12[i].plot(toggle_time[:, 1][up_shift_index],
                     toggle_time[:, 0][up_shift_index],
                     '.-', color='#ff8900', lw=10)
        ax12[i].set_xscale('log')
        ax12[i].set_yscale('log')
        ax12[i].set_xlim(1e-1, 3e3)
        ax12[i].set_ylim(1e-2, 1e3)
        ax12[i].set_xticks([], minor=True)
        ax12[i].set_yticks([], minor=True)
        ax12[i].set_yticks([1e-1, 1e1, 1e3])
        aspect_ratio(1, ax12[i])

    fig12.show()


