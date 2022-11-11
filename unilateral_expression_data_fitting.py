# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
"""

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import pandas as pd
import numpy as np  # Or any other
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import sciplot as splt
from typing import Union
splt.whitegrid()


# […]

# Own modules
def reponse_laci(pars, input, n_l=3.651):
    alpha, k_i, n_i, gamma, beta = pars
    return alpha / (1. + (gamma / (k_i ** n_i + input ** n_i)) ** n_l) + beta


def reponse_laci_tpars(pars, input, fixed_pars, n_l=3.651):
    alpha, gamma, beta = pars
    k_i, n_i = fixed_pars
    return alpha / (1. + (gamma / (k_i ** n_i + input ** n_i)) ** n_l) + beta


def laci_inhibit(pars, ip, op, func, fix_pars: dict):
    y_prime = func(pars, ip, **fix_pars)
    # return y_prime - op
    return np.log(np.abs(y_prime / op))


def inhibit_response(pars, ip, **kwargs):
    """
    pars: list; init of [y_min, y_max, n, k ]
    """
    y_min, y_max, n, k = pars
    return y_min + y_max * (1. / ((ip / k) ** n + 1.))


def inhibit_response_fixn(pars, ip, **kwargs):
    """
    pars: list; init of [y_min, y_max, k]
    """
    y_min, y_max, k = pars
    n = kwargs['n']
    return y_min + y_max * (1. / ((ip / k) ** n + 1.))


def inhibit_response_cgk(pars, ip, **kwargs):
    k = pars
    n = kwargs['n']
    y_min = kwargs['y_min']
    y_max = kwargs['y_max']
    return y_min + y_max * (1. / ((ip / k) ** n + 1.))


def inhibit_respnse_normalize(pars, ip, **kwargs):
    """
    pars: list; init of [n, k]
    """
    n, k = pars
    return 1. / ((ip / k) ** n + 1.)


def inhibit_respnse_normalize_fixn(pars, ip, **kwargs):
    k = pars
    n = kwargs['n']
    return 1. / ((ip / k) ** n + 1.)


def inhibit_response_nfix_curve_fit(ip, pars, n):
    k = pars
    return np.log(1. / ((ip / k) ** n + 1.))


def target_response_tpars(pars, input, output, fixedpars, n_l=3.651):
    out_fit = reponse_laci_tpars(pars, input, fixedpars, n_l)
    return output - out_fit


def target_response(pars, input, output, n_l=3.651):
    out_fit = reponse_laci(pars, input, n_l)
    return output - out_fit


def plasmid_copy_number(growth_rate, n=1., alpha=1.):
    return ((alpha / growth_rate) ** (1 / n) - 1.) * n


class ColE1Plasmid:
    def __init__(self, alpha: float = 24.12, tau: float = 24.12, k_p: float = 10., n: float = 0.7, k_1: float = 1.,
                 growth_rate: float = 1.):
        self._alpha = alpha
        self._tau = tau
        self._k_p = k_p
        self._k_1 = k_1
        self._n = n
        self._lambda = growth_rate
        self._g = None
        self._r1 = None
        self._r0 = None
        self._g_sst = None

    @property
    def r0(self):
        self._r0 = (1. + self._r1 / (self._n * self._k_1)) ** (-self._n)
        return self._r0

    @property
    def r1(self):
        return self._r1

    @r0.setter
    def r0(self, r0):
        self._r0 = r0

    def dev_g(self):
        dev_g = (self._k_p * self.r0 - self._lambda) * self._g
        return dev_g

    def dev_r1(self):
        dev_r1 = self._alpha * self._g - self._tau * self._r1
        return dev_r1

    def dev_plasmid(self, *variables):
        """
        Parameters
        ----------
        variables: list
            [growth rate, plasmid copy number, RI mRNA]

        Returns
        -------
        list of dydt:
            [d[plasmid]/dt, d[RI]/dt]
        """
        self._lambda, self._g, self._r1 = variables

        return [self.dev_g(), self.dev_r1()]

    @property
    def g_sst(self):
        self._g_sst = self._tau / self._alpha * (
                (self._k_p / self._lambda) ** (1. / self._n) - 1.) * self._n * self._k_1
        self._g = self._g_sst
        self._r1 = self._alpha * self._g / self._tau
        self._r0 = (1. + self._r1 / (self._n * self._k_p)) ** (-self._n)
        return self._g_sst

    def get_g_sst(self, growth_rate):
        return self._tau / self._alpha * ((self._k_p / growth_rate) ** (1. / self._n) - 1.) * self._n * self._k_1


def plasmid_cn_exp(growth_rate, a=10.):
    return np.log(a / growth_rate)


def plasmid_copy_number_hill(growth_rate, a=10., b=0.5):
    return 1 / np.log(1 + np.exp(a * (growth_rate - b)))


def protein_trans_rate(growth_rate, a=21.60534191, b=0.0536743, c=0.2116, d=0.08):
    phi_r = phi_ribosome(growth_rate, c, d)
    return phi_r * a / (phi_r + b)


def phi_ribosome(growth_rate, c=0.2116, d=0.08, sigma=2.21):
    """
    Note: r=RNA/Protein, r = sigma * phi_r, the defaults are obtained from Dai XF et al. 2017.
    """
    return (c * growth_rate + d) / sigma


def const_transcript(growth_rate, k=1.457, n=1., tau=0.2):
    # phi_r = phi_ribosome(growth_rate)
    return (1. - tau) / (1. + (k / growth_rate) ** n) + tau


def frc_act_ribo(growth_rate, a=1.01494589, b=0.15090343, n=1.35893622):
    return 1 / (a + (b / growth_rate) ** n)


def gene_expression(growth_rate: Union[np.ndarray, float] = 1.0,
                    rbs=100.,
                    plsmd_na=4.2, plasmd_gr_thr=0.6,
                    tp_ribo_t=0.4, tp_n=3., tp_tau=.24,
                    ):
    """
    Green lateral pars : dict(rbs=163305., plsmd_na=4.2, plasmd_gr_thr=0.6, tp_ribo_t=0.4, tp_n=3., tp_tau=.25,)
    Red lateral pars: dict(rbs=94545.0., plsmd_na=4.2, plasmd_gr_thr=0.6, tp_ribo_t=0.4, tp_n=3., tp_tau=.25,)
    """
    plmd_cpnb = plasmid_copy_number_hill(growth_rate, a=plsmd_na, b=plasmd_gr_thr)
    plasmid_ratio = plmd_cpnb / (plmd_cpnb + 1000)
    transcript_affinity = const_transcript(growth_rate, k=tp_ribo_t, n=tp_n, tau=tp_tau)
    active_ribo_ratio = frc_act_ribo(growth_rate)
    alph = protein_trans_rate(growth_rate)
    phi_r = phi_ribosome(growth_rate)
    exp_eff = plasmid_ratio * transcript_affinity * alph * active_ribo_ratio * phi_r * rbs
    return exp_eff


def gene_expression2(growth_rate: Union[np.ndarray, float] = 1.0,
                     rbs=2.87252353,
                     tp_ribo_t=0.25, tp_n=3.2, tp_tau=0.01,
                     plsmd_na=0.70, plasmd_alpha=10,
                     ):
    """
    Green lateral pars : dict( rbs=5.47, tp_ribo_t=0.25, tp_n=3.2, tp_tau=0.01, plsmd_na=0.70, plasmd_alpha=10,)
    Red lateral pars: dict( rbs=2.87, tp_ribo_t=0.25, tp_n=3.2, tp_tau=0.01, plsmd_na=0.70, plasmd_alpha=10,)
    """
    plmd_cpnb = plasmid_copy_number(growth_rate, n=plsmd_na, alpha=plasmd_alpha)
    plasmid_ratio = plmd_cpnb
    transcript_affinity = const_transcript(growth_rate, k=tp_ribo_t, n=tp_n, tau=tp_tau)
    active_ribo_ratio = frc_act_ribo(growth_rate)
    alph = protein_trans_rate(growth_rate)
    phi_r = phi_ribosome(growth_rate)
    exp_eff = plasmid_ratio * transcript_affinity * alph * active_ribo_ratio * phi_r * rbs
    return exp_eff


# def gene_expression(growth_rate: Union[np.ndarray, float] = 1.0,
#                     plsmd_na=3.6, plasmd_gr_thr=0.6,
#                     tp_ribo_t=0.06, tp_n=3.,
#                     rbs=100.
#                     ):
#     plmd_cpnb = plasmid_copy_number_hill(growth_rate, a=plsmd_na, b=plasmd_gr_thr)
#     plasmid_ratio = plmd_cpnb / (plmd_cpnb + 1000)
#     transcript_affinity = const_transcript(growth_rate, k=tp_ribo_t, n=tp_n)
#     active_ribo_ratio = frc_act_ribo(growth_rate)
#     alph = protein_trans_rate(growth_rate, b=0.05, a=20)
#     phi_r = phi_ribosome(growth_rate)
#     exp_eff = plasmid_ratio * transcript_affinity * alph * active_ribo_ratio * phi_r \
#               / growth_rate * rbs
#
#     return exp_eff

def opt_frac_active(args, x, y, func):
    return y - func(x, *args)


# %%
if __name__ == '__main__':
    # %%
    from scipy.optimize import leastsq

    trans_data = pd.read_excel(r'./data/translation_data_from_papers.xlsx', sheet_name=-1)
    gr, frac_ribo = trans_data['Grwoth_rate'], trans_data['Fraction_active_ribosome']
    elong_data = pd.read_excel(r'./data/translation_data_from_papers.xlsx', sheet_name=0)
    gr_elong, elong = elong_data['Growth_rate'], elong_data['Elongation_rate']
    r_data = pd.read_excel(r'./data/translation_data_from_papers.xlsx', sheet_name=1)
    gr_r, r = r_data['Growth_rate'], r_data['RNA/Protein']

    plasmid_data = pd.read_excel(r'./data/growth_rate_vs_plasmid_copy_number.xlsx', usecols=range(3))
    plasmid_exp = plasmid_data[plasmid_data['Sample'] == 'M4']['Copy_number']
    plasmid_gr_exp = plasmid_data[plasmid_data['Sample'] == 'M4']['Growth_rate']

    sigle_side = pd.read_excel(r'./data/single_side_response.xlsx')
    gr_single_g, flu_g = sigle_side[sigle_side['Sample'] == 'G']['Growth_rate'], \
                         sigle_side[sigle_side['Sample'] == 'G']['Expression_level']

    gr_single_r, flu_r = sigle_side[sigle_side['Sample'] == 'R']['Growth_rate'], \
                         sigle_side[sigle_side['Sample'] == 'R']['Expression_level']

    # frac_act_ribo_pars = leastsq(opt_frac_active, np.array([0.8, 0.18, 1.]), args=(gr, frac_ribo, frc_act_ribo))[0]
    #
    # elong_rate_pars = leastsq(opt_frac_active, np.array([10.2, 0.12]), args=(gr_elong, elong, protein_trans_rate))[0]

    # expression_pars = leastsq(opt_frac_active, np.array([3.6, 0.6, 0.06, 1.5, 300]),
    # args=(gr_single_g, flu_g, gene_expression))

    expression_pars = leastsq(opt_frac_active, np.array([1.27515069e+02]),
                              args=(gr_single_g, flu_g, gene_expression2))[0]

    # %% plasmid

    lambda_list = np.linspace(0.2, 1.8, num=200)
    dbltime = np.log(2) / lambda_list * 60
    # translation
    alph = protein_trans_rate(lambda_list)
    phi_r = phi_ribosome(lambda_list)
    # transcription
    # transcript_affinity = const_transcript(lambda_list, 1.51322440e+00, -6.70833640e+01,  6.74213344e-01)
    # transcript_affinity = const_transcript(lambda_list, 0.30, 2.5, .08)
    transcript_affinity = const_transcript(lambda_list, 0.25, 3.2, 0.01)

    # translation
    rbs_strength = 1.72158805

    # plasmid copy number
    # plmd_cpnb = plasmid_copy_number(lambda_list, n=3, alpha=100)
    # plmd_cpnb = plasmid_copy_number_hill(lambda_list, a=6, b=0.1)
    # plmd_cpnb = plasmid_copy_number_hill(lambda_list, a=3.6, b=0.6)
    # plmd_cpnb = plasmid_copy_number_hill(lambda_list, 4.2, 0.6)
    plmd_cpnb = plasmid_copy_number(lambda_list, 0.70, 10)

    plasmid_ratio = plmd_cpnb
    # plasmid_ratio /= plasmid_ratio.max()
    pnc_lambda = lambda_list * plmd_cpnb

    fig1, ax1 = plt.subplots(3, 4, figsize=(35, 24))
    ax = ax1.flatten()
    for axe in ax:
        axe.set_xlim(0, 2)
        # axe.set_ylim(ymin=0)
    ax[0].plot(lambda_list, plmd_cpnb)
    ax[0].scatter(plasmid_gr_exp, plasmid_exp)
    ax[0].set_xlabel('Growth rate ($h^{-1}$)')
    ax[0].set_ylabel('Plasmid CN')

    ax[1].plot(lambda_list, 1. / plmd_cpnb)
    ax[1].set_xlabel('Growth rate ($h^{-1}$)')
    ax[1].set_ylabel('1/[CN]')

    ax[2].plot(lambda_list, alph)
    ax[2].scatter(gr_elong, elong)
    ax[2].set_ylim(ymin=0)
    ax[2].set_xlabel('Growth rate ($h^{-1}$)')
    ax[2].set_ylabel('Translation efficiency')

    ax[3].plot(lambda_list, pnc_lambda)
    ax[3].set_xlabel('Growth rate ($h^{-1}$)')
    ax[3].set_ylim(ymin=0)
    ax[3].set_ylabel('$[\mathrm{CN}] \\cdot \lambda$')

    ax[4].plot(lambda_list, transcript_affinity)
    ax[4].set_ylabel('Transcription rate')

    trans_rate = plasmid_ratio * transcript_affinity
    norm_trans_rate = trans_rate / trans_rate.max()
    ax[5].plot(lambda_list, norm_trans_rate)
    ax[5].set_ylim(ymin=0)

    active_ribo_ratio = frc_act_ribo(lambda_list)
    ax[6].plot(lambda_list, active_ribo_ratio)
    ax[6].scatter(gr, frac_ribo)
    ax[6].set_ylabel('Active ribosome fraction')

    ax[7].plot(lambda_list, alph * active_ribo_ratio * phi_r)
    # ax[7].scatter(gr_r, r)

    # ax[8].plot(lambda_list, gene_expression(lambda_list, **dict(rbs=550., plsmd_na=4.2, plasmd_gr_thr=0.6,
    # tp_ribo_t=0.4, tp_n=3., tp_tau=.25, )))

    ax[9].plot(lambda_list,
               gene_expression2(lambda_list,
                                *expression_pars))
    ax[9].scatter(gr_single_g, flu_g)
    exp_eff = plasmid_ratio * transcript_affinity * alph * active_ribo_ratio * phi_r
    ax[-2].plot(lambda_list, exp_eff / exp_eff.max())
    ax[-2].set_ylabel('Expression efficiency')
    exp_level = exp_eff / lambda_list * rbs_strength
    ax[-1].plot(lambda_list, exp_level / exp_level.max())
    ax[-1].scatter(gr_single_r, flu_r / flu_r.max())
    ax[-1].set_ylabel('Expression level')

    fig1.show()

    # %%
    plasmid_data = pd.read_excel(r'./data/growth_rate_vs_plasmid_copy_number.xlsx', usecols=range(3))
    plasmid_exp = plasmid_data[plasmid_data['Sample'] == 'M5']['Copy_number']
    plasmid_gr_exp = plasmid_data[plasmid_data['Sample'] == 'M5']['Growth_rate']

    lambda_list = np.linspace(0.15, 1.8, num=500)

    rests = leastsq(opt_frac_active, np.array([1.48, 265.71]), args=(plasmid_gr_exp, plasmid_exp, plasmid_copy_number))[
        0]
    print(rests)
    plasmid_cp = plasmid_copy_number(lambda_list, *rests)
    fig3, ax3 = plt.subplots(2, 2, figsize=(16, 16))
    ax3 = ax3.flatten()
    ax3[0].plot(lambda_list, plasmid_cp)
    ax3[0].scatter(plasmid_gr_exp, plasmid_exp)
    ax3[1].plot(lambda_list, 1 / plasmid_cp)
    ax3[2].plot(lambda_list, plasmid_cp * lambda_list)
    fig3.show()

    # %% test plasmid
    from toggle_dynamic import gr_pars, hill_pp, assign_vars, ToggleBasic
    from scipy.integrate import odeint
    from functools import partial


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
        dev_green, dev_red = toggle_obj.field_flow(growth_rate)
        return [dev_green, dev_red, dev_green_exp_eff, dev_red_exp_eff, dev_g, dev_r1]


    gr_init = hill_pp(gr_pars, 1e-3)
    time_space = np.linspace(1e-3, 55, num=8000)

    factor = 0.98
    kg = 1
    toggle_pars = dict(k_t=4.14 * factor, k_l=5.71 * factor * kg, tau_p_ltet=0.0016, tau_p_trc=0.015, n_l=1.4)
    toggle = ToggleBasic(growth_rate=gr_init)
    assign_vars(toggle, toggle_pars)
    toggle.solve_sst(optimize=True)
    plasmid = ColE1Plasmid(growth_rate=gr_init, n=0.7)

    _ = plasmid.g_sst
    vars_init = [plasmid.g_sst, plasmid.r1]

    plasmid_t = odeint(plasmid_dydt, vars_init, time_space, args=(plasmid, partial(hill_pp, gr_pars)))

    exp_eff_init = [gene_expression2(gr_init), *vars_init]

    exp_eff_t = odeint(exp_eff_dydt, exp_eff_init, time_space, args=(plasmid, partial(hill_pp, gr_pars)))

    fig4, ax4 = plt.subplots(2, 2, figsize=(18, 16))
    ax = ax4.flatten()
    ax[0].plot(time_space, exp_eff_t[:, 1])
    ax[0].plot(time_space, plasmid_t[:, 0])
    ax[0].set_xlabel('Time (h)')
    ax[0].set_ylabel('Plasmid copy number')

    ax[1].plot(hill_pp(gr_pars, time_space), exp_eff_t[:, 1])
    ax[1].plot(np.linspace(0.1, 1.8), plasmid.get_g_sst(np.linspace(0.1, 1.8)), '--')
    ax[1].set_xlabel('Growth rate')
    ax[1].set_ylabel('Plasmid copy number')

    ax[2].plot(time_space, exp_eff_t[:, 0] / exp_eff_t[:, 0].max())
    ax[2].plot(time_space, exp_eff_t[:, 1] / exp_eff_t[:, 1].max(), '--')

    ax[2].set_xlabel('Time (h)')
    ax[2].set_ylabel('expression efficiency')

    ax[3].plot(hill_pp(gr_pars, time_space), exp_eff_t[:, 0])

    ax[3].plot(np.linspace(0.1, 1.8), gene_expression2(np.linspace(0.1, 1.8)), '--')
    ax[3].set_xlabel('Growth rate')
    ax[3].set_ylabel('expression efficiency')

    fig4.show()
