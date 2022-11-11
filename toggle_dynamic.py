# -*- coding: utf-8 -*-

"""
This file is the main script describing the toggle switch.
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
import matplotlib.tri as tri
from scipy.integrate import odeint
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.optimize import fsolve
from matplotlib.colors import ListedColormap
import sciplot as splt
from typing import List, Union, Optional
from unilateral_expression_data_fitting import gene_expression, gene_expression2
from sub_model.gene_expression_rate import expression_level
from sub_model.emperical_gene_expression_rate import alpha_sst

splt.whitegrid()

# fitting = ['empirical mRNA']
fitting = ['zjw_fitting']


def trans_level10(lambda_, pars):
    a, gamma, r0, nu, beta, K = pars
    return a * (gamma * lambda_ + r0) / (K + gamma * lambda_ + r0) * (nu - beta * lambda_)


def trasns_fitting(lambda_, pars):
    a, b, c, d, e = pars
    return lambda_ * (a + b / (c + (lambda_ / d) ** e))


if fitting[0] == 'expression_sim':
    alpha_fuc = gene_expression2
    tetR_pars = dict(rbs=2.87, tp_ribo_t=0.25, tp_n=3.2, tp_tau=0.01, plsmd_na=0.70, plasmd_alpha=10, )
    lacI_pars = dict(rbs=5.47, tp_ribo_t=0.25, tp_n=3.2, tp_tau=0.01, plsmd_na=0.70, plasmd_alpha=10, )
elif fitting[0] == 'hill_function':
    alpha_fuc = trans_level10
    tetR_pars = dict(pars=[3.17063752e+03, 1 / 10.2, 0.02, 3.36377491e-01, 1.97091207e-01, 0.45])
    lacI_pars = dict(pars=[1.03338361e+04, 1 / 10.2, 0.02, 2.38209519e-01, 1.37637354e-01, 0.45])
elif fitting[0] == 'gene_expression_rate_model':
    alpha_fuc = expression_level
    tetR_pars = dict(pars=[0.007, 5, 1.3E-1])
    lacI_pars = dict(pars=[0.03, 6., 1E-1])
elif fitting[0] == 'empirical mRNA':
    alpha_fuc = alpha_sst
    tetR_pars = dict(pars='tetr')
    lacI_pars = dict(pars='laci')
elif fitting[0] == 'zjw_fitting':
    alpha_fuc = trasns_fitting
    tetR_pars = dict(pars=[26.836, 320.215, 1.0, 0.661, 4.09])
    lacI_pars = dict(pars=[25.609, 627.747, 1.0, 0.865, 4.635])  # lacI_pars = dict(pars=[16.609, 627.747, 1.0, 0.865, 4.635])


class ToggleBasic:
    def __init__(self, growth_rate=1.0):
        self.k_t = 1.  # type: float # TetR Kd
        self.k_l = 2.  # type: float # LacI Kd
        self.n_t = 2.0  # type: float # TetR binding coefficience
        self.n_l = 4.0  # type: float # LacI binding coefficience
        self.tau_p_trc = 0.035  # type: float # Ptrc leakage, TetR leakage expression level
        self.tau_p_ltet = 0.002  # PLtetO-1 leakage, LacI leakage # type: float
        self.alphal_factor = 1.
        self.alphat_factor = 1.
        self.gr = growth_rate  # type: float # cell growth rate
        self.alpha_trc = self.alphal_factor * alpha_fuc(self.gr, **tetR_pars)  # type: float
        # Ptrc, TetR expression rate
        self.alpha_ltet = self.alphat_factor * alpha_fuc(self.gr, **lacI_pars)  # type: float
        # PLtetO-1, LacI expression rate
        self.protein_decay = 0.
        self.alpha_trc_over_gr = self.alpha_trc / (self.gr + self.protein_decay)  # type: float # TetR expression level
        self.alpha_ltet_over_gr = self.alpha_ltet / (
                self.gr + self.protein_decay)  # type: float # LacI expression level
        self.atc_conc = 0.
        self.iptg_conc = 0.
        self.k_atc = 1.
        self.k_iptg = 1.
        self.m = 1.
        self.n = 1.
        self.sst_laci_conc = None
        self.sst_tetr_conc = None
        self.sst_state = None  # type: Optional[List] # values < 0 are steady state, otherwise are unstable state
        self.bistable = False  # type: bool
        self.tilde_k_t = self.k_t / self.alpha_trc_over_gr  # type: float
        self.tilde_k_l = self.k_l / self.alpha_ltet_over_gr  # type: float

    @property
    def growth_rate(self):
        return self.gr

    @growth_rate.setter
    def growth_rate(self, growth_rate):
        self.gr = growth_rate
        # Ptrc, TetR expression rate
        self.alpha_trc = self.alphal_factor * alpha_fuc(self.gr, **tetR_pars)  # type: float
        # PLtetO-1, LacI expression rate
        self.alpha_ltet = self.alphat_factor * alpha_fuc(self.gr, **lacI_pars)  # type: float
        self.alpha_trc_over_gr = self.alpha_trc / (self.gr + self.protein_decay)
        self.alpha_ltet_over_gr = self.alpha_ltet / (self.gr + self.protein_decay)
        self.tilde_k_t = self.k_t / self.alpha_trc_over_gr  # type: float
        self.tilde_k_l = self.k_l / self.alpha_ltet_over_gr  # type: float

    def set_k_l(self, k_l: float) -> None:
        self.k_l = k_l
        self.tilde_k_l = self.k_l / self.alpha_ltet_over_gr

    def set_k_t(self, k_t: float) -> None:
        self.k_t = k_t
        self.tilde_k_t = self.k_t / self.alpha_trc_over_gr

    def set_alpha_trc(self, alpha_trc: float) -> None:
        self.alpha_trc = self.alphal_factor * alpha_trc
        self.alpha_trc_over_gr = self.alpha_trc / (self.gr + self.protein_decay)

    def set_alpha_ltet(self, alpha_ltet: float) -> None:
        self.alpha_ltet = self.alphat_factor * alpha_ltet
        self.alpha_ltet_over_gr = self.alpha_ltet / (self.gr + self.protein_decay)

    def h_l(self, laci):
        """
        Hill function of LacI (Ptrc)
        """
        return self.tau_p_trc + (1. - self.tau_p_trc) / (1. + (laci / self.k_l) ** self.n_l)

    def h_t(self, tetr):
        """
        Hill function of TetR (PLtetO-1)
        """
        return self.tau_p_ltet + (1. - self.tau_p_ltet) / (1. + (tetr / self.k_t) ** self.n_t)

    def null_cline_tetr(self, laci_tot):
        laci_free = laci_tot * (1. + self.iptg_conc / self.k_iptg * laci_tot) ** -self.n
        return self.alpha_trc * self.h_l(laci_free) / (self.gr + self.protein_decay)

    def null_cline_laci(self, tetr_tot):
        tetr = tetr_tot * (1. + self.atc_conc / self.k_atc * tetr_tot) ** -self.m
        return self.alpha_ltet * self.h_t(tetr) / (self.gr + self.protein_decay)

    def field_flow(self, laci_tetr: List) -> List:
        """
        calculate the field flow of the toggle in given concentrations of LacI and TetR

        Parameters
        ---------------
        laci_tetr : array like
            a list given the concentrations of LacI and TetR.

        Return
        ------------
        list of dydt : list
            [d[LacI]/dt, d[TetR]/dt]
        """
        laci_tot, tetr_tot = laci_tetr
        tetr = tetr_tot * (1. + self.atc_conc / self.k_atc * tetr_tot) ** -self.m
        laci = laci_tot * (1. + self.iptg_conc / self.k_iptg * laci_tot) ** -self.n
        dev_laci = self.alpha_ltet * self.h_t(tetr) - laci_tot * (self.gr + self.protein_decay)
        dev_tetr = self.alpha_trc * self.h_l(laci) - tetr_tot * (self.gr + self.protein_decay)
        return [dev_laci, dev_tetr]

    def dev_laci(self, laci):
        free_laci = laci * (1. + self.iptg_conc / self.k_iptg * laci) ** -self.n
        sst_tetr = self.null_cline_tetr(free_laci)
        return self.alpha_ltet * self.h_t(sst_tetr) - laci * (self.gr + self.protein_decay)

    def dev_laci_timenormalized(self, laci):
        free_laci = laci * (1. + self.iptg_conc / self.k_iptg * laci) ** -self.n
        sst_tetr = self.null_cline_tetr(free_laci)
        return self.alpha_ltet_over_gr * self.h_t(sst_tetr) - laci

    def dev_tetr(self, tetr):
        free_tetr = tetr * (1. + self.atc_conc / self.k_atc * tetr) ** -self.m
        sst_laci = self.null_cline_laci(free_tetr)
        return self.alpha_trc * self.h_l(sst_laci) - tetr * (self.gr + self.protein_decay)

    def dev_tetr_timenormalized(self, tetr) -> Union[np.ndarray, float, List]:
        free_tetr = tetr * (1. + self.atc_conc / self.k_atc * tetr) ** -self.m
        sst_laci = self.null_cline_laci(free_tetr)
        return self.alpha_trc_over_gr * self.h_l(sst_laci) - tetr

    def potential_laci(self, u, laci) -> Union[np.ndarray, float, List]:
        return - self.dev_laci_timenormalized(laci)

    def potential_tetr(self, u, tetr) -> Union[np.ndarray, float, List]:
        return - self.dev_tetr_timenormalized(tetr)

    def calu_tetr_potential(self, tetr_conc_list: Union[np.ndarray, List], init: float = 0.):
        tetr_potential = odeint(self.potential_tetr, init, tetr_conc_list)  # type: np.ndarray
        return tetr_potential.flatten()

    def calu_laci_potential(self, laci_conc_list, init=0.):
        laci_pot = odeint(self.potential_laci, init, laci_conc_list)  # type: np.ndarray
        return laci_pot.flatten()

    def solve_sst(self, laci_conc_list: np.ndarray = np.arange(0, 500, 0.1), optimize: bool = False):
        """
        Class method for solving the fix points of the toggle.

        """
        sign_dev_laci = np.sign(self.dev_laci(laci_conc_list))
        root = np.diff(sign_dev_laci)
        sst_index = np.nonzero(root)[0]
        self.sst_tetr_conc = self.null_cline_tetr(laci_conc_list[sst_index])
        self.sst_laci_conc = self.null_cline_laci(self.sst_tetr_conc)
        if optimize is True:
            sst_tetr = []
            sst_laci = []
            for i in range(len(self.sst_tetr_conc)):
                sst_laci_tetr = fsolve(self.field_flow, np.array([self.sst_laci_conc[i], self.sst_tetr_conc[i]]))
                sst_laci.append(sst_laci_tetr[0])
                sst_tetr.append(sst_laci_tetr[1])
            self.sst_laci_conc = np.array(sst_laci)
            self.sst_tetr_conc = np.array(sst_tetr)
        self.sst_state = root[sst_index]
        if len(sst_index) == 3:
            self.bistable = True
        else:
            self.bistable = False

    def quasipotential(self, t_end, laci_tetr, num=1000):
        def dev_potential(pars, t):
            dev_laci_tetr = self.field_flow(pars[:2])
            dp = np.sqrt(np.sum(np.array(dev_laci_tetr) ** 2))
            return [dev_laci_tetr[0],
                    dev_laci_tetr[1],
                    dp]

        # p0 = np.sqrt(np.sum(np.array(self.fild_flow(laci_tetr)) ** 2))
        pars0 = laci_tetr + [0.]
        t_list = np.linspace(0, t_end, num=num)
        laci_tetr_p_t = odeint(dev_potential, pars0, t_list)
        return laci_tetr_p_t


def titrate_alphovgr(pars: list, laci_range: np.ndarray = np.linspace(0, 20),
                     tetr_range: np.ndarray = np.linspace(0, 20), **kwargs):
    """
    return two null-clines when titrate the alpha_t/growth_rate and alpha_l/growth_rate

    Parameters
    -----------
    pars : List[float, float]
        a list including alpha_t/growth_rate and alpha_l/growth_rate.
    laci_range : np.ndarray
        an ndarray list, constrain the LacI concentration.
    tetr_range : np.ndarray

    """
    toggle = ToggleBasic()
    toggle.alpha_ltet_over_gr, toggle.alpha_trc_over_gr = pars
    assign_vars(toggle, kwargs)
    reslts = [toggle.null_cline_laci(tetr_range), toggle.null_cline_tetr(laci_range)]
    return reslts


def cal_quasipotential(t_end, laci_tetr, **kwargs):
    tg = ToggleBasic()
    assign_vars(tg, kwargs)
    resluts = tg.quasipotential(t_end, laci_tetr)
    return resluts[-1:, 2]


def cal_laci_poetntial(laci_list: np.ndarray = np.linspace(0, 1000, num=1000), growth_rate: float = 1.0, **kwargs) \
        -> np.ndarray:
    """
    Calculate the 1D potential landscape of LacI.

    Parameters
    -----------
    laci_list : np.ndarray
        LacI conc list defining the integration path.

    Returns
    -------
    np.ndarray
        An array of potential of the toggle switch

    """
    toggle_inst = ToggleBasic(growth_rate=growth_rate)
    assign_vars(toggle_inst, kwargs)
    reslut = toggle_inst.calu_laci_potential(laci_list)
    return reslut


def bistability(pars, **kwargs):
    toggle = ToggleBasic()
    assign_vars(toggle, kwargs)
    toggle.alpha_trc_over_gr, toggle.alpha_ltet_over_gr = pars
    toggle.solve_sst()
    return toggle.bistable


def bistability_titrate_k(pars, **kwargs):
    toggle = ToggleBasic()
    assign_vars(toggle, kwargs)
    al_ricip_kt, al_ricip_kl = pars
    toggle.set_k_t(1. / al_ricip_kt)
    toggle.set_k_l(1. / al_ricip_kl)
    toggle.solve_sst(np.linspace(0, 500, num=1000))
    return toggle.bistable


def steady_state(gr, **kwargs) -> List:
    """ Create a toggle instance, return its TetR conc., LacI conc., and bistable states.

    Parameters
    ----------
    gr : float
        growth rate
    **kwargs : dict
        attributes of ToggleBasic

    Returns
    --------
    list
        containing steady-state concentrations of TetR, LacI and its bistability.
        [toggle.sst_tetr_conc, toggle.sst_laci_conc, toggle.bistable]
    """
    toggle = ToggleBasic()
    assign_vars(toggle, kwargs)
    toggle.growth_rate = gr
    toggle.solve_sst(optimize=True)
    return [toggle.sst_tetr_conc, toggle.sst_laci_conc, toggle.bistable]


def assign_vars(obj, keys):
    obj_vars = list(obj.__dict__.keys())
    for var in obj_vars:
        if var in keys:
            obj.__dict__[var] = keys[var]


# parameters in thesis
# gr_pars = [-19.23669571,   2.6033802 , 2.10852488,   4 ,
#          1.63754155,   0.19,   0.68346174,   5.3086755 ,
#          0.8491872, 98.12303837,  45.42347936,   1.44]

gr_pars = [-19.23669571, 3.033802, 2.10852488, 4,
           1.63754155, 0.16819914, 0.68346174, 5.3086755,
           0.8491872, 98.19158635, 45.42363766, 1.36716679]


def hill_pp(pars, time):
    n1, k1, n2, k2, g1, g2, a, b, c, n3, k3, g3 = pars
    rate1 = (g1 * (time / k1) ** n1) / (1 + (time / k1) ** n1)
    rate2 = (g2 * (time / k2) ** n2) / (1 + (time / k2) ** n2)
    rate3 = a * np.exp(-(time - b) ** 2 / (2 * c ** 2))
    rate4 = (g3 * (time / k3) ** n3) / (1 + (time / k3) ** n3)
    return rate1 + rate2 + rate3 + rate4


def toggle_dynamic(lac_tetr_init=None, time_max=60, step=0.01, **kwargs):
    tg = ToggleBasic(gr=hill_pp(gr_pars, 0.001))
    assign_vars(tg, kwargs)
    tg.solve_sst()
    if lac_tetr_init is None:
        fold = tg.sst_laci_conc / tg.sst_tetr_conc
        index = np.where(fold == fold.max())[0][0]
        print(fold, index)
        lac_tetr_init = [tg.sst_laci_conc[index], tg.sst_tetr_conc[index]]

    def dev_tetr_laci(y, t):
        gr = hill_pp(gr_pars, t)
        tg.growth_rate = gr
        return tg.field_flow(y)

    resl = odeint(dev_tetr_laci, y0=lac_tetr_init, t=np.arange(0.001, time_max, step=step))
    return resl


# %%
if __name__ == '__main__':
    # %% verification code
    toggle_pars = dict(k_t=1.2, k_l=1.6, tau_p_ltet=0.0015, tau_p_trc=0.005, n_l=1.4)
    growth_rate = 1.8
    toggle_inst = ToggleBasic(growth_rate=growth_rate)
    assign_vars(toggle_inst, toggle_pars)
    toggle_inst.solve_sst()
    laci_conc_list = np.linspace(0, 125, num=1000)
    dGdt = toggle_inst.dev_laci(laci_conc_list)
    tetr_sst_conc, laci_sst_conc, _ = steady_state(growth_rate, **toggle_pars)
    laci_optential = cal_laci_poetntial(laci_conc_list, growth_rate=growth_rate, **toggle_pars)
    index_sst_lacI = np.argmin(np.abs(laci_conc_list.reshape(1, -1) - laci_sst_conc.reshape(-1, 1)), axis=1)

    fig1, ax1 = plt.subplots(2, 1)
    ax1 = ax1.flatten()
    ax1[0].plot(laci_conc_list, dGdt)
    ax1[0].scatter(laci_sst_conc, np.zeros(laci_sst_conc.shape))
    ax1[0].hlines(y=0, xmin=laci_conc_list.min(), xmax=laci_conc_list.max())
    ax1[0].set_ylim(-5)
    ax1[0].set_ylabel('$\mathrm{\\frac{d[LacI]}{dt}}$')
    ax1[1].plot(laci_conc_list, laci_optential)
    ax1[1].scatter(laci_conc_list[index_sst_lacI], laci_optential[index_sst_lacI])
    ax1[1].set_ylabel('U')
    ax1[1].set_xlabel('LacI conc.')

    fig1.show()

    # %%
    # plot the steady states along the growth rate, bifurcation plot
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

    factor = 9
    kg = 1.4
    toggle_pars = dict(k_t=1. * factor, k_l=1. * factor * kg, tau_p_ltet=0.015, tau_p_trc=0.13, n_l=4.0, n_t=2.0,
                       alphal_factor=1.1, alphat_factor=1.1)

    ret = Parallel(n_jobs=-1)(delayed(steady_state)(growth_rate[i], **toggle_pars)
                              for i in tqdm(range(len(growth_rate))))

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

    ax2.scatter(phase_portrait['TetR'], phase_portrait['LacI'])
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(False)
    # ax2.set_xlim(.1, 1000)
    # ax2.set_ylim(10, 1000)
    ax2.set_xlabel('TetR Conc.')
    ax2.set_ylabel('LacI Conc.')
    fig2.show()
    # fig2.savefig(r'./data/fig_sim_toggle_sst.svg')
    # %%
    laci_list_sim = np.linspace(0, 250, num=500)
    growth_rate_list = [1.59, 1.32, 1.15, 0.97, 0.68, 0.39, 0.19]
    potential_laci = Parallel(n_jobs=-1)(delayed(cal_laci_poetntial)(laci_list=laci_list_sim,
                                                                     growth_rate=gr, **toggle_pars)
                                         for gr in tqdm(growth_rate_list))
    ret = Parallel(n_jobs=-1)(delayed(steady_state)(gr, **toggle_pars)
                              for gr in tqdm(growth_rate_list))
    red_state_tetr = np.array([sst[0][0] for sst in ret])
    red_state_laci = np.array([sst[1][0] for sst in ret])

    green_state_tetr = np.array([sst[0][-1] for sst in ret])
    green_state_laci = np.array([sst[1][-1] for sst in ret])
    fig6, ax6 = plt.subplots(1, 1, figsize=(13, 11))
    for index_gr, laci_u in enumerate(potential_laci):
        a = ax6.plot(laci_list_sim, laci_u, label='%2.2f $h^{-1}$' % growth_rate_list[index_gr])
        greenlacisst = green_state_laci[index_gr]
        green_y_indice = np.argmin(abs(laci_list_sim - greenlacisst))
        redlacisst = red_state_laci[index_gr]
        red_y_indice = np.argmin(abs(laci_list_sim - redlacisst))
        ax6.scatter(laci_list_sim[green_y_indice], laci_u[green_y_indice], marker='o', color=a[0].get_color())
        ax6.scatter(laci_list_sim[red_y_indice], laci_u[red_y_indice], marker='v', color=a[0].get_color())
    ax6.set_xlim(1e-1, 200)
    ax6.set_ylim(-2500, 2000)
    ax6.legend()
    ax6.set_xlabel('LacI conc.')
    ax6.grid(False)
    ax6.set_ylabel('U')
    # ax6.set_xscale('log')
    fig6.show()
    fig6.savefig('./data/fig_laci_potential.svg')
    # %%
    fig3, ax3 = plt.subplots(1, 1)

    ax3.scatter(growth_rate[sim_index], green_state_laci[sim_index])
    ax3.scatter(growth_rate[sim_index], red_state_laci[sim_index])
    ax3.scatter(growth_rate[sim_index], green_unstable[sim_index])
    ax3.set_yscale('log')
    ax3.set_ylim(1e-1, 150)
    ax3.set_xlabel('$\lambda$')
    ax3.set_ylabel('$LacI$')
    fig3.show()

    fig4, ax4 = plt.subplots(1, 1)
    ax4.scatter(growth_rate[sim_index], green_state_tetr[sim_index])
    ax4.scatter(growth_rate[sim_index], red_state_tetr[sim_index])
    ax4.scatter(growth_rate[sim_index], red_unstable[sim_index])
    ax4.set_yscale('log')
    ax4.set_xlabel('$\lambda$')
    ax4.set_ylabel('$TetR$')
    ax4.set_ylim(1e-1, 1e3)
    fig4.show()

    fig5, ax5 = plt.subplots(1, 1)
    ax5.scatter(growth_rate[sim_index], (green_unstable / (red_unstable + green_unstable))[sim_index],
                color='y')
    ax5.scatter(growth_rate[sim_index], (red_state_laci / (red_state_laci + red_state_tetr))[sim_index],
                color='r')
    ax5.scatter(growth_rate[sim_index], (green_state_laci / (green_state_laci + green_state_tetr))[sim_index],
                color='g')

    ax5.scatter(green_ratio_gr, green_ratio_exp,
                marker='o', facecolors='none', edgecolors='r', lw=4)
    ax5.set_yscale('linear')
    ax5.set_xlabel('$\lambda$')
    ax5.set_ylabel('green ratio')
    ax5.set_ylim(0, 1)
    ax5.grid(False)
    fig5.show()
    # %% phase diagram mapped with experimental data
    data_expression = pd.read_csv(r'./data/single_arm_lacI_TetR_expression_level.csv')
    al_t = np.linspace(1, 700, num=200)
    al_l = np.linspace(1, 700, num=200)
    extent = [al_l[0], al_l[-1], al_t[0], al_t[-1]]
    al_l_list, al_t_list = np.meshgrid(al_l, al_t)

    sst_state = Parallel(n_jobs=64)(delayed(bistability)([l, t])
                                    for t, l in tqdm(zip(al_t_list.flatten(), al_l_list.flatten())))

    sst_state = np.array(sst_state).reshape(al_l_list.shape)

    fig1, ax1 = plt.subplots(1, 1)
    ct = ax1.contour(al_l, al_t, sst_state, linewidths=4, cmap='Set3')
    ax1.imshow(sst_state, extent=extent, origin='lower', cmap='coolwarm', alpha=0.3)
    ax1.grid(False)
    # ax1.scatter(data_expression['TetR'], data_expression['LacI'], s=200, marker='>', color='#EF6C00')
    ax1.set_xlabel('TetR')
    ax1.set_ylabel('LacI')
    fig1.show()
    fig1.savefig(r'./data/states_bifurcation.png', transparent=True)

    # %% only diagram phase

    al_t = np.linspace(0.1, 2000, num=200)
    al_l = np.linspace(0.1, 2000, num=200)
    extent = [al_l[0], al_l[-1], al_t[0], al_t[-1]]
    al_l_list, al_t_list = np.meshgrid(al_l, al_t)

    pars_toggle = dict(k_t=2.3, k_l=7, tau_p_t=0.002, tau_p_l=0.04)
    sst_state = Parallel(n_jobs=64)(delayed(bistability)([l, t], **pars_toggle)
                                    for t, l in tqdm(zip(al_t_list.flatten(), al_l_list.flatten())))
    print(np.sum(sst_state))
    sst_state = np.array(sst_state).reshape(al_l_list.shape)

    fig1, ax1 = plt.subplots(1, 1)
    ct = ax1.contour(al_l, al_t, sst_state, linewidths=4, cmap='Set3')
    ax1.imshow(sst_state, extent=extent, origin='lower', cmap='coolwarm', alpha=0.3)
    ax1.grid(False)
    ax1.set_xlabel('TetR')
    ax1.set_ylabel('LacI')
    ax1.set_title(f'''n:{pars_toggle.get('n_t')}; tau: {pars_toggle.get('tau_p_ltet')}''')
    fig1.show()

    # %%
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

    # surface_data = pd.DataFrame(data=dict(x=lac_mesh.flatten(),
    #                                       y=growth_mesh.flatten(),
    #                                       z=potential_laci_space[:, :80].flatten()))
    # surface_data.to_csv(r'./data/potential_surface.csv')
    # import scipy.io

    # %%
    # create a new color map
    new_coolwarm = cm.get_cmap('coolwarm')
    new_coolwarm = new_coolwarm(np.linspace(0, 1, 256))[60:-60]
    new_coolwarm = ListedColormap(new_coolwarm)
    vmax = 1385
    cut_potential_laci_space = potential_laci_space.copy()
    mask = cut_potential_laci_space <= vmax
    nan_cut_potential_laci_space = cut_potential_laci_space.copy()
    nan_cut_potential_laci_space[nan_cut_potential_laci_space > vmax] = np.nan
    max_potential_along_y = np.nanmax(nan_cut_potential_laci_space, axis=1)
    axis_max_laci_along_y = np.argmax(nan_cut_potential_laci_space, axis=1)
    max_laci_index = axis_max_laci_along_y.max()
    mask2 = lac_mesh < lac_mesh[0, max_laci_index]

    # plot 3d surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(24, 12))
    # surface = ax.plot_surface(lac_mesh[mask2].reshape(len(growth_rate_sim), -1), growth_mesh[mask2].reshape(len(growth_rate_sim), -1),
    #                           cut_potential_laci_space[mask2].reshape(len(growth_rate_sim), -1),
    #                           rcount=100, ccount=100, alpha=0.8,
    #                           antialiased=False, cmap=new_coolwarm,
    #                           norm=colors.PowerNorm(vmin=0, vmax=vmax, gamma=18))
    # ===================== Delaunay triangle for surface ======================
    triang = tri.Triangulation(lac_mesh[mask], growth_mesh[mask])
    laci_trig = lac_mesh[mask][triang.triangles].mean(axis=1)
    gr_trig = growth_mesh[mask][triang.triangles].mean(axis=1)
    index_gr_trig = np.argmin(np.abs(gr_trig.reshape(-1, 1) - growth_rate_sim.reshape(1, -1)), axis=1)
    tri_mask = laci_trig > lac_mesh[0, :][axis_max_laci_along_y][index_gr_trig]  # unwanted triangle
    triang.set_mask(tri_mask)  # remove unwanted triangles
    surface = ax.plot_trisurf(triang, cut_potential_laci_space[mask],
                              linewidth=0, antialiased=True, cmap=new_coolwarm,
                              norm=colors.PowerNorm(vmin=0, vmax=vmax, gamma=30))
    # surface = ax.plot_trisurf(lac_mesh[mask2], growth_mesh[mask2],
    #                           cut_potential_laci_space[mask2],
    #                           linewidth=0, antialiased=False, cmap=new_coolwarm, alpha=0.5,
    #                           norm=colors.PowerNorm(vmin=0, vmax=vmax, gamma=18))
    # ========================= delaunay triangles =======================
    cb = fig.colorbar(surface, shrink=0.5, aspect=5, pad=0.1, ticks=[800, 1300, 1360, 1380, 1400])
    cb.set_label('$U$')
    ax.plot_wireframe(lac_mesh, growth_mesh, nan_cut_potential_laci_space,
                      rstride=80, cstride=500, alpha=0.8, ls='--', lw=2, color='k')

    ax.plot(green_state_laci[sim_index], growth_rate[sim_index], green_potential,
            color='#3CE74C', ls=':')
    ax.plot(green_state_laci[sim_index][bistable_mask[sim_index]], growth_rate[sim_index][bistable_mask[sim_index]],
            green_potential[bistable_mask[sim_index]], color='#3CE74C', ls='-')
    ax.plot(red_state_laci[sim_index], growth_rate[sim_index], red_potential,
            color='#E74C3C', ls='-')
    ax.plot(red_state_laci[sim_index][bistable_mask[sim_index]], growth_rate[sim_index][bistable_mask[sim_index]],
            red_potential[bistable_mask[sim_index]], color='#E74C3C', ls='-')

    # ax.plot(lac_mesh[0, :][axis_max_laci_along_y], growth_mesh[:, 0], max_potential_along_y,
    #         color='k', ls='-')

    color_norm = colors.PowerNorm(vmin=0, vmax=vmax, gamma=30)
    shape = lac_mesh[mask2].reshape(len(growth_rate_sim), -1).shape
    ax.plot_surface(lac_mesh[mask2].reshape(len(growth_rate_sim), -1),
                    growth_mesh[mask2].reshape(len(growth_rate_sim), -1),
                    np.ones(shape=shape) * -1, zorder=1,
                    facecolors=new_coolwarm(color_norm(cut_potential_laci_space[mask2]).reshape(shape)))
    contours = ax.contour(lac_mesh[mask2].reshape(len(growth_rate_sim), -1),
                          growth_mesh[mask2].reshape(len(growth_rate_sim), -1),
                          cut_potential_laci_space[mask2].reshape(len(growth_rate_sim), -1),
                          zdir='z', offset=1, levels=[400, 800, 1350, 1400],
                          colors='k', linestyles=':', zorder=5)
    linwidth = 8
    ax.plot(green_state_laci[sim_index], growth_rate[sim_index], 0,
            color='#008000', ls=':', lw=linwidth, zorder=6)
    ax.plot(green_state_laci[sim_index][bistable_mask[sim_index]], growth_rate[sim_index][bistable_mask[sim_index]],
            0, color='#008000', ls='-', lw=linwidth, zorder=7)
    ax.plot(red_state_laci[sim_index], growth_rate[sim_index],
            0, color='#FF6000', ls='-', lw=linwidth, zorder=8)
    ax.plot(red_state_laci[sim_index][bistable_mask[sim_index]], growth_rate[sim_index][bistable_mask[sim_index]],
            0, color='#FF6000', ls='-', lw=linwidth, zorder=9)
    ax.tick_params(axis='z', pad=25)
    ax.tick_params(axis='y', pad=25)
    ax.set_xlabel('LacI conc. (a.u.)', labelpad=25, fontsize=35)
    ax.set_ylabel('Growth rate ($h^{-1}$)', labelpad=50, fontsize=35)
    ax.set_zlabel('U (a.u.)', labelpad=50)
    ax.view_init(30, -80)

    ax.set_xlim(0, 120)
    ax.set_zlim(-1, 1600)

    fig.show()
    # fig.savefig(r'./data/3d_potential_laci.png', transparent=True)

    # scio.savemat(r'./data/potential_data.mat', dict(lac_mesh=lac_mesh,
    #                                                 growth_mesh=growth_mesh,
    #                                                 mask2=mask2,
    #                                                 mask=mask,
    #                                                 green_state_laci=green_state_laci,
    #                                                 red_state_laci=red_state_laci,
    #                                                 bistable_mask=bistable_mask,
    #                                                 growth_rate=growth_rate,
    #                                                 red_potential=red_potential,
    #                                                 growth_rate_sim=growth_rate_sim,
    #                                                 ))
    # %%
    new_coolwarm = cm.get_cmap('RdYlBu')
    new_coolwarm = new_coolwarm(np.linspace(0, 1, 256))[80:-60][-1::-1]  # [60:-60]
    new_coolwarm = ListedColormap(new_coolwarm)
    vmax = 1400

    fig8, ax8 = plt.subplots(1, 1, figsize=(14, 10))
    img = ax8.imshow(cut_potential_laci_space[mask2].reshape(len(growth_rate_sim), -1),
                     cmap=new_coolwarm, interpolation='nearest',
                     norm=colors.PowerNorm(vmin=0, vmax=vmax, gamma=10), aspect='auto',
                     origin='lower', extent=[lac_mesh[mask2].min() / lac_mesh[mask2].max(), 1.,
                                             growth_mesh[mask2].min(), growth_mesh[mask2].max()])
    cb = fig8.colorbar(img, shrink=0.5, aspect=5, pad=0.1, ticks=[800, 1300, 1360, 1380, 1400])
    cb.set_label('$U$')
    linwidth = 8
    ax8.plot(green_state_laci[sim_index] / lac_mesh[mask2].max(), growth_rate[sim_index],
             color='#2F5597', ls=':', lw=linwidth)
    ax8.plot(green_state_laci[sim_index][bistable_mask[sim_index]] / lac_mesh[mask2].max(),
             growth_rate[sim_index][bistable_mask[sim_index]],
             color='#008000', ls='-', lw=linwidth)
    ax8.plot(red_state_laci[sim_index] / lac_mesh[mask2].max(), growth_rate[sim_index],
             color='#FF6000', ls='-', lw=linwidth)
    ax8.plot(red_state_laci[sim_index][bistable_mask[sim_index]] / lac_mesh[mask2].max(),
             growth_rate[sim_index][bistable_mask[sim_index]],
             color='#FF6000', ls='-', lw=linwidth)
    # contours = ax8.contour(lac_mesh[mask2].reshape(len(growth_rate_sim), -1) / lac_mesh[mask2].max(),
    #                        growth_mesh[mask2].reshape(len(growth_rate_sim), -1),
    #                        cut_potential_laci_space[mask2].reshape(len(growth_rate_sim), -1),
    #                        levels=[1350], colors='k', linestyles=':')
    # fmt = lambda x: '%.1f' % x
    # ax8.clabel(contours, inline=True, fmt=fmt)
    ax8.grid(False)
    ax8.set_xlabel('$\widetilde{[G]}$', labelpad=25, fontsize=35)
    ax8.set_ylabel('Growth rate ($h^{-1}$)', labelpad=25, fontsize=35)
    ax8.set_xlim((3e-3, 1e0))
    ax8.set_xscale('log')
    fig8.show()
    fig8.savefig(r'./data/potential_bifurcation_heat_map.png', transparent=True)
    # %%
    growth_select_index = [210]
    fig6, ax6 = plt.subplots(1, 1, figsize=(10, 10))
    for index_gr in growth_select_index:
        a = ax6.plot(lac_mesh[mask2].reshape(len(growth_rate_sim), -1)[index_gr, :] / lac_mesh[mask2].max(),
                     cut_potential_laci_space[mask2].reshape(len(growth_rate_sim), -1)[index_gr, :],
                     label='%2.2f $h^{-1}$' % growth_mesh[mask2].reshape(len(growth_rate_sim), -1)[index_gr, 0],
                     lw=8, color='#B8BDDF')
        # greenlacisst = green_sst[index_gr]
        # green_y_indice = np.argmin((laci_list_sim - greenlacisst) ** 2)
        # redlacisst = red_sst[index_gr]
        # red_y_indice = np.argmin((laci_list_sim - redlacisst) ** 2)
        # ax6.scatter(greenlacisst, laci_u[green_y_indice], marker='o', color=a[0].get_color())
        # ax6.scatter(redlacisst, laci_u[red_y_indice], marker='v', color=a[0].get_color())
    ax6.set_xlim(1e-2, 1.)
    # ax6.set_ylim(1, 1600)
    ax6.set_xscale('log')
    # ax6.set_yscale('log')
    ax6.set_ylim(0, 1600)
    # ax6.legend()
    ax6.set_xlabel('$\widetilde{[G]}$', labelpad=25, fontsize=35)
    ax6.grid(False)
    ax6.set_ylabel('U', labelpad=25, fontsize=35)
    fig6.show()

    # %% Test quasi potential landscape
    # laci_range = np.linspace(0, 10000, num=100)
    laci_range = np.logspace(-1, 3, num=500)

    tetr_range = np.logspace(-1, 3, num=500)

    extent = [-1, 3, -1, 3, ]
    init_laci, init_tetr = np.meshgrid(laci_range, tetr_range)
    init_laci = init_laci.flatten()
    init_tetr = init_tetr.flatten()

    gr_list = [2.3, 1.65, 1.0, 0.6, 0.4, 0.2]
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
        pos = ax.imshow(grid_p_list[i], interpolation='nearest',
                        cmap='coolwarm', vmin=0, vmax=1000,
                        norm=colors.PowerNorm(gamma=0.1),
                        origin='lower', extent=extent)
        ct = ax.contour(grid_p_list[i], levels=[20, 300, 900], origin='lower', linewidths=3, cmap='Set3', extent=extent)
        ax.clabel(ct, inline=True, fontsize=15, fmt='%.f')
        cb = fig1.colorbar(pos, ax=ax)

        ax.grid(False)
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        cb.set_label("$P$")
        ax.set_ylabel('LacI conc. (a.u.)')
        ax.set_xlabel('TetR conc. (a.u.)')
    fig1.show()
    fig1.savefig(r'./data/fig_potential.svg')
    # %%
    """
    This part shows a symmetric toggle and its null-clines
    """
    laci_range = np.linspace(0, 10000, num=10000)
    tetr_range = np.linspace(0, 10000, num=10000)

    toggle = ToggleBasic()
    toggle_pars = dict(k_t=10, k_l=26, tau_p_t=0.01, tau_p_l=0.05, n_t=3, alphal_over_gr=1, alphat_over_gr=1)
    assign_vars(toggle, toggle_pars)
    tetr_nucl = toggle.null_cline_tetr(laci_range)
    laci_nucl = toggle.null_cline_laci(tetr_range)

    fig8, ax8 = plt.subplots(1, 1)
    ax8.plot(tetr_range, laci_nucl)
    ax8.plot(tetr_nucl, laci_range)
    ax8.grid(False)
    # ax8.set_xscale('log')
    # ax8.set_yscale('log')
    ax8.set_xlim(0, 1.2)
    ax8.set_ylim(0, 1.2)
    ax8.set_xlabel('$\\tilde{T}$', labelpad=20)
    ax8.set_ylabel('$\\tilde{L}$', labelpad=20)
    fig8.show()

    phase_portrait = pd.read_csv('./data/M5_L3_phase_diagram.csv')
    green_ratio_exp = phase_portrait['LacI'] / (phase_portrait['TetR'] + phase_portrait['LacI'])
    green_ratio_gr = phase_portrait['Growth_rate']
    growth_rate_range = np.linspace(0.1, 1.6)
    dist_growth_sim = np.ones((len(green_ratio_gr), 1)) * growth_rate_range.reshape((1, -1))
    dist_growth_sim = (dist_growth_sim - green_ratio_gr.values.reshape((-1, 1))) ** 2
    exp_growth_index = np.argmin(dist_growth_sim, axis=1)
    assign_vars(toggle, toggle_pars)
    # tilde_k = np.array([[toggle.tilde_k_l, toggle.tilde_k_t] for growth_rate])
    tilde_k = []
    for gr in growth_rate_range:
        toggle.growth_rate = gr
        tilde_k.append([toggle.tilde_k_l, toggle.tilde_k_t])
    tilde_k = np.array(tilde_k)
    recip_tilde_k = np.ones(tilde_k.shape) / tilde_k

    ricip_kt = np.linspace(1, 40, num=500)
    ricip_kl = np.linspace(1, 50, num=500)
    extent = [ricip_kt[0], ricip_kt[-1], ricip_kl[0], ricip_kl[-1]]
    al_ricip_kt, al_ricip_kl = np.meshgrid(ricip_kt, ricip_kl)

    # pars_toggle = dict(k_t=2.3, k_l=7, tau_p_ltet=0.002, tau_p_trc=0.04)
    sst_state = Parallel(n_jobs=64)(delayed(bistability_titrate_k)([rec_t, rec_l], **toggle_pars)
                                    for rec_t, rec_l in tqdm(zip(al_ricip_kt.flatten(), al_ricip_kl.flatten())))
    print(np.sum(sst_state))
    sst_state = np.array(sst_state).reshape(al_ricip_kt.shape)

    fig1, ax1 = plt.subplots(1, 1)

    ax1.imshow(sst_state.T, origin='lower', cmap='coolwarm', alpha=0.3,
               aspect='auto', extent=[ricip_kl[0], ricip_kl[-1], ricip_kt[0], ricip_kt[-1]])
    ct = ax1.contour(ricip_kl, ricip_kt, sst_state.T, linewidths=4, cmap='Set3',
                     extent=[ricip_kl[0], ricip_kl[-1], ricip_kt[0], ricip_kt[-1]])
    ax1.grid(False)
    ax1.set_ylabel('$1/\\widetilde{K}_{\mathrm{DR}} (\\widetilde{\\alpha}_{\mathrm{R}}/K_{\mathrm{DR}})$')
    ax1.set_xlabel('$1/\\widetilde{K}_{\mathrm{DG}} (\\widetilde{\\alpha}_{\mathrm{G}}/K_{\mathrm{DG}})$')
    ax1.set_title(f'''n:{toggle_pars.get('n_t')}; tau: {toggle_pars.get('tau_p_ltet')}''', pad=20)
    ax1.plot(recip_tilde_k[:, 0], recip_tilde_k[:, 1], '--', color='k')
    ax1.scatter(recip_tilde_k[:, 0][exp_growth_index], recip_tilde_k[:, 1][exp_growth_index],
                c=green_ratio_gr, cmap='coolwarm', s=500)
    ax1.set_xlim(1, 50)
    ax1.set_ylim(1, 40)
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    fig1.show()
    fig1.savefig(r'./data/rescaled_states_bifurcation_with_tau.png', transparent=True)

    # %%
    """
    This part shows the dynamic cell
    """
    green_red_df = pd.read_csv(r'./data/Green_Red_time.csv')
    laci_tetr_init = [10, 1]
    pars_toggle = dict(k_t=2.3, k_l=7, tau_p_t=0.002, tau_p_l=0.04)
    laci_tetr_dyn = toggle_dynamic(time_max=60, step=0.01, **pars_toggle)
    fig3, ax3 = plt.subplots(1, 1)
    ax3.plot(np.arange(0.001, 60, step=0.01), laci_tetr_dyn[:, 0] / laci_tetr_dyn[:, 0].max(), '-g')
    ax3.plot(np.arange(0.001, 60, step=0.01), laci_tetr_dyn[:, 1] / laci_tetr_dyn[:, 1].max(), '-r')
    ax3.scatter(green_red_df['Time'], green_red_df['Green'])
    ax3.scatter(green_red_df['Time'], green_red_df['Red'])
    ax3.grid(False)
    ax3.set_xlabel('Time (h)')
    ax3.set_ylabel('Normalized expression level (a.u.)')
    fig3.show()
    sim_green_red = pd.DataFrame(data=dict(Time=np.arange(0.001, 60, step=0.01)[::10],
                                           Green=(laci_tetr_dyn[:, 0] / laci_tetr_dyn[:, 0].max())[::10],
                                           Red=(laci_tetr_dyn[:, 1] / laci_tetr_dyn[:, 1].max())[::10]))
    sim_green_red.to_csv(r'./data/Sim_Green_Red_time.csv')
    fig8, ax8 = plt.subplots(1, 1)
    for a, i in enumerate(range(len(green_state_tetr) - len(growth_rate_point), len(green_state_tetr))):
        ax8.scatter(green_state_tetr[i], green_state_laci[i], color='#43A047', marker=markers[a], s=500)
        ax8.scatter(red_state_tetr[i], red_state_laci[i], color='#EF5350', marker=markers[a], s=500)
    for a, i in enumerate(range(len(green_state_tetr) - len(growth_rate_point), len(green_state_tetr))):
        ax8.scatter(red_unstable[i], green_unstable[i], color='#FFC107', alpha=0.2, marker=markers[a], s=500)
    i_max = len(green_state_tetr) - len(growth_rate_point) - 1
    ax8.plot(green_state_tetr[0:i_max], green_state_laci[0:i_max], color='#43A047', ls='--', alpha=0.3)
    ax8.plot(red_state_tetr[0:i_max], red_state_laci[0:i_max], color='#EF5350', ls='--', alpha=0.3)
    ax8.plot(red_unstable[0:i_max], green_unstable[0:i_max], color='#FFC107', ls='--', alpha=0.3)
    ax8.plot(laci_tetr_dyn[:, 1], laci_tetr_dyn[:, 0])
    ax8.set_xscale('log')
    ax8.set_yscale('log')
    ax8.grid(False)
    # ax8.set_xlim(1, 1000)
    # ax8.set_ylim(1, 1000)
    ax8.set_xlabel('TetR Conc.')
    ax8.set_ylabel('LacI Conc.')
    fig8.show()
    fig8.savefig(r'./data/fig_sim_toggle_sst_with_kinetic_change.svg', transparent=True)
