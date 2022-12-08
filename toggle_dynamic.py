# -*- coding: utf-8 -*-

"""
This file is the main script describing the toggle switch.
"""
#%%
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


fitting = ['hill_function']
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
elif fitting[0] == 'gene_expression_rate_model':
    alpha_fuc = expression_level
    tetR_pars = dict(pars=[0.007, 5, 1.3E-1])
    lacI_pars = dict(pars=[0.03, 6., 1E-1])
elif fitting[0] == 'empirical mRNA':
    alpha_fuc = alpha_sst
    tetR_pars = dict(pars='tetr')
    lacI_pars = dict(pars='laci')
elif fitting[0] == 'hill_function':
    alpha_fuc = trasns_fitting
    tetR_pars = dict(pars=[26.836, 320.215, 1.0, 0.661, 4.09])
    lacI_pars = dict(pars=[16.609, 627.747, 1.0, 0.865, 4.635])


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
                       alphal_factor=1., alphat_factor=1.)

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

