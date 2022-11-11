# %%
import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm import tqdm
import seaborn as sns
import sciplot as splt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import fsolve
from joblib import Parallel, delayed, dump, load


class ToggleSwitch:
    '''
    Toggle switch model
    '''

    def __init__(self, paras=None):
        # initial the model parameters
        self.tau_p_t = 0.01
        self.tau_p_l = 0.2
        self.p_t = 100.4601  # for translate lacI
        self.p_l = 7  # for translate tetR
        self.Alpha2 = 1  # scaler for two repressor, lacI & tetR
        self.k_t = 0.5 * self.Alpha2  # k_t tetR binding affinity
        self.k_l = 5 * self.Alpha2  # k_l lacI binding affinity
        self.n_t = 2.  # n_t tetR cooperate coefficient
        self.n_l = 4.  # n_l lacI cooperate coefficient
        self.lambda_0 = 1.04  # growth rate, this parameter indicate host cell's growth rate without circuits (h^-1)
        self.theta_0 = 1.26  # from Klummp 2013, hr^-1
        self.beta = self.lambda_0 / (self.lambda_0 + self.theta_0)
        self.Alpha = 0.015  # scaler for translational flow
        self.alpha_t = 0.3 * self.Alpha
        self.alpha_l = 0.7 * self.Alpha
        self.k_iptg = 8  # iptg binding affinity
        self.k_atc = 10.1  # atc binding affinity
        self.iptg_conc = 0.  # iptg conc.
        self.atc_conc = 0.
        self.m = 2.  # cooperate of atc
        self.n = 2.  # cooperate of iptg
        self.leak_p_t = self.p_t * self.tau_p_t
        self.leak_p_l = self.p_l * self.tau_p_l

        self.paras_list = dict(tau_p_t=self.tau_p_t, tau_p_l=self.tau_p_l, p_t=self.p_t,
                               p_l=self.p_l, Alpha2=self.Alpha2, k_t=self.k_t, k_l=self.k_l,
                               n_t=self.n_t, n_l=self.n_l, lambda_0=self.lambda_0, beta=self.beta,
                               Alpha=self.Alpha, alpha_t=self.alpha_t, alpha_l=self.alpha_l)
        self.lac_cons = None
        self.tet_cons = None
        self.dev_laci_list = None
        self.gr_list = None
        self.dev_laci = None
        self.sst_index = None
        self.sst_laci_conc = None
        self.sst_tetr_conc = None
        self.sst_gr = None
        self.bistable = None
        self.laci_potential = None
        self.sst_state = None
        if paras is not None:
            self.paras_list = paras

    def tune_Alpha(self, alpha_):
        """
        function for dynamicl change paramer, alpha, tune the gene expression burden
        :param alpha_:
        :return:
        """
        self.Alpha = alpha_
        self.alpha_t = 0.4 * self.Alpha
        self.alpha_l = 0.6 * self.Alpha

    def tune_lambda_0(self, lambda_):
        """
        tune the parameter lambda_0
        :param lambda_: float
        :return:
        """
        self.lambda_0 = lambda_
        self.beta = self.lambda_0 / (self.lambda_0 + self.theta_0)

    def tune_tau_p_t(self, tau):
        self.tau_p_t = tau
        self.leak_p_t = tau * self.p_t

    def tune_tau_p_l(self, tau):
        self.tau_p_l = tau
        self.leak_p_l = tau * self.p_l

    def eq_func(self, tetr_, laci_):
        """
        This function use to numerical solve the steady state when assigned TetR or LacI, and return the growth difference
        for determining whether function is balance.
        compute steay state growth rate when assigned TetR and LacI conc.
        :param tetr: Total TetR protein in cell, including binding in DNA, free in cytoplasm and binging with cTc, float
        :param laci: Total lacI protein in cell, float
        :return: d[lambda]
        """
        laci_tot = laci_
        tetr_tot = tetr_
        tetr = tetr_tot * (1. + self.atc_conc / self.k_atc * tetr_tot) ** -self.m
        laci = laci_tot * (1. + self.iptg_conc / self.k_iptg * laci_tot) ** -self.n
        a = self.h_l(laci)  # tetR expression rate (tetR flux)
        # let d[TetR]/dt == 0, to get the growth rate.
        gr = self.lambda_0 / self.beta * (1. - tetr_tot / a)
        b = gr * (1. - self.beta * gr / self.lambda_0)
        c = self.h_t(tetr)  # lacI expression rate (lacI flux)
        return gr - self.lambda_0 * (1. - self.alpha_t * b * a - self.alpha_l * b * c)

    def h_l(self, laci):
        return self.leak_p_l + self.p_l * (1. - self.tau_p_l) / (1. + (laci / self.k_l) ** self.n_l)

    def h_t(self, tetr):
        return self.leak_p_t + self.p_t * (1. - self.tau_p_t) / (1. + (tetr / self.k_t) ** self.n_t)

    def eq_func_tetR(self, tetr, laci):
        """
        This function use to numerical solve the steady state when assigned TetR or LacI, and return the growth difference
        for determining whether function is balance.
        compute steay state growth rate when assigned TetR and LacI conc.
        :param tetr: Total TetR protein in cell, including binding in DNA, free in cytoplasm and binging with cTc, float
        :param laci: Total lacI protein in cell, float
        :return: d[lambda]
        """
        laci_tot = laci
        tetr_tot = tetr
        tetr = tetr_tot * (1. + self.atc_conc / self.k_atc * tetr_tot) ** -self.m
        laci = laci_tot * (1. + self.iptg_conc / self.k_iptg * laci_tot) ** -self.n
        a = self.h_l(laci)  # laci promoter, PLtetO
        # let d[TetR]/dt == 0, to get the growth rate.
        gr = self.lambda_0 / self.beta * (
                1. - laci_tot / a)
        b = gr * (1. - self.beta * gr / self.lambda_0)
        c = self.leak_p_t + self.p_t * (1. - self.tau_p_t) / (
                1. + (tetr / self.k_t) ** self.n_t)  # lacI expression rate (lacI flux)
        return gr - self.lambda_0 * (1. - self.alpha_t * b * a - self.alpha_l * b * c)

    def dev_laci_func(self, gr, tetr_tot, laci_tot):
        """
        After steady state variables are determined, compute lacI expression rate
        :param gr: growth rate
        :param tetr_tot: TetR total conc
        :param laci: LacI conc.
        :return: d[LacI]/dt
        """
        tetr = tetr_tot * (1. + self.atc_conc / self.k_atc * tetr_tot) ** -self.m  # free TetR
        laci = laci_tot * (1. + self.iptg_conc / self.k_iptg * laci_tot) ** -self.n
        w_dev_g = (1. - self.beta * gr / self.lambda_0)
        return w_dev_g * self.h_t(tetr) - laci_tot

    def compute_dev_laci(self, laci_):
        '''
        compute steady when assigned LacI conc.
        :param laci_: lacI conc.
        :return: a list containing [tetR conc., lambda, d[LacI]/dt]
        '''
        if laci_ == np.nan:
            return [np.nan, np.nan, np.nan]
        laci_tot = laci_
        laci = laci_tot * (1. + self.iptg_conc / self.k_iptg * laci_tot) ** -self.n
        root_ = fsolve(self.eq_func, 1., args=(laci_,))[0]  # root_ == tetR conc.
        p_lac_act = self.h_l(laci)
        lambda_t_ = self.lambda_0 / self.beta * (1. - root_ / p_lac_act)
        dev_laci_ = self.dev_laci_func(lambda_t_, root_, laci_)
        return [root_, lambda_t_, dev_laci_]

    def potential(self, u, laci_):
        # only return d[LacI]/dt, for less computational load for integrate LacI potential.
        laci_tot = laci_
        laci = laci_tot * (1. + self.iptg_conc / self.k_iptg * laci_tot) ** -self.n
        root_ = newton(self.eq_func, 1, args=(laci_tot,))  # root_ == tetR conc.
        p_lac_act = self.h_l(laci)
        lambda_t_ = self.lambda_0 / self.beta * (1. - root_ / p_lac_act)
        dev_laci_ = self.dev_laci_func(lambda_t_, root_, laci_)
        return - dev_laci_

    def comput_potenital_laci(self):
        # compute potential
        laci_potential = odeint(self.potential, 0, self.lac_cons)
        self.laci_potential = laci_potential.flatten()

    def solve_landscape(self, lac_cons_range=None):
        # get the data:  laci_conc vs dev_laci
        if ~(lac_cons_range is None):
            self.lac_cons = lac_cons_range
        self.dev_laci_list = np.array([self.compute_dev_laci(lac_conc) for lac_conc in self.lac_cons])
        self.tet_cons = self.dev_laci_list[:, 0]
        self.gr_list = self.dev_laci_list[:, 1]
        self.dev_laci = self.dev_laci_list[:, 2]

    def sovle_sst(self):
        # find out the steady state point and
        # attention! this function can only used after solve_landscape
        sign_d_laci = np.sign(self.dev_laci)
        root = np.diff(sign_d_laci)
        self.sst_index = np.nonzero(root)[0]
        self.sst_laci_conc = self.lac_cons[self.sst_index]
        self.sst_tetr_conc = self.tet_cons[self.sst_index]
        self.sst_gr = self.gr_list[self.sst_index]
        self.sst_state = root[self.sst_index]
        if (len(self.sst_index) == 3):
            self.bistable = True
        else:
            self.bistable = False

    def get_all_pars(self):
        self.paras_list = dict(tau_p_t=self.tau_p_t, tau_p_l=self.tau_p_l, p_t=self.p_t,
                               p_l=self.p_l, Alpha2=self.Alpha2, k_t=self.k_t, k_l=self.k_l,
                               n_t=self.n_t, n_l=self.n_l, lambda_0=self.lambda_0, beta=self.beta,
                               Alpha=self.Alpha, alpha_t=self.alpha_t, alpha_l=self.alpha_l)
        return self.paras_list

    def atc_laci_bifurcation(self, inducer_conc_rang, laci_conc_rang):
        """
        :param inducer_conc_rang: ndarray, [aTc, IPTG]
        :param laci_conc_rang:
        :return:
        """
        inducer_conc_rang = inducer_conc_rang
        laci_sst_list = np.ones((len(inducer_conc_rang), 3)) * np.nan
        tetr_sst_list = np.ones((len(inducer_conc_rang), 3)) * np.nan
        # Parallel(n_jobs=-1, require='sharedmem')(
        #     delayed(self.parallel_func)(index) for index in range(len(inducer_conc_rang))
        # )
        for index in tqdm(range(len(inducer_conc_rang))):
            self.atc_conc, self.iptg_conc = inducer_conc_rang[index, :]
            self.solve_landscape(laci_conc_rang)
            sign_d_laci = np.sign(self.dev_laci)
            root = np.diff(sign_d_laci)
            self.sst_index = np.nonzero(root)[0]
            self.sst_state = root[self.sst_index]
            len_of_index = len(self.sst_index)
            if len_of_index > 3:
                print(f'inducer atc: {self.atc_conc} and IPTG: {self.iptg_conc} may cause more than two sss!')
            # saddle_mask = self.sst_state == -2
            self.sst_index = self.sst_index[:3]  # trimmed last unstable state
            # reverse apply the sst point!
            # saddle_lacI = np.min(self.lac_cons[self.sst_index[saddle_mask]])
            # saddle_tetR = np.min(self.tet_cons[self.sst_index[saddle_mask]])
            if self.atc_conc >= self.iptg_conc:
                laci_sst_list[index, -1:-(len_of_index + 1):-1] = self.lac_cons[self.sst_index][::-1]
                tetr_sst_list[index, -1:-(len_of_index + 1):-1] = self.tet_cons[self.sst_index][::-1]
            else:
                laci_sst_list[index, 0:len_of_index] = self.lac_cons[self.sst_index]
                tetr_sst_list[index, 0:len_of_index] = self.tet_cons[self.sst_index]
        return laci_sst_list, tetr_sst_list

def titrate_lambda0(lambda_, lac_cons=None, iptg_cons=None, atc_cons=None):
    if lac_cons == None:
        lac_cons = np.linspace(0, 15, num=2048)
    if iptg_cons == None:
        iptg_cons = np.linspace(0, 2.0, num=512)
    if atc_cons == None:
        atc_cons = np.linspace(0, 1.0, num=512)
    toggle = ToggleSwitch()
    toggle.tune_lambda_0(lambda_)
    inducer_conc = np.zeros((len(atc_cons) + len(iptg_cons), 2))
    inducer_conc[0:len(atc_cons), 0] = atc_cons
    inducer_conc[len(atc_cons):, 1] = iptg_cons
    curves_lacI_atc, curves_tetR_atc = toggle.atc_laci_bifurcation(inducer_conc, lac_cons)
    lacI_cons = curves_lacI_atc
    mean_uss = np.nanmean(lacI_cons[:, 1])
    arranged_lacI = np.ones(lacI_cons.shape) * np.nan
    for i in range(len(lacI_cons[:, 0])):
        if np.isnan(lacI_cons[i, 0]):
            pass
        elif lacI_cons[i, 0] >= mean_uss:
            arranged_lacI[i, 2] = lacI_cons[i, 0]
        else:
            arranged_lacI[i, 0] = lacI_cons[i, 0]
    for j in range(len(lacI_cons)):
        if np.isnan(lacI_cons[j, 2]):
            pass
        elif lacI_cons[j, 2] >= mean_uss:
            arranged_lacI[j, 2] = lacI_cons[j, 2]
        else:
            arranged_lacI[j, 0] = lacI_cons[j, 2]
    arranged_lacI[:, 1] = lacI_cons[:, 1]
    tetR_cons = curves_tetR_atc
    mean_uss = np.nanmean(tetR_cons[:, 1])
    arranged_tetR = np.ones(tetR_cons.shape) * np.nan
    for i in range(len(tetR_cons[:, 0])):
        if np.isnan(tetR_cons[i, 0]):
            pass
        elif tetR_cons[i, 0] < mean_uss:
            arranged_tetR[i, 2] = tetR_cons[i, 0]
        else:
            arranged_tetR[i, 0] = tetR_cons[i, 0]
    for j in range(len(tetR_cons)):
        if np.isnan(tetR_cons[j, 2]):
            pass
        elif tetR_cons[j, 2] < mean_uss:
            arranged_tetR[j, 2] = tetR_cons[j, 2]
        else:
            arranged_tetR[j, 0] = tetR_cons[j, 2]
    arranged_tetR[:, 1] = tetR_cons[:, 1]
    return [arranged_lacI, arranged_tetR]


# %%
if __name__ == '__main__':
    # %%
    toggle = ToggleSwitch()
    lac_cons = np.linspace(0, 10, 800)
    toggle.solve_landscape(lac_cons)
    # dev_laci_inter = interpolate.CubicSpline(x=toggle.lac_cons, y=toggle.dev_laci ** 2, )
    # dd_laci_inter = dev_laci_inter.derivative(1)
    # dd_laci_inter = dd_laci_inter(toggle.lac_cons)
    toggle.sovle_sst()

    promoter_landscape = np.linspace(0.05, 80)
    p_l, p_t = np.meshgrid(promoter_landscape, promoter_landscape)
    bistable_matrix = np.ones(p_l.shape) * 0
    for i in tqdm(range(p_l.shape[0])):
        for j in range(p_l.shape[1]):
            toggle.p_l = p_l[i, j]
            toggle.p_t = p_t[i, j]
            toggle.solve_landscape(lac_cons)
            toggle.sovle_sst()
            bistable_matrix[i, j] = toggle.bistable

    # %% bistable or monostable ?
    fig1, ax = plt.subplots()
    im_matrix = bistable_matrix[-1::-1, :]
    pf = pd.DataFrame(data=im_matrix,
                      index=np.around(promoter_landscape[-1::-1]),
                      columns=np.around(promoter_landscape))
    sns.heatmap(pf, ax=ax, cmap='YlGnBu',
                xticklabels=5,
                yticklabels=5)
    fig1.show()

    # %%
    toggle = ToggleSwitch()
    lac_cons = np.linspace(0, 10, 800)
    toggle.lac_cons = lac_cons
    toggle.comput_potenital_laci()
    fig2, ax = plt.subplots()
    splt.whitegrid()
    sns.lineplot(x=toggle.lac_cons, y=toggle.laci_potential)
    fig2.show()
    # %% test the bifurcation curve
    toggle = ToggleSwitch()
    # lac_cons = np.append(np.linspace(0, 5, 256, endpoint=True), np.linspace(15, 17, 40))
    lac_cons = np.linspace(0, 15, num=2048)
    # atc_cons = np.linspace(0, 0.4, 200)
    # atc_cons = np.logspace(np.log(1e-15), np.log(0.8), base=np.e, num=50)
    # iptg_cons = np.logspace(np.log(1e-15), np.log(0.12), base=np.e, num=50)
    atc_cons = np.linspace(0, 1.0, num=512)
    iptg_cons = np.linspace(0, 2.0, num=512)
    toggle.tune_lambda_0(0.18)
    inducer_conc = np.zeros((len(atc_cons) + len(iptg_cons), 2))
    inducer_conc[0:len(atc_cons), 0] = atc_cons
    inducer_conc[len(atc_cons):, 1] = iptg_cons
    # titrate atc
    curves_lacI_atc, curves_tetR_atc = toggle.atc_laci_bifurcation(inducer_conc, lac_cons)

    # fig3, ax = plt.subplots(1, 2, figsize=(17, 8))
    # splt.whitegrid()
    # ax[0].plot(atc_cons, curves_lacI_atc[:len(atc_cons), 0], '-k', label='Low Level', lw=3)
    # ax[0].plot(atc_cons, curves_lacI_atc[:len(atc_cons), 1], '--', label='Unstable', lw=3)
    # ax[0].plot(atc_cons, curves_lacI_atc[:len(atc_cons), 2], '-g', label='High Level', lw=3)
    # ax[1].plot(iptg_cons, curves_lacI_atc[len(atc_cons):, 0], '-k', lw=3)
    # ax[1].plot(iptg_cons, curves_lacI_atc[len(atc_cons):, 1], '--', lw=3)
    # ax[1].plot(iptg_cons, curves_lacI_atc[len(atc_cons):, 2], '-g', lw=3)
    # fig3.legend()
    # ax[0].set_xlim((0, atc_cons.max()))
    # ax[0].set_xlabel('aTc conc. (ng/mL)')
    # ax[0].set_ylabel('LacI conc. ($\mathrm{\mu M}$)')
    # ax[1].set_xlim((0, iptg_cons.max()))
    # ax[1].set_xlabel('IPTG conc. (ng/mL)')
    # ax[1].set_ylabel('LacI conc. ($\mathrm{\mu M}$)')
    # fig3.show()

    # fig4, ax4 = plt.subplots(1, 2, figsize=(17, 8))
    # splt.whitegrid()
    # ax4[0].plot(atc_cons, curves_tetR_atc[:len(atc_cons), 0], '-k', label='Low Level', lw=3)
    # ax4[0].plot(atc_cons, curves_tetR_atc[:len(atc_cons), 1], '--', label='Unstable', lw=3)
    # ax4[0].plot(atc_cons, curves_tetR_atc[:len(atc_cons), 2], '-g', label='High Level', lw=3)
    # ax4[1].plot(iptg_cons, curves_tetR_atc[len(atc_cons):, 0], '-k', label='Low Level', lw=3)
    # ax4[1].plot(iptg_cons, curves_tetR_atc[len(atc_cons):, 1], '--', label='Unstable', lw=3)
    # ax4[1].plot(iptg_cons, curves_tetR_atc[len(atc_cons):, 2], '-g', label='High Level', lw=3)
    # fig4.legend()
    # ax4[0].set_xlim((0, atc_cons.max()))
    # ax4[0].set_xlabel('aTc conc. (ng/mL)')
    # ax4[0].set_ylabel('TetR conc. ($\mathrm{\mu M}$)')
    # ax4[1].set_xlim((0, iptg_cons.max()))
    # ax4[1].set_xlabel('IPTG conc. (ng/mL)')
    # ax4[1].set_ylabel('TetR conc. ($\mathrm{\mu M}$)')
    # fig4.show()

    fig5, ax5 = plt.subplots(figsize=(8, 8))
    splt.whitegrid()
    ax5.scatter(curves_tetR_atc[:, 0], curves_lacI_atc[:, 0], c='g', label='Red State')
    ax5.scatter(curves_tetR_atc[:, 2], curves_lacI_atc[:, 2], c='r', label='Green State')
    ax5.set_xlabel('TetR conc. (a. u.)')
    ax5.set_ylabel('LacI conc. (a. u.)')
    # ax5.scatter(curves_lacI_atc[:, 1], curves_tetR_atc[:, 1], c='k', label='Unstable')
    # ax5.scatter(curves_lacI_atc[:, 2], curves_tetR_atc[:, 2], c='r', label='Green State')
    # # ax5.scatter(curves_lacI_iptg[:, 0], curves_tetR_atc[len(atc_cons):, 0], c='g', label='Red State')
    # ax5.scatter(curves_lacI_iptg[:, 1], curves_tetR_atc[len(atc_cons):, 1], c='k', label='Unstable')
    # ax5.scatter(curves_lacI_iptg[:, 2], curves_tetR_atc[len(atc_cons):, 2], c='r', label='Green State')
    fig5.show()

    fig6, ax6 = plt.subplots(1, 1, figsize=(8, 8))
    splt.whitegrid()
    ax6.plot(atc_cons, curves_lacI_atc[:len(atc_cons), 0], '-r', label='Low Level', lw=3)
    ax6.plot(atc_cons, curves_lacI_atc[:len(atc_cons), 1], '--k', label='Unstable', lw=3)
    ax6.plot(atc_cons, curves_lacI_atc[:len(atc_cons), 2], '-g', label='High Level', lw=3)
    ax6.plot(-iptg_cons, curves_lacI_atc[len(atc_cons):, 0], '-r', lw=3)
    ax6.plot(-iptg_cons, curves_lacI_atc[len(atc_cons):, 1], '--k', lw=3)
    ax6.plot(-iptg_cons, curves_lacI_atc[len(atc_cons):, 2], '-g', lw=3)
    fig6.legend(loc='upper left')
    ax6.set_xlim((-iptg_cons.max(), atc_cons.max()))
    ax6.set_xlabel('Inducer conc. (ng/mL)')
    ax6.set_ylabel('LacI conc. (a. u.)')
    fig6.show()

    fig7, ax7 = plt.subplots(1, 1, figsize=(8, 8))
    splt.whitegrid()
    ax7.plot(atc_cons, curves_tetR_atc[:len(atc_cons), 0], '-r', label='Low Level', lw=3)
    ax7.plot(atc_cons, curves_tetR_atc[:len(atc_cons), 1], '--k', label='Unstable', lw=3)
    ax7.plot(atc_cons, curves_tetR_atc[:len(atc_cons), 2], '-g', label='High Level', lw=3)
    ax7.plot(-iptg_cons, curves_tetR_atc[len(atc_cons):, 0], '-r', lw=3)
    ax7.plot(-iptg_cons, curves_tetR_atc[len(atc_cons):, 1], '--k', lw=3)
    ax7.plot(-iptg_cons, curves_tetR_atc[len(atc_cons):, 2], '-g', lw=3)
    fig7.legend(loc='upper left')
    ax7.set_xlim((-iptg_cons.max(), atc_cons.max()))
    ax7.set_xlabel('Inducer conc. (ng/mL)')
    ax7.set_ylabel('TetR conc. (a. u.)')
    fig7.show()

    # %%
    toggle = ToggleSwitch()
    lac_cons = np.linspace(0, 300, 1000)
    toggle.tune_lambda_0(0.2)
    toggle.iptg_conc = 0.  # 1.3
    toggle.atc_conc = 1000  # 0.5
    toggle.solve_landscape(lac_cons)
    fig1, ax = plt.subplots()
    ax.plot(toggle.lac_cons, toggle.dev_laci)
    #ax.plot(toggle.lac_cons, toggle.gr_list)
    ax.plot(toggle.lac_cons, [0] * len(toggle.lac_cons), '--r')
    # ax.set_ylim(-0.2, .2)
    # ax.set_xlim(0, 20)
    fig1.show()
    toggle.sovle_sst()
    print(toggle.sst_laci_conc)
    print(toggle.sst_tetr_conc)

    # %% tune lambda_
    toggle = ToggleSwitch()
    def tune_lambda(lambda_):
        toggle.tune_lambda_0(lambda_)
        return (1. - toggle.beta) * lambda_


    lambdas = np.linspace(0.1, 2.3)
    trans_levle = [tune_lambda(lambda_) for lambda_ in lambdas]
    fig, ax = plt.subplots()
    ax.plot(lambdas, trans_levle)
    fig.show()

    # %%
    lamndas = np.linspace(0.2, 1.6, 32)
    lacI_tetR_along_gr = Parallel(n_jobs=32)(delayed(titrate_lambda0)(lambda_)
                                             for lambda_ in lamndas)
    dump(dict(toggle_obj=toggle,
              inducer_concentration=inducer_conc,
              lambda_list=lamndas,
              lacI_TetR_change=lacI_tetR_along_gr),
         r'./saved_mem/20200919_titrate_growth_rate_bifurcation_curve_3.joblib')
    #%%
    gr_index = [0]
    fig6, ax6 = plt.subplots(1, 1, figsize=(8, 8))
    splt.whitegrid()
    for index in gr_index:
        lacI_cons = lacI_tetR_along_gr[index][0]
        ax6.plot(atc_cons/atc_cons.max(), lacI_cons[:len(atc_cons), 1], '-r', lw=3)
        # ax6.plot(atc_cons/atc_cons.max(), lacI_cons[:len(atc_cons), 1], '--k', lw=3)
        # ax6.plot(atc_cons/atc_cons.max(), lacI_cons[:len(atc_cons), 2], '-g', lw=3)
        ax6.plot(-iptg_cons/iptg_cons.max(), lacI_cons[len(atc_cons):, 1], '-r', lw=3)
        # ax6.plot(-iptg_cons/iptg_cons.max(), lacI_cons[len(atc_cons):, 1], '--k', lw=3)
        # ax6.plot(-iptg_cons/iptg_cons.max(), lacI_cons[len(atc_cons):, 2], '-g', lw=3)
    # fig6.legend(loc='upper left')
    # ax6.set_xlim(-.2, .5)
    ax6.set_xlabel('Inducer conc. (ng/mL)')
    ax6.set_ylabel('LacI conc. (a. u.)')
    # ax6.set_yscale('log')
    fig6.show()