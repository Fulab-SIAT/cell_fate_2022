# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
"""

import numpy as np  # Or any other
from typing import Union


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


def opt_frac_active(args, x, y, func):
    return y - func(x, *args)


# %%
if __name__ == '__main__':
    pass
