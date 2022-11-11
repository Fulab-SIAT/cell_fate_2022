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
from scipy.linalg import lstsq
from sciplot import whitegrid
from typing import Optional, Union

whitegrid()


# Own modules
def plasmid_prime(plasmid_CN, t, args=None, growth_rate: Optional[float] = None):
    """
    the ODEs of plasmid copy number control.

    Parameters
    ------
    plasmid_CN: float,
        The plasmid copy number at time t.
    t: float,
        Time, current time.
    args: list,
        the parameters of the ODEs. [alpha_II, tilde_k_I, growth_rate].
    growth_rate: float
        the growth rate of cells.

    Returns
    ------------
    CN_prime: float
        The derivative of plasmid copy number.
    """
    if args is None:
        args = [3.536615165485739, 15.09102618178972, 1.0]
    alpha_II, tilde_k_I, gr = args
    if growth_rate is not None:
        gr = growth_rate
    cn_prime = alpha_II * (1. / (1. + plasmid_CN / tilde_k_I)) * plasmid_CN - gr * plasmid_CN
    return cn_prime


def plasmid_sst(growth_rate: Union[float, np.ndarray], args=None):
    """
    The plasmid copy number at the steady state.

    Parameters
    ---------
    growth_rate: float, array
        the steady state growth rate, h-1.

    Returns
    ----------
    plasmid_CN
        plasmid copy number.

    """
    if args is None:
        args = [3.536615165485739, 15.09102618178972]
    alpha_II, tilde_k_I = args
    plasmid_CN = (1. / growth_rate - 1. / alpha_II) * (alpha_II * tilde_k_I)
    return plasmid_CN


def plasmid_prime_v2(plasmid_CN, t, args=None, growth_rate: Optional[float] = None):
    """
    the ODEs of plasmid copy number control.

    Parameters
    ------
    plasmid_CN: float,
        The plasmid copy number at time t.
    t: float,
        Time, current time.
    args: list,
        the parameters of the ODEs. [alpha_II, tilde_k_I, growth_rate].
    growth_rate: float
        the growth rate of cells.

    Returns
    ------------
    CN_prime: float
        The derivative of plasmid copy number.
    """
    if args is None:
        args = [3.536615165485739, 15.09102618178972, 0.5, 1.0, 0.1]
    alpha_II, tilde_k_I, gr, K_g, m, a = args
    if growth_rate is not None:
        gr = growth_rate
    R_rep = 1. / (1. + K_g / (m * gr + a))
    cn_prime = alpha_II * R_rep * (1. / (1. + plasmid_CN / tilde_k_I)) * plasmid_CN - gr * plasmid_CN
    return cn_prime


def plasmid_sst_v2(growth_rate: Union[float, np.ndarray], args=None):
    """
    The plasmid copy number at the steady state.

    Parameters
    ---------
    growth_rate: float, array
        the steady state growth rate, h-1.

    Returns
    ----------
    plasmid_CN
        plasmid copy number.

    """
    if args is None:
        args = [3.536615165485739, 15.09102618178972, 0.5, 1.0, 0.1]
    alpha_II, tilde_k_I, K_g, m, a  = args
    R_rep = 1. / (1. + (K_g / (m * growth_rate + a))**2)
    # plasmid_CN = (1. / growth_rate - 1. / (alpha_II * R_rep)) * (alpha_II * R_rep * tilde_k_I)
    plasmid_CN = tilde_k_I * (alpha_II * R_rep / growth_rate - 1.)
    return plasmid_CN



# %%
if __name__ == '__main__':
    # %% Data Fitting
    data_pass = r'.\sub_model\growth_rate_vs_plasmid_copy_number.xlsx'
    data = pd.read_excel(data_pass, sheet_name='Sheet1')
    y_plasmid = data['CN'].to_numpy()
    x_growth_rate = data['lambda'].to_numpy()

    # least square fitting
    x_1_over_lambda = 1 / x_growth_rate
    A_plasmid = y_plasmid[:, np.newaxis] ** [0, 1]
    pars, res, rank, s = lstsq(A_plasmid, x_1_over_lambda)

    fig1, ax1 = plt.subplots(1, 1, figsize=(13, 13))
    ax1.scatter(y_plasmid, x_1_over_lambda, c='#898686', s=500)
    fit_x = np.linspace(y_plasmid.min() * 0.9, y_plasmid.max() * 1.1)
    ax1.plot(fit_x, pars[0] + pars[1] * fit_x, c='#00A8FF')
    ax1.set_xlabel('Plasmid copy number (Relative to Chr.)')
    ax1.set_ylabel('1 / $\lambda$ (h)')
    fig1.show()

    fig2, ax2, = plt.subplots(1, 1, figsize=(13, 13))
    ax2.scatter(x_growth_rate, y_plasmid, s=500, c='#898686')
    growth_range = np.linspace(0.2, 1.8)
    ax2.plot(growth_range, plasmid_sst(growth_range), c='#00A8FF')
    ax2.set_xlabel('Growth rate ($h^{-1}$)')
    ax2.set_ylabel('Plasmid copy number (Relative to Chr.)')
    fig2.show()

    # %%

    fig3, ax3, = plt.subplots(1, 1, figsize=(13, 13))
    # ax2.scatter(x_growth_rate, y_plasmid, s=500, c='#898686')
    growth_range = np.linspace(0.12, 1.2)
    ax3.plot(1/growth_range, plasmid_sst_v2(growth_range, [3.536615165485739, 15.09102618178972, 0.1, .2, 0.01]),
             c='#00A8FF')
    ax3.plot(1/growth_range, plasmid_sst_v2(growth_range, [4, 15.09102618178972, 0.1, .2, 0.01]),
             c='#00A8FF')
    ax3.set_xlabel('Growth rate ($h^{-1}$)')
    ax3.set_ylabel('Plasmid copy number (Relative to Chr.)')
    fig3.show()
