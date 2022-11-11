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
from sub_model.plasmid_control import plasmid_prime, plasmid_sst
from functools import partial
from scipy.optimize import leastsq

from typing import Optional


# […]

# Own modules


def protein_trans_rate(growth_rate, a=21.6, b=0.05, c=0.21, d=0.08):
    """
    Ribosomes elongation rate VS growth rate.
    Parameters fitted from Article a=21.60534191, b=0.0536743, c=0.2116, d=0.08
    """
    phi_r = phi_ribosome(growth_rate, c, d)
    return phi_r * a / (phi_r + b)


def phi_ribosome(growth_rate, c=0.2116, d=0.08, sigma=2.21):
    """
    Note: r=RNA/Protein, r = sigma * phi_r, the defaults are obtained from Dai XF et al. 2016.
    parameters in article. c=0.2116, d=0.08, sigma=2.21
    """
    return (c * growth_rate + d) / sigma


def frc_act_ribo(growth_rate, a=1.01494589, b=0.15090343, n=1.35893622):
    """
    Fraction of activated ribosome conc. VS growth rate. The parameters are collected from Dai XF et al. 2016
    """
    return 1 / (a + (b / growth_rate) ** n)


def ribo_elongation_klumpp(_lambda, args=None):
    """
    Klumpp et. al. 2013 model for ribosome elongation rate.

    Parameters
    ---------------
    _lambda: float or list
        growth rate, h-1
    args: list
        [gamma, lambda_M]
    """
    if args is None:
        args = [10.2, 0.49]
    gamma, lambda_M = args
    return gamma * _lambda / (lambda_M + _lambda)


def ribo_klumpp(_lambda, args=None):
    if args is None:
        args = [0.035, 0.029, 10.2]
    phi_0, phi_M, gamma_max = args

    return phi_0 + phi_M + _lambda / gamma_max


def const_transcript(growth_rate, k=1.457, n=1., tau=0.2, strain: Optional[str] = None):
    """
    Normalized transcription rate for different protein

    """
    if strain is None:
        pass
    elif strain == 'Red':
        k, n, tau = [0.01, 4, 1.3E-1]
    elif strain == 'Green':
        k, n, tau = [0.01, 5.5, 1.4E-1]
        # phi_r = phi_ribosome(growth_rate)
    trans_rate = (1. - tau) * growth_rate ** n / (growth_rate ** n + k) + tau
    # trans_rate = k * (growth_rate - tau) ** 1
    return trans_rate


def expression_level(growth_rate, pars=None):
    plasmid_args = [2, 15.]
    if pars is None:
        alpha_i = frc_act_ribo(growth_rate) * protein_trans_rate(growth_rate) * const_transcript(growth_rate) \
                  * plasmid_sst(growth_rate, args=plasmid_args)
        # alpha_i = alpha_i / alpha_i.max()
        return alpha_i
    alpha_i = frc_act_ribo(growth_rate) * protein_trans_rate(growth_rate) * const_transcript(growth_rate, *pars) \
              * plasmid_sst(growth_rate, args=plasmid_args)
    # alpha_i = alpha_i / alpha_i.max()
    return alpha_i


def lsq_loss(pars, growth_rate, target_y):
    y_prime = expression_level(growth_rate, pars)
    return np.abs(target_y - y_prime)


# %%
if __name__ == '__main__':
    # %%

    alpha_data = pd.read_excel(r'sub_model/growth_rate_vs_alpha.xlsx', sheet_name='Sheet1')

    red_alpha = alpha_data[alpha_data['strain'] == 'Red']
    green_alpha = alpha_data[alpha_data['strain'] == 'Green']

    red_expression_level = red_alpha['alpha'].values
    red_expression_level = red_expression_level / red_expression_level.max()
    green_expression_level = green_alpha['alpha'].values
    green_expression_level = green_expression_level / green_expression_level.max()

    red_error_func = partial(lsq_loss,
                             growth_rate=red_alpha['Growth Rate'].values,
                             target_y=red_expression_level)
    green_error_func = partial(lsq_loss,
                               growth_rate=red_alpha['Growth Rate'].values,
                               target_y=green_expression_level)

    red_fitting_pars = leastsq(red_error_func, np.array([30, 1, -1]))
    print(f'RED Pars: {red_fitting_pars}')
    green_fitting_pars = leastsq(green_error_func, np.array([30, 1, -1]))
    print(f'Green Pars: {green_fitting_pars}')
    #
    lambda_list = np.linspace(0.2, 1.7, num=1000)

    alpha_red = expression_level(lambda_list, [0.01, 4, 1.3E-1])
    alpha_green = expression_level(lambda_list, [0.01, 5.5, 1.4E-1])

    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))

    ax1.plot(lambda_list, alpha_red / alpha_red.max(), color='#FE6508')
    ax1.plot(lambda_list, alpha_green / alpha_green.max(), color='#188C18')

    ax1.scatter(red_alpha['Growth Rate'].values,
                red_expression_level, color='#FE6508')
    ax1.scatter(green_alpha['Growth Rate'].values,
                green_expression_level, color='#188C18')

    ax1.set_xlabel('Growth rate ($h^{-1}$)')
    ax1.set_ylabel('$\\alpha_{R, G}$')

    fig1.show()
    # %%
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))

    ax2.plot(lambda_list, alpha_red / lambda_list, color='#FE6508')
    ax2.plot(lambda_list, alpha_green / lambda_list, color='#188C18')

    ax2.scatter(red_alpha['Growth Rate'].values,
                red_expression_level / red_alpha['Growth Rate'].values, color='#FE6508')
    ax2.scatter(green_alpha['Growth Rate'].values,
                green_expression_level / green_alpha['Growth Rate'].values, color='#188C18')

    ax2.set_xlabel('Growth rate ($h^{-1}$)')
    ax2.set_ylabel('$\widetilde{\\alpha}_{R, G}$')

    fig2.show()
