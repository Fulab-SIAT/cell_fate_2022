"""
@author: CHU Pan
@Date: 20220427
"""

import pandas as pd
import numpy as np  # Or any other
from toggle_dynamic import ToggleBasic, assign_vars
from joblib import Parallel, delayed 
from tqdm import tqdm

import matplotlib.pyplot as plt
# from matplotlib import cm
# import matplotlib.colors as colors
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from sub_model.emperical_gene_expression_rate import mRNA_level
from sub_model.gene_expression_rate import protein_trans_rate, phi_ribosome, frc_act_ribo
import sciplot as splt
from phase_plot_dyn_toggle import bistability_titrate_al_ricip_k, get_toggle_tilde_k, white_grey_bar

splt.whitegrid()


#%%
summary_data = pd.read_excel(r'./sub_model/Experimental_data_summary_for_steday_state.xlsx',
                             sheet_name='Summary Data')
growth_rate = summary_data['Growth_rate']
transcript_level = summary_data['mRNA_number_fraction']

lac_data_filter = np.logical_and(summary_data['Strain'] == 'NH2.23',
                                 ~np.isnan(summary_data['mRNA_number_fraction']))
tet_data_filter = np.logical_and(summary_data['Strain'] == 'NH2.24',
                                 ~np.isnan(summary_data['mRNA_number_fraction']))

# %%
sim_gr = np.linspace(0.194, 1.651)
laci_mRNA_arg = [18.90e+04,  1.8e0, 1.1e1, .66, .98, 4.2, 1.9]

tetr_mRNA_arg = [8.8e+04, 9.08e+00, 3.32e+01, 5.43e-01, 3.3e-01, 3.1e+00, 1.8e00]

sim_lac_mRNA = mRNA_level(sim_gr, args=laci_mRNA_arg)
sim_tetr_mRNA = mRNA_level(sim_gr, args=tetr_mRNA_arg)

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

#%%
gr_std_index = np.argmin(np.abs(sim_gr - 1.))
sim_lac_alpha = sim_lac_mRNA * phi_ribosome(sim_gr) * protein_trans_rate(sim_gr) * frc_act_ribo(sim_gr)
sim_tetr_alpha = sim_tetr_mRNA * phi_ribosome(sim_gr) * protein_trans_rate(sim_gr) * frc_act_ribo(sim_gr)

fig2, ax2 = plt.subplots(1, 1)

ax2.plot(sim_gr, sim_lac_alpha / sim_lac_alpha[gr_std_index], '*-', c='#1d8e1d')
ax2.plot(sim_gr, sim_tetr_alpha / sim_tetr_alpha[gr_std_index], '*-', c='#ff6000')
ax2.set_xlim((0, 2))
splt.aspect_ratio(1)
ax2.set_xlabel('Growth rate ($h^{-1}$)')
ax2.set_ylabel('Expression rate \n($\\alpha_{G,R}$)')
fig2.show()

df_export = pd.DataFrame(data=dict(gr=sim_gr,
                                   alpha_g=sim_lac_alpha / sim_lac_alpha[gr_std_index],
                                   alpha_r=sim_tetr_alpha / sim_tetr_alpha[gr_std_index]))

translation_level = summary_data['Protein expression']
lac_p_data_filter = np.logical_and(summary_data['Strain'] == 'NH2.23',
                                   ~np.isnan(summary_data['Protein expression']))
tet_p_data_filter = np.logical_and(summary_data['Strain'] == 'NH2.24',
                                   ~np.isnan(summary_data['Protein expression']))

# MOPS Glu. Green 0.884327; Red 0.968006
# RDM Glu. Green 1.525330; Red 1.587604

# sim_norm_laci_exp = sim_lac_alpha / sim_gr / (sim_lac_alpha / sim_gr)[np.argmin(np.abs(sim_gr - 1.525330))]
# sim_norm_tetr_exp = sim_tetr_alpha / sim_gr / (sim_tetr_alpha / sim_gr)[np.argmin(np.abs(sim_gr - 1.587604))]

sim_norm_laci_exp = sim_lac_alpha / sim_gr * 10**-2 * .9
sim_norm_tetr_exp = sim_tetr_alpha / sim_gr * 10**-2 * .5

fig3, ax3 = plt.subplots(1, 1)
# ax3.scatter(growth_rate[lac_p_data_filter][:-1], (translation_level[lac_p_data_filter][:-1] /
#                                                   translation_level[lac_p_data_filter].iloc[0]), c='#1d8e1d')

# ax3.scatter(growth_rate[tet_p_data_filter][:-1],
#             (translation_level[tet_p_data_filter][:-1] / translation_level[tet_p_data_filter].iloc[0]), c='#ff6000')

ax3.scatter(growth_rate[lac_p_data_filter][:-1],
            translation_level[lac_p_data_filter][:-1], c='#1d8e1d')

ax3.scatter(growth_rate[tet_p_data_filter][:-1],
            translation_level[tet_p_data_filter][:-1], c='#ff6000')


ax3.plot(sim_gr, sim_norm_laci_exp, '--', c='#1d8e1d')
ax3.plot(sim_gr, sim_norm_tetr_exp, '--', c='#ff6000')
splt.aspect_ratio(1)
ax3.set_xlabel('Growth rate (1/h)')
ax3.set_ylabel('Expression level \n(a.u., Normalized)')
fig3.show()


#%%
toggle = ToggleBasic()

factor = 9
kg = 0.4

toggle_pars = dict(k_t=factor, k_l=factor * kg, tau_p_ltet=0.002, tau_p_trc=0.035, n_l=2, n_t=4)
# toggle_pars = dict(k_t=factor, k_l=factor * kg, tau_p_ltet=0.002, tau_p_trc=0.035/0.021*0.0084, n_l=2, n_t=4)
# toggle_pars = dict(k_t=factor, k_l=factor * kg, tau_p_ltet=0.002, tau_p_trc=0.035/0.021*0.0042, n_l=2, n_t=4)


phase_portrait = pd.read_csv('./data/M5_L3_phase_diagram.csv')

green_ratio_gr = phase_portrait['Growth_rate']
growth_rate_range = np.linspace(0.2, 1.65)
dist_growth_sim = np.ones((len(green_ratio_gr), 1)) * growth_rate_range.reshape((1, -1))
dist_growth_sim = (dist_growth_sim - green_ratio_gr.values.reshape((-1, 1))) ** 2
exp_growth_index = np.argmin(dist_growth_sim, axis=1)
assign_vars(toggle, toggle_pars)

ricip_kt = np.logspace(-1, 3, num=100)
ricip_kl = np.logspace(-1, 4, num=100)
extent = [ricip_kl[0], ricip_kl[-1], ricip_kt[0], ricip_kt[-1]]
al_ricip_kl, al_ricip_kt = np.meshgrid(ricip_kl, ricip_kt)


sst_state = Parallel(n_jobs=64)(delayed(bistability_titrate_al_ricip_k)([rec_t, rec_l], **toggle_pars)
                                for rec_t, rec_l in tqdm(zip(al_ricip_kt.flatten(), al_ricip_kl.flatten())))
print('Bistable sum: ', np.sum(sst_state))
sst_state = np.array(sst_state).reshape(al_ricip_kt.shape)

tilde_k = np.array([get_toggle_tilde_k(toggle, toggle_pars, gr) for gr in growth_rate_range])
recip_tilde_k = np.ones(tilde_k.shape) / tilde_k

c = np.zeros(sst_state.shape)
c[sst_state] = 1.
# Plot the trajectory of growth dependent alpha in phase diagram of toggle.
fig1, ax = plt.subplots(1, 2, figsize=(16, 8))
ax1 = ax[0]
ax1.pcolormesh(ricip_kl, ricip_kt, c[:-1, :-1], cmap=white_grey_bar, alpha=0.3)
ct = ax1.contour(ricip_kl, ricip_kt, c, cmap='coolwarm', linestyles='dotted', levels=[.5],
                 extent=[ricip_kl[0], ricip_kl[-1], ricip_kt[0], ricip_kt[-1]])

ax1.set_ylabel('$1/\\widetilde{K}_{\mathrm{DR}} (\\widetilde{\\alpha}_{\mathrm{R}}/K_{\mathrm{DR}})$')
ax1.set_xlabel('$1/\\widetilde{K}_{\mathrm{DG}} (\\widetilde{\\alpha}_{\mathrm{G}}/K_{\mathrm{DG}})$')
ax1.set_title(f'''n:{toggle_pars.get('n_t')}; tau: {toggle_pars.get('tau_p_ltet')}''', pad=20)
ax1.plot(recip_tilde_k[:, 0], recip_tilde_k[:, 1], '--', label='L1', c='#41F0AE')
ax1.plot(recip_tilde_k[:, 0] / (5 / 10.5), recip_tilde_k[:, 1], '--', label='L2', c='#41F0AE')
ax1.plot(recip_tilde_k[:, 0] / (2 / 10.5), recip_tilde_k[:, 1], '--', label='L3', c='#41F0AE')

move_line = ax1.scatter(recip_tilde_k[:, 0][exp_growth_index], recip_tilde_k[:, 1][exp_growth_index],
                        c=green_ratio_gr, cmap='coolwarm', s=500)

colbar = plt.colorbar(move_line, cax=ax[-1])
ax1.set_xlim(1, 6000)
ax1.set_ylim(1, 400)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks([], minor=True)
ax1.set_yticks([], minor=True)
splt.aspect_ratio(1, ax1)
ax1.grid(False)
splt.aspect_ratio(10, ax[-1])
fig1.show()