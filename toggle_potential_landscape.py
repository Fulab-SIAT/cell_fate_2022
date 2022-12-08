# -*- coding: utf-8 -*-

"""
 This script is used to calculate the probability potential landscape.
 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""
# %%
# Built-in/Generic Imports
import os
# […]

# Libs
import numpy as np  # Or any other
import matplotlib.pyplot as plt
import pandas as pd

from pde_solver import generate_adi_A  # generate the Matrices for ADI
from pde_solver_cpy import crate_init_p, evolve_ADI  # evaluating the PDEs via C compiled codes.
from joblib import Parallel, delayed, dump, load
from toggle_dynamic import ToggleBasic, assign_vars
from tqdm import tqdm
from typing import Optional
import datetime
from multiprocessing import Process


class TogglePotential:
    def __init__(self, toggle_pars=None, landscape_pars=None):
        self.gr = None  # type: Optional[float]
        self.laci_list = None  # type: Optional[np.ndarray]
        self.tetr_list = None  # type: Optional[np.ndarray]
        self.time_list = None  # type: Optional[np.ndarray]
        self.toggle_pars = toggle_pars
        self.landscape_pars = landscape_pars
        self.cov = None  # type: Optional[np.ndarray]
        self.mean = None  # type: Optional[np.ndarray]
        self.p0 = None  # type: Optional[np.ndarray]

        self.D_1 = None  # type: Optional[float]
        self.D_2 = None  # type: Optional[float]

        self.A_1 = None  # type: Optional[list[np.ndarray]]
        self.A_2 = None  # type: Optional[list[np.ndarray]]
        self.b_1 = None  # type: Optional[list[np.ndarray]]
        self.b_2 = None  # type: Optional[list[np.ndarray]]
        self.p_t = None  # type: Optional[np.ndarray]
        self.laci_step = None  # type: Optional[np.ndarray]
        self.tetr_step = None  # type: Optional[float]
        self.time_step = None  # type: Optional[float]
        self.toggle = ToggleBasic()
        if toggle_pars is not None:
            assign_vars(self.toggle, self.toggle_pars)
        if landscape_pars is not None:
            assign_vars(self, self.landscape_pars)

    def cef_A(self, X, t=None):
        """
        \nabla(f(X))
        f_1(X)
        """
        x1, x2 = X
        return - 2. * self.toggle.growth_rate

    def cef_B(self, X, t=None):
        """
        f_1(X)
        """
        # x1, x2 = X
        return self.toggle.field_flow(X)[0]

    def cef_C(self, X, t=None):
        """
        f_2(X)
        """
        # x1, x2 = X
        return self.toggle.field_flow(X)[1]

    def cef_D_1(self, X, t=None):
        # x1, x2 = X
        return self.D_1

    def cef_D_2(self, X, t=None):
        # x1, x2 = X
        return self.D_2

    def start_evolution(self, save_time_step=1):
        x1_v, x2_v = np.meshgrid(self.laci_list, self.tetr_list, indexing='ij')
        self.p0 = np.array(
            Parallel(n_jobs=-1, require='sharedmem')(delayed(crate_init_p)(x1_v[i], x2_v[i], self.cov, self.mean)
                                                     for i in tqdm(range(len(x1_v)))))
        self.p0[:, 0] = 0.
        self.p0[0, :] = 0.

        self.laci_step = self.laci_list[1] - self.laci_list[0]
        self.tetr_step = self.tetr_list[1] - self.tetr_list[0]
        self.time_step = self.time_list[1] - self.time_list[0]
        self.toggle.growth_rate = self.gr
        self.A_1, self.A_2, self.b_1, self.b_2 = generate_adi_A(self.cef_A, self.cef_B, self.cef_C,
                                                                self.cef_D_1, self.cef_D_2,
                                                                self.laci_list, self.tetr_list,
                                                                self.laci_step, self.tetr_step, self.time_step)

        self.p_t = evolve_ADI(self.time_list, self.p0, save_time_step, self.A_1, self.A_2, self.b_1, self.b_2)

    def save_rets(self, ps=None, keys=None):
        if keys is None:
            keys = ['laci_list', 'tetr_list', 'p_t', 'D_1', 'D_2', 'time_step',
                    'toggle_pars', 'landscape_pars']
        save_dict = dict()
        for key, data in self.__dict__.items():
            if key in keys:
                save_dict[key] = data
        if ps is None:
            time_now = datetime.datetime.now().strftime('%Y_%h_%d_%H')
            file_index = 1
            while True:
                file_name = f'{time_now}_{file_index}.bin'
                if file_name in [file.name for file in os.scandir(r'./data') if file.name.split('.')[-1] == 'bin']:
                    file_index += 1
                else:
                    ps = os.path.join('./data', file_name)
                    break
        print(save_dict['toggle_pars'])
        print(save_dict['landscape_pars']['gr'])
        print(ps)
        dump(save_dict, ps)


def simulation_process(toggle_args=None, potential_args=None, save_dir=None):
    toggle_potential = TogglePotential(toggle_pars=toggle_args, landscape_pars=potential_args)
    toggle_potential.start_evolution()
    if save_dir is None:
        toggle_potential.save_rets()
    else:
        toggle_potential.save_rets(save_dir)
    return None


def find_index(array, value):
    diff_value = np.abs(array - value)
    index = np.argmin(diff_value)
    return index


# %%
if __name__ == '__main__':
    # %%
    toggle = ToggleBasic()
    toggle_pars = {'k_t': 9.0,
                   'k_l': 12.6,
                   'tau_p_ltet': 0.015,
                   'tau_p_trc': 0.13,
                   'n_l': 4.0,
                   'n_t': 2.0,
                   'alphal_factor': 1.1,
                   'alphat_factor': 1.1}
    growth_rate = 1.4
    time_duration = 20
    grid_size = 1024
    D = 20
    x1 = np.linspace(0, 150, endpoint=True, num=grid_size)
    x2 = np.linspace(0, 100, endpoint=True, num=grid_size)
    cov = np.array([D, D]).reshape(-1, 1) * np.eye(2)

    # t_step = .9 * 4 * D / np.max(np.array(toggle.field_flow([x1, x2])) ** 2)
    t_step = 1/60.


    assign_vars(toggle, toggle_pars)
    toggle.growth_rate = growth_rate
    toggle.solve_sst(x1)
    mean = np.array([toggle.sst_laci_conc[-2], toggle.sst_tetr_conc[-2]]).reshape(-1, 1)
    x1_len, x2_len = len(x1), len(x2)
    x1_v, x2_v = np.meshgrid(x1, x2, indexing='ij')

    fixpoint_index = []
    for i in range(len(toggle.sst_laci_conc)):
        indexI = [find_index(x1, toggle.sst_laci_conc[i]), find_index(x2, toggle.sst_tetr_conc[i])]
        fixpoint_index.append(indexI)

    # p_0 = np.array(
    #     Parallel(n_jobs=-1)(delayed(crate_init_p)(x1_v[i], x2_v[i], cov, mean) for i in tqdm(range(len(x1_v)))))

    p_0 = crate_init_p(x1_v.flatten(), x2_v.flatten(), cov, mean).reshape(x1_v.shape)
    # set edge of initiation distribution to 0
    p_0[:, 0] = 0.
    p_0[0, :] = 0.
    p_0[:, -1] = 0.
    p_0[-1, :] = 0.
    time = np.arange(0, time_duration, t_step)


    fig1, ax1 = plt.subplots(1, 1, figsize=(20, 20))
    ax1.plot(x1, toggle.null_cline_tetr(x1), '--r')
    ax1.plot(toggle.null_cline_laci(x2), x2, '--g')
    ax1.imshow(p_0.T, origin='lower', cmap='coolwarm', extent=(x1.min(), x1.max(), x2.min(), x2.max()))
    ax1.set_xlim(x1[0], x1[-1])
    ax1.set_ylim(x2[0], x2[-1])
    ax1.set_xlabel('LacI Conc.')
    ax1.set_ylabel('TetR Conc.')
    fig1.show()

    potential_args = dict(laci_list=x1, tetr_list=x2, time_list=time, D_1=cov[0, 0] / 2, D_2=cov[1, 1] / 2, mean=mean,
                          cov=cov,
                          gr=growth_rate)

    toggle_potential = TogglePotential(toggle_pars=toggle_pars, landscape_pars=potential_args)
    toggle_potential.start_evolution(10 * t_step)

    diff = np.abs(np.diff(toggle_potential.p_t, axis=0))
    error = np.array([np.mean(diff[i]) for i in range(len(diff))])

    sst_index = np.argmin(error[error > 0])
    sst_potential = -np.log(toggle_potential.p_t[sst_index] + 1E-9)
    fixPotential = []
    for fixPoint in fixpoint_index:
        fixPotential.append(sst_potential[fixPoint[0], fixPoint[1]])

    fig2, ax2 = plt.subplots(1, 1, figsize=(20, 20))
    ax2.imshow(toggle_potential.p_t[35].T, origin='lower', cmap='coolwarm',
               extent=(x1.min(), x1.max(), x2.min(), x2.max()))
    ax2.set_xlim(x1[0], x1[-1])
    ax2.set_ylim(x2[0], x2[-1])
    ax2.set_xlabel('LacI Conc.')
    ax2.set_ylabel('TetR Conc.')
    fig2.show()

    data = pd.DataFrame(data=dict(lacI=x1_v.flatten(), tetR=x2_v.flatten(), potential=sst_potential.flatten()))

    data.to_csv(r'./data.csv')


    # %%
    data_ps = r'./data/2021_Nov_22_17_1.bin'
    strain_name = 'M5'
    potential_data_dict = load(data_ps)
    landscape_data = potential_data_dict['landscape_pars']
    toggle = ToggleBasic()

    p_t = potential_data_dict['p_t']
    x1 = potential_data_dict['laci_list']
    x2 = potential_data_dict['tetr_list']
    growth_rate = landscape_data['gr']
    print(growth_rate)
    toggle_pars = potential_data_dict['toggle_pars']
    assign_vars(toggle, toggle_pars)
    toggle.growth_rate = growth_rate
    x1_v, x2_v = np.meshgrid(x1, x2, indexing='ij')
    # %% Plot potential dynamics
    for index in tqdm(range(len(p_t))):
        if index % 1 == 0:
            fig, ax = plt.subplots(1, 1, figsize=(18, 18))

            ax.imshow(p_t[index, ...].T, cmap='coolwarm', origin='lower', extent=(x1[0], x1[-1], x2[0], x2[-1]),
                      aspect='auto')

            ax.plot(x1, toggle.null_cline_tetr(x1), '--r')
            ax.plot(toggle.null_cline_laci(x2), x2, '--g')
            ax.set_xlim(x1[0], x1[-1])
            ax.set_ylim(x2[0], x2[-1])
            ax.text(0, 8, s=f'Time: %i' % index)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            fig.show()
            # fig.savefig(os.path.join('./data', 'bistable', f'{index}.png'))
    # %% Plot Potential Landscape at Pss
    diff = [np.max((p_t[i + 1, ...] - p_t[i, ...]) ** 2) for i in range(len(p_t) - 1)]

    min_err_idx = np.argmin(diff)

    fig2, ax = plt.subplots(1, 1, figsize=(15, 15))

    potential = - np.log(p_t[5, ...] + 1.1 * np.abs(p_t[5, ...].min()))
    cmp = ax.imshow(potential.T, cmap='coolwarm', origin='lower', aspect='auto',
                    extent=(x1[0], x1[-1], x2[0], x2[-1]), interpolation='spline16', )
    ax.plot(x1, toggle.null_cline_tetr(x1), '--r')
    ax.plot(toggle.null_cline_laci(x2), x2, '--g')
    # ax.quiver(x_v, y_v, v_x, v_y, scale_units='x')
    ax.set_xlim(x1[0], x1[-1])
    ax.set_ylim(x2[0], x2[-1])
    ax.set_xlabel('LacI conc.', labelpad=30)
    ax.set_ylabel('TetR conc.', labelpad=30)
    ax.set_title(f'{strain_name}_{growth_rate}', pad=30)
    fig2.colorbar(cmp)
    fig2.show()

    # %% potential 3D
    import matplotlib.tri as mtri
    import sciplot as splt
    from scipy.ndimage import gaussian_filter

    blured_potental = gaussian_filter(potential, sigma=3)

    splt.whitegrid()
    fig2, ax2 = plt.subplots(subplot_kw={"projection": '3d'}, figsize=(15, 15))
    ax2.view_init(elev=35, azim=-135)

    mask = (x1_v < 1 * x1.max()) & (x2_v < 1 * x2.max())
    sparse_poiint = 12
    tri = mtri.Triangulation(x1_v[mask][::sparse_poiint].flatten(), x2_v[mask][::sparse_poiint].flatten())
    surf = ax2.plot_trisurf(x1_v[mask][::sparse_poiint].flatten(), x2_v[mask][::sparse_poiint].flatten(),
                            blured_potental[mask][::sparse_poiint].flatten(),
                            triangles=tri.triangles,
                            cmap='coolwarm')

    cset = ax2.contour(x1_v, x2_v, blured_potental, offset=potential.min() - 10,
                       zdir='z', cmap='coolwarm', linewidths=6)

    ax2.set_zlim(potential.min() - 10)
    ax2.set_xlim(0, 1 * x1.max())
    ax2.set_ylim(0, 1 * x2.max())
    ax2.xaxis.set_pane_color((1, 1, 1, 0))
    ax2.yaxis.set_pane_color((1, 1, 1, 0))
    ax2.zaxis.set_pane_color((1, 1, 1, 0))
    ax2.set_title(f'{strain_name}_{growth_rate}')
    fig2.colorbar(surf, shrink=0.5)
    ax2.set_xlabel('LacI conc.', labelpad=30)
    ax2.set_ylabel('TetR conc.', labelpad=45)
    ax2.tick_params(axis='z', which='major', pad=20)
    ax2.tick_params(axis='y', which='major', pad=15)
    fig2.show()



