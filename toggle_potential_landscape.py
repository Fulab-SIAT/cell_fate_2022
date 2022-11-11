# -*- coding: utf-8 -*-

"""
 This script is used to calculate the probability potential landscape.
 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports
import os
# […]

# Libs
import numpy as np  # Or any other
import matplotlib.pyplot as plt
from pde_solver import generate_adi_A  # generate the Matrices for ADI
from toggle_mini.pde_solver_cpy import crate_init_p, evolve_ADI  # evaluating the PDEs via C compiled codes.
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

    def start_evolution(self):
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

        self.p_t = evolve_ADI(self.time_list, self.p0, 1., self.A_1, self.A_2, self.b_1, self.b_2)

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
        dump(save_dict, ps)


def simulation_process(toggle_args=None, potential_args=None):
    toggle_potential = TogglePotential(toggle_pars=toggle_args, landscape_pars=potential_args)
    toggle_potential.start_evolution()
    toggle_potential.save_rets()
    return None


if __name__ == '__main__':
# %%
    toggle = ToggleBasic()
    toggle_pars = dict(k_t=3.6, k_l=6.67, tau_p_t=0.02, tau_p_l=0.002, n_t=2, n_l=4, alphal_factor=1, alphat_factor=1)
    growth_rate = 0.8
    time_duration = 24
    t_step = 0.01
    x1 = np.linspace(0, 350, endpoint=True, num=512)
    x2 = np.linspace(0, 160, endpoint=True, num=512)
    cov = np.array([20, 20]).reshape(-1, 1) * np.eye(2)

    x1_step = x1[1] - x1[0]
    x2_step = x1[1] - x1[0]
    toggle.growth_rate = growth_rate
    assign_vars(toggle, toggle_pars)
    toggle.solve_sst(x1)
    mean = np.array([toggle.sst_laci_conc[-2], toggle.sst_tetr_conc[-2]]).reshape(-1, 1)
    # mean = np.array([32.07, 0.53]).reshape(-1, 1)
    x1_len, x2_len = len(x1), len(x2)
    x1_v, x2_v = np.meshgrid(x1, x2, indexing='ij')
    p_0 = np.array(Parallel(n_jobs=-1)(delayed(crate_init_p)(x1_v[i], x2_v[i], cov, mean) for i in tqdm(range(len(x1_v)))))

    p_0[:, 0] = 0.
    p_0[0, :] = 0.
    p_0[:, -1] = 0.
    p_0[-1, :] = 0

    time = np.arange(0, time_duration, t_step)

    fig2, ax = plt.subplots(1, 1, figsize=(20, 20))

    ax.plot(x1, toggle.null_cline_tetr(x1), '--r')
    ax.plot(toggle.null_cline_laci(x2), x2, '--g')
    ax.imshow(p_0.T, origin='lower', cmap='coolwarm', extent=(x1.min(), x1.max(), x2.min(), x2.max()))
    ax.set_xlim(x1[0], x1[-1])
    ax.set_ylim(x2[0], x2[-1])
    ax.set_xlabel('LacI Conc.')
    ax.set_ylabel('TetR Conc.')
    fig2.show()


    potential_args = dict(laci_list=x1, tetr_list=x2, time_list=time, D_1=cov[0, 0]/2, D_2=cov[1, 1]/2, mean=mean, cov=cov,
                          gr=growth_rate)

    # toggle_potential = TogglePotential(toggle_pars=toggle_pars, landscape_pars=potential_args)
    # toggle_potential.start_evolution()

    p = Process(target=simulation_process, args=(toggle_pars, potential_args))
    p.start()




    #%%
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
            # ax.set_xscale('log')
            # ax.set_yscale('log')

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
    fig2.savefig(f'./data/{strain_name}_{growth_rate}_potential_2d.png', transparent=True)
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
    # nuclines
    # ax2.plot(x1, toggle.null_cline_tetr(x1), '--r', zs=potential.min() - 50, zdir='z')
    # ax2.plot(toggle.null_cline_laci(x2), x2, '--g', zs=potential.min() - 50, zdir='z')
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
    fig2.savefig(f'./data/{strain_name}_{growth_rate}_potential_3d.png', transparent=True)


    # %% marginal probability
    p_ssx1 = np.mean(p_t[-1], axis=1)
    p_ssx2 = np.mean(p_t[-1], axis=0)

    u_ssx1 = -np.log(p_ssx1 + 1.001 * np.abs(p_ssx1.min()))
    u_ssx2 = -np.log(p_ssx2 + 1.001 * np.abs(p_ssx2.min()))
    min_u = np.min([u_ssx1.min(), u_ssx2.min()])
    max_u = np.max([u_ssx1, u_ssx2])
    fig3, ax3 = plt.subplots(2, 1, figsize=(15, 15))
    ax3 = ax3.flatten()
    ax3[0].plot(x1[::sparse_poiint], u_ssx1[::sparse_poiint], 'g')
    ax3[0].set_xlabel('LacI conc.')
    ax3[0].set_ylabel('$U$')
    # ax3.set_yscale('log')
    ax3[1].plot(x2[::sparse_poiint], u_ssx2[::sparse_poiint], 'r')
    ax3[1].set_xlabel('TetR conc.')
    ax3[1].set_ylabel('$U$')
    # ax3.set_xscale('log')
    # ax3.plot(x1, p_ssx1)
    ax3[0].set_ylim(min_u - 0.2*np.abs(min_u), max_u + 0.2*np.abs(max_u))
    ax3[1].set_ylim(min_u - 0.2*np.abs(min_u), max_u + 0.2*np.abs(max_u))
    ax3[0].set_title(f'{strain_name}_{growth_rate}', pad=30)
    fig3.show()
    fig3.savefig(f'./data/{strain_name}_{growth_rate}_potential_single_axis.png', transparent=True)

    # %%
    #
    # fig4, ax4 = plt.subplots(1, 1)
    # ax4.plot(range(len(diff)), diff)
    # # ax4.set_yscale('log')
    # ax4.set_xlabel('time')
    # ax4.set_ylabel('$<\Delta X^{2}> $')
    # fig4.show()


    # %% define
    # def cef_A(X, t=None):
    #     """
    #     \nabla(f(X))
    #     f_1(X)
    #     """
    #     x1, x2 = X
    #     return - 2. * toggle.growth_rate
    #
    #
    # def cef_B(X, t=None):
    #     """
    #     f_1(X)
    #     """
    #     # x1, x2 = X
    #     return toggle.field_flow(X)[0]
    #
    #
    # def cef_C(X, t=None):
    #     """
    #     f_2(X)
    #     """
    #     # x1, x2 = X
    #     return toggle.field_flow(X)[1]
    #
    #
    # def cef_D_1(X, t=None):
    #     x1, x2 = X
    #     return 1
    #
    #
    # def cef_D_2(X, t=None):
    #     x1, x2 = X
    #     return 1
    #
    #
    # v_x1, v_x2 = cef_B([x1_v, x2_v]), cef_C([x1_v, x2_v])
    #
    # #
    # # p_t = fokker_planck_pde_solver(cef_A, cef_B, cef_C, cef_D_1, cef_D_2, x1, x2, time, x1_step, x2_step, t_step, p_0,
    # #                                parallel=False)
    # A_1, A_2, b_1, b_2 = generate_adi_A(cef_A, cef_B, cef_C, cef_D_1, cef_D_2, x1, x2, x1_step, x2_step, t_step)
    #
    # p_t = evolve_ADI(time, p_0, 1., A_1, A_2, b_1, b_2)
