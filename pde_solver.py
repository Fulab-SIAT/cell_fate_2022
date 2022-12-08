# -*- coding: utf-8 -*-

"""

 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""

# Built-in/Generic Imports


# Libs

import numpy as np  # Or any other
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
# […]
import matplotlib.pyplot as plt
from typing import Callable
from joblib import Parallel, delayed
import cython


def multi_var_guassian(x: np.ndarray, cov: np.ndarray, mean: np.ndarray) -> cython.double:
    """
    Parameters
    ----------
    x:
        ndarray 2 X 1
    cov:
        ndarray 2 X 2
    mean:
        ndarray 2 X 1
    """
    x -= mean
    return (1. / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov))) *
            np.exp(-(x.T.dot(np.linalg.solve(cov, x))) / 2))[0][0]


def crate_init_p(x1: np.ndarray, x2: np.ndarray, cov: np.ndarray, mean: np.ndarray) -> np.ndarray:
    length_x: cython.int
    length_x = len(x1)
    p0 = [multi_var_guassian(np.array([[x1[x_index]], [x2[x_index]]]), cov, mean)
          for x_index in tqdm(range(length_x))]
    return np.array(p0)


def generate_adi_A(cef_A: Callable, cef_B: Callable, cef_C: Callable, cef_D_1: Callable, cef_D_2: Callable,
                   x1: np.ndarray, x2: np.ndarray, x1_step: float, x2_step: float, t_step: float):
    """
    Generating Matrices for ADI, solving FP equation.

    Parameters
    ----------- vb 
    cef_A: Callable
        coefficient A, \babla(f(X))
    cef_B: Callable
        coefficient B, f(x1)
    cef_C: Callable
        coefficient C, f(x2)
    cef_D_1: Callable
        coefficient D1, diffusion D(x1)
    cef_D_2: Callable
        coefficient D2, diffusion D(x2)
    x1； array-like
        dimension 1
    x2: array-like
        dimension 2
    x1_step: float
        the step length of x1
    x2_step: float
        the step length of x2
    t_step: float
        the step length of time

    Returns
    --------
    A_1: list
    A_2: list
    b_1: list
    d_2: list
    """

    x1_len: cython.int
    x2_len: cython.int
    index: cython.Py_tss_t
    x1_i: cython.double
    x2_i: cython.double
    x1_len, x2_len = len(x1), len(x2)

    # x1_v, x2_v = np.meshgrid(x1, x2, indexing='ij')

    A_1 = [None] * x2_len
    b_1 = [None] * x1_len
    A_2 = [None] * x1_len
    b_2 = [None] * x2_len

    p1, p2, q1, q2 = t_step / (4 * x1_step), t_step / (4 * x2_step), t_step / (2 * x1_step ** 2), t_step / (
            2 * x2_step ** 2)

    print('Generate matrix A')
    for index, x1_i in enumerate(x1):
        poss = [[x1_i, x2_i] for x2_i in x2]  # positions along x2 direction
        b_main_1 = np.array([1. - 2 * q2 * cef_D_2(pos) - cef_A(pos) * t_step / 2 for pos in poss])
        b_lower_1 = np.array([p2 * cef_C(pos) + q2 * cef_D_2(pos) for pos in poss])
        b_upper_1 = np.array([- p2 * cef_C(pos) + q2 * cef_D_2(pos) for pos in poss])
        # b_1_boundary = np.array([2 * q2 * cef_D_2(pos) for pos in poss])
        b_lower_1[-1] = b_lower_1[-1] + b_upper_1[-1]
        b_upper_1[0] = b_lower_1[0] + b_upper_1[0]
        # b_1 matrix have a x2 X x2 size
        b_1[index] = sparse.diags([b_lower_1[1:], b_main_1, b_upper_1[:-1]], [-1, 0, 1], format='csr')

    for index, x2_i in enumerate(x2):
        poss = [[x1_i, x2_i] for x1_i in x1]
        A_main_1 = np.array([1. + 2 * q1 * cef_D_1(pos) for pos in poss])
        A_lower_1 = np.array([- p1 * cef_B(pos) - q1 * cef_D_1(pos) for pos in poss])
        A_upper_1 = np.array([p1 * cef_B(pos) - q1 * cef_D_1(pos) for pos in poss])
        # A_1_boundary = np.array([-2 * q1 * cef_D_1(pos) for pos in poss])
        A_upper_1[0] = A_upper_1[0] + A_lower_1[0]
        A_lower_1[-1] = A_upper_1[-1] + A_lower_1[-1]
        # A_1 matrix have x1 X x1 size
        A_1[index] = sparse.diags([A_lower_1[1:], A_main_1, A_upper_1[:-1]], [-1, 0, 1], format='csr')

    for index, x2_i in enumerate(x2):
        poss = [[x1_i, x2_i] for x1_i in x1]
        b_main_2 = np.array([1 - 2 * q1 * cef_D_1(pos) - cef_A(pos) * t_step / 2 for pos in poss])
        b_lower_2 = np.array([q1 * cef_D_1(pos) + p1 * cef_B(pos) for pos in poss])
        b_upper_2 = np.array([q1 * cef_D_1(pos) - p1 * cef_B(pos) for pos in poss])
        # b_2_boundary = np.array([2 * q1 * cef_D_1(pos) for pos in poss])
        b_upper_2[0] = b_upper_2[0] + b_lower_2[0]
        b_lower_2[-1] = b_upper_2[-1] + b_lower_2[-1]
        # b2 have x1 X x1 size
        b_2[index] = sparse.diags([b_lower_2[1:], b_main_2, b_upper_2[:-1]], [-1, 0, 1], format='csr')

    for index, x1_i in enumerate(x1):
        poss = [[x1_i, x2_i] for x2_i in x2]
        A_lower_2 = np.array([- p2 * cef_C(pos) - q2 * cef_D_2(pos) for pos in poss])
        A_main_2 = np.array([1 + 2 * q2 * cef_D_2(pos) for pos in poss])
        A_upper_2 = np.array([p2 * cef_C(pos) - q2 * cef_D_2(pos) for pos in poss])
        # A_boundary_2 = np.array([-2 * q2 * cef_D_2(pos) for pos in poss])
        A_lower_2[-1] = A_lower_2[-1] + A_upper_2[-1]
        A_upper_2[0] = A_lower_2[0] + A_upper_2[0]
        # A_2 have x2 X x2 size
        A_2[index] = sparse.diags([A_lower_2[1:], A_main_2, A_upper_2[:-1]], [-1, 0, 1], format='csr')

    return A_1, A_2, b_1, b_2


def fokker_planck_pde_solver(cef_A: Callable, cef_B: Callable, cef_C: Callable, cef_D_1: Callable, cef_D_2: Callable,
                             x1: np.ndarray, x2: np.ndarray, time: np.ndarray,
                             x1_step: float, x2_step: float, t_step: float,
                             p_0: np.ndarray, save_step=1.,
                             parallel=False) -> np.ndarray:
    x1_len: cython.int
    x2_len: cython.int
    index: cython.int
    x1_i: cython.double
    x2_i: cython.double
    t: cython.int

    x1_len, x2_len = len(x1), len(x2)
    # x1_v, x2_v = np.meshgrid(x1, x2, indexing='ij')
    t_for_save = [t_i for t_i in time if t_i % save_step == 0.]
    p_t = np.zeros((len(t_for_save), x1_len, x2_len))

    A_1 = [None] * x2_len
    b_1 = [None] * x1_len
    A_2 = [None] * x1_len
    b_2 = [None] * x2_len

    p1, p2, q1, q2 = t_step / (4 * x1_step), t_step / (4 * x2_step), t_step / (2 * x1_step ** 2), t_step / (
            2 * x2_step ** 2)

    print('Generate matrix A')
    for index, x1_i in enumerate(x1):
        poss = [[x1_i, x2_i] for x2_i in x2]  # positions along x2 direction
        b_main_1 = np.array([1. - 2 * q2 * cef_D_2(pos) - cef_A(pos) * t_step / 2 for pos in poss])
        b_lower_1 = np.array([p2 * cef_C(pos) + q2 * cef_D_2(pos) for pos in poss])[1:]
        b_upper_1 = np.array([- p2 * cef_C(pos) + q2 * cef_D_2(pos) for pos in poss])[:-1]
        b_1_boundary = np.array([2 * q2 * cef_D_2(pos) for pos in poss])
        b_lower_1[-1] = b_1_boundary[-1]
        b_upper_1[0] = b_1_boundary[0]
        # b_1 matrix have a x2 X x2 size
        b_1[index] = sparse.diags([b_lower_1, b_main_1, b_upper_1], [-1, 0, 1], format='csr')

    for index, x2_i in enumerate(x2):
        poss = [[x1_i, x2_i] for x1_i in x1]
        A_main_1 = np.array([1. + 2 * q1 * cef_D_1(pos) for pos in poss])
        A_lower_1 = np.array([- p1 * cef_B(pos) - q1 * cef_D_1(pos) for pos in poss])[1:]
        A_upper_1 = np.array([p1 * cef_B(pos) - q1 * cef_D_1(pos) for pos in poss])[:-1]
        A_1_boundary = np.array([-2 * q1 * cef_D_1(pos) for pos in poss])
        A_upper_1[0] = A_1_boundary[0]
        A_lower_1[-1] = A_1_boundary[-1]
        # A_1 matrix have x1 X x1 size
        A_1[index] = sparse.diags([A_lower_1, A_main_1, A_upper_1], [-1, 0, 1], format='csr')

    for index, x2_i in enumerate(x2):
        poss = [[x1_i, x2_i] for x1_i in x1]
        b_main_2 = np.array([1 - 2 * q1 * cef_D_1(pos) - cef_A(pos) * t_step / 2 for pos in poss])
        b_lower_2 = np.array([q1 * cef_D_1(pos) + p1 * cef_B(pos) for pos in poss])[1:]
        b_upper_2 = np.array([q1 * cef_D_1(pos) - p1 * cef_B(pos) for pos in poss])[:-1]
        b_2_boundary = np.array([2 * q1 * cef_D_1(pos) for pos in poss])
        b_upper_2[0] = b_2_boundary[0]
        b_lower_2[-1] = b_2_boundary[-1]
        # b2 have x1 X x1 size
        b_2[index] = sparse.diags([b_lower_2, b_main_2, b_upper_2], [-1, 0, 1], format='csr')

    for index, x1_i in enumerate(x1):
        poss = [[x1_i, x2_i] for x2_i in x2]
        A_lower_2 = np.array([- p2 * cef_C(pos) - q2 * cef_D_2(pos) for pos in poss])[1:]
        A_main_2 = np.array([1 + 2 * q2 * cef_D_2(pos) for pos in poss])
        A_upper_2 = np.array([p2 * cef_C(pos) - q2 * cef_D_2(pos) for pos in poss])[:-1]
        A_boundary_2 = np.array([-2 * q2 * cef_D_2(pos) for pos in poss])
        A_lower_2[-1] = A_boundary_2[-1]
        A_upper_2[0] = A_boundary_2[0]
        # A_2 have x2 X x2 size
        A_2[index] = sparse.diags([A_lower_2, A_main_2, A_upper_2], [-1, 0, 1], format='csr')

    p_tilde = np.zeros((x1_len, x2_len))
    u_1 = np.zeros((x2_len, x1_len))
    u_2 = np.zeros((x1_len, x2_len))

    # sweep along x1 axis get b1 matrix
    print('Start evolution.')
    t_current = p_0
    save_index = 0
    if not parallel:
        for t in tqdm(range(len(time) - 1)):
            if time[t] % save_step == 0.:
                p_t[save_index] = t_current.copy()
                save_index += 1
            # p = p_t[t]
            for index in range(x1_len):
                # u_1[:, index] = b_1[index].dot(p_t[t][index, :].reshape(-1, 1)).flatten()
                u_1[:, index] = b_1[index].dot(t_current[index, :])

            for index in range(x2_len):
                # p_tilde[:, index] = spsolve(A_1[index], u_1[index, :].reshape(-1, 1))
                p_tilde[:, index] = spsolve(A_1[index], u_1[index, :])

            for index in range(x2_len):
                # u_2[:, index] = b_2[index].dot(p_tilde[:, index].reshape(-1, 1)).flatten()
                u_2[:, index] = b_2[index].dot(p_tilde[:, index])

            for index in range(x1_len):
                # p_t[t + 1, index, :] = spsolve(A_2[index], u_2[index, :].reshape(-1, 1))
                t_current[index, :] = spsolve(A_2[index], u_2[index, :])

    else:
        def parl_u1(index):
            u_1[:, index] = b_1[index].dot(t_current[index, :])
            return None

        def parl_p_tilde(index):
            p_tilde[:, index] = spsolve(A_1[index], u_1[index, :])
            return None

        def parl_u2(index):
            u_2[:, index] = b_2[index].dot(p_tilde[:, index])
            return None

        def parl_pt(index):
            t_current = spsolve(A_2[index], u_2[index, :])
            return None

        for t in tqdm(range(len(time) - 1)):
            if time[t] % save_step == 0.:
                p_t[save_index] = t_current.copy()
                save_index += 1
            # p = p_t[t]
            # for index, x1_i in enumerate(x1):
            _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(parl_u1)(index) for index, x1_i in enumerate(x1))

            # for index, x2_i in enumerate(x2):
            #     p_tilde[:, index] = spsolve(A_1[index], u_1[index, :].reshape(-1, 1))
            _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(parl_p_tilde)(index) for index, x2_i in enumerate(x2))

            # for index, x2_i in enumerate(x2):
            #     u_2[:, index] = b_2[index].dot(p_tilde[:, index].reshape(-1, 1)).flatten()
            _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(parl_u2)(index) for index, x2_i in enumerate(x2))

            # for index, x1_i in enumerate(x1):
            #     p_t[t + 1, index, :] = spsolve(A_2[index], u_2[index, :].reshape(-1, 1))
            _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(parl_pt)(index) for index, x1_i in enumerate(x1))
    return p_t


if __name__ == '__main__':
    # %%
    from toggle_dynamic import ToggleBasic, assign_vars

    toggle = ToggleBasic()

    toggle_pars = dict(k_t=12, k_l=18, tau_p_t=0.008, tau_p_l=0.05, n_t=2, n_l=1.414)

    assign_vars(toggle, toggle_pars)

    toggle.growth_rate = 0.18

    # %%
    parallel = False
    eta = 0.2
    omega = 1.
    gamma = 0.1
    time_duration = 10
    cov = 50 * np.eye(2)
    mean = np.array([25, 60]).reshape(-1, 1)
    t_step = 0.01
    x1_step = 1
    x2_step = 1
    x1 = np.arange(0, 500, x1_step)
    x2 = np.arange(0, 500, x2_step)

    fig2, ax = plt.subplots(1, 1, figsize=(20, 20))

    ax.plot(x1, toggle.null_cline_tetr(x1), '--r')
    ax.plot(toggle.null_cline_laci(x2), x2, '--g')

    ax.set_xlim(x1[0], x1[-1])
    ax.set_ylim(x2[0], x2[-1])
    ax.set_xlabel('LacI Conc.')
    ax.set_ylabel('TetR Conc.')
    fig2.show()
    # %%
    # initial stats matrix
    x1_len, x2_len = len(x1), len(x2)
    x1_v, x2_v = np.meshgrid(x1, x2, indexing='ij')
    p_0 = Parallel(n_jobs=-1)(delayed(multi_var_guassian)(np.array([i, j]).reshape(-1, 1), cov, mean)
                              for index, (i, j) in
                              enumerate(tqdm(zip(x1_v.flatten(), x2_v.flatten()))))
    # initialize time scale and probability matrix
    time = np.arange(0, time_duration, t_step)
    p_0 = np.array(p_0).reshape((x1_len, x2_len))
    p_t = np.zeros((len(time), x1_len, x2_len))
    p_t[0, ...] = p_0

    fig2, ax = plt.subplots(1, 1, figsize=(20, 20))

    ax.plot(x1, toggle.null_cline_tetr(x1), '--r')
    ax.plot(toggle.null_cline_laci(x2), x2, '--g')
    ax.imshow(p_0, origin='lower', cmap='coolwarm', extent=(x1.min(), x1.max(), x2.min(), x2.max()))
    # ax.set_xlim(x1[0], x1[-1])
    # ax.set_ylim(x2[0], x2[-1])
    ax.set_xlabel('LacI Conc.')
    ax.set_ylabel('TetR Conc.')
    fig2.show()


    # %% define
    def cef_A(X, t=None):
        """
        \nabla(f(X))
        f_1(X)
        """
        x1, x2 = X
        return - 2. * toggle.growth_rate


    def cef_B(X, t=None):
        """
        f_1(X)
        """
        # x1, x2 = X
        return toggle.field_flow(X)[0]


    def cef_C(X, t=None):
        """
        f_2(X)
        """
        # x1, x2 = X
        return toggle.field_flow(X)[1]


    def cef_D_1(X, t=None):
        # x1, x2 = X
        return 5


    def cef_D_2(X, t=None):
        # x1, x2 = X
        return 5


    v_x1, v_x2 = cef_B([x1_v, x2_v]), cef_C([x1_v, x2_v])
    # %%
    A_1 = [None] * x2_len
    b_1 = [None] * x1_len
    A_2 = [None] * x1_len
    b_2 = [None] * x2_len

    p1, p2, q1, q2 = t_step / (4 * x1_step), t_step / (4 * x2_step), t_step / (2 * x1_step ** 2), t_step / (
            2 * x2_step ** 2)

    print('Generate matrix A')
    for index, x1_i in enumerate(x1):
        poss = [[x1_i, x2_i] for x2_i in x2]  # positions along x2 direction
        b_main_1 = np.array([1. - 2 * q2 * cef_D_2(pos) - cef_A(pos) * t_step / 2 for pos in poss])
        b_lower_1 = np.array([p2 * cef_C(pos) + q2 * cef_D_2(pos) for pos in poss])[1:]
        b_upper_1 = np.array([- p2 * cef_C(pos) + q2 * cef_D_2(pos) for pos in poss])[:-1]
        b_1_boundary = np.array([2 * q2 * cef_D_2(pos) for pos in poss])
        b_lower_1[-1] = b_1_boundary[-1]
        b_upper_1[0] = b_1_boundary[0]
        # b_1 matrix have a x2 X x2 size
        b_1[index] = sparse.diags([b_lower_1, b_main_1, b_upper_1], [-1, 0, 1], format='csr')

    for index, x2_i in enumerate(x2):
        poss = [[x1_i, x2_i] for x1_i in x1]
        A_main_1 = np.array([1. + 2 * q1 * cef_D_1(pos) for pos in poss])
        A_lower_1 = np.array([- p1 * cef_B(pos) - q1 * cef_D_1(pos) for pos in poss])[1:]
        A_upper_1 = np.array([p1 * cef_B(pos) - q1 * cef_D_1(pos) for pos in poss])[:-1]
        A_1_boundary = np.array([-2 * q1 * cef_D_1(pos) for pos in poss])
        A_upper_1[0] = A_1_boundary[0]
        A_lower_1[-1] = A_1_boundary[-1]
        # A_1 matrix have x1 X x1 size
        A_1[index] = sparse.diags([A_lower_1, A_main_1, A_upper_1], [-1, 0, 1], format='csr')

    for index, x2_i in enumerate(x2):
        poss = [[x1_i, x2_i] for x1_i in x1]
        b_main_2 = np.array([1 - 2 * q1 * cef_D_1(pos) - cef_A(pos) * t_step / 2 for pos in poss])
        b_lower_2 = np.array([q1 * cef_D_1(pos) + p1 * cef_B(pos) for pos in poss])[1:]
        b_upper_2 = np.array([q1 * cef_D_1(pos) - p1 * cef_B(pos) for pos in poss])[:-1]
        b_2_boundary = np.array([2 * q1 * cef_D_1(pos) for pos in poss])
        b_upper_2[0] = b_2_boundary[0]
        b_lower_2[-1] = b_2_boundary[-1]
        # b2 have x1 X x1 size
        b_2[index] = sparse.diags([b_lower_2, b_main_2, b_upper_2], [-1, 0, 1], format='csr')

    for index, x1_i in enumerate(x1):
        poss = [[x1_i, x2_i] for x2_i in x2]
        A_lower_2 = np.array([- p2 * cef_C(pos) - q2 * cef_D_2(pos) for pos in poss])[1:]
        A_main_2 = np.array([1 + 2 * q2 * cef_D_2(pos) for pos in poss])
        A_upper_2 = np.array([p2 * cef_C(pos) - q2 * cef_D_2(pos) for pos in poss])[:-1]
        A_boundary_2 = np.array([-2 * q2 * cef_D_2(pos) for pos in poss])
        A_lower_2[-1] = A_boundary_2[-1]
        A_upper_2[0] = A_boundary_2[0]
        # A_2 have x2 X x2 size
        A_2[index] = sparse.diags([A_lower_2, A_main_2, A_upper_2], [-1, 0, 1], format='csr')

    p_tilde = np.zeros((x1_len, x2_len))
    u_1 = np.zeros((x2_len, x1_len))
    u_2 = np.zeros((x1_len, x2_len))

    # sweep along x1 axis get b1 matrix
    print('Start evolution.')

    if not parallel:
        for t in tqdm(range(len(p_t) - 1)):
            # p = p_t[t]
            for index, x1_i in enumerate(x1):
                # u_1[:, index] = b_1[index].dot(p_t[t][index, :].reshape(-1, 1)).flatten()
                u_1[:, index] = b_1[index].dot(p_t[t][index, :])

            for index, x2_i in enumerate(x2):
                # p_tilde[:, index] = spsolve(A_1[index], u_1[index, :].reshape(-1, 1))
                p_tilde[:, index] = spsolve(A_1[index], u_1[index, :])

            for index, x2_i in enumerate(x2):
                # u_2[:, index] = b_2[index].dot(p_tilde[:, index].reshape(-1, 1)).flatten()
                u_2[:, index] = b_2[index].dot(p_tilde[:, index])

            for index, x1_i in enumerate(x1):
                # p_t[t + 1, index, :] = spsolve(A_2[index], u_2[index, :].reshape(-1, 1))
                p_t[t + 1, index, :] = spsolve(A_2[index], u_2[index, :])
    else:
        def parl_u1(index):
            u_1[:, index] = b_1[index].dot(p_t[t][index, :])
            return None


        def parl_p_tilde(index):
            p_tilde[:, index] = spsolve(A_1[index], u_1[index, :])
            return None


        def parl_u2(index):
            u_2[:, index] = b_2[index].dot(p_tilde[:, index])
            return None


        def parl_pt(index):
            p_t[t + 1, index, :] = spsolve(A_2[index], u_2[index, :])
            return None


        for t in tqdm(range(len(p_t) - 1)):
            # p = p_t[t]
            # for index, x1_i in enumerate(x1):
            _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(parl_u1)(index) for index, x1_i in enumerate(x1))

            # for index, x2_i in enumerate(x2):
            #     p_tilde[:, index] = spsolve(A_1[index], u_1[index, :].reshape(-1, 1))
            _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(parl_p_tilde)(index) for index, x2_i in enumerate(x2))

            # for index, x2_i in enumerate(x2):
            #     u_2[:, index] = b_2[index].dot(p_tilde[:, index].reshape(-1, 1)).flatten()
            _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(parl_u2)(index) for index, x2_i in enumerate(x2))

            # for index, x1_i in enumerate(x1):
            #     p_t[t + 1, index, :] = spsolve(A_2[index], u_2[index, :].reshape(-1, 1))
            _ = Parallel(n_jobs=-1, require='sharedmem')(delayed(parl_pt)(index) for index, x1_i in enumerate(x1))

    # potential[np.isnan(potential)] = potential[~np.isnan(potential)].max()
    # %% Plot potential dynamics
    for index in tqdm(range(len(time))):
        if index % 500 == 0:
            fig, ax = plt.subplots(1, 1)

            ax.imshow(p_t[index, ...].T, cmap='coolwarm', origin='lower', extent=(x1[0], x1[-1], x2[0], x2[-1]),
                      aspect='equal')

            ax.plot(x1, 5. / (1. + x1 ** 2.), '--r')
            ax.plot(5. / (1. + x2 ** 2.), x2, '--g')
            ax.set_xlim(x1[0], x1[-1])
            ax.set_ylim(x2[0], x2[-1])
            ax.text(0, 8, s=f'Time: %.2f s' % time[index])
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            fig.show()
            # fig.savefig(os.path.join('./data', 'bistable', f'{index}.png'))
    # %% Plot Potential Landscape at Pss
    diff = [np.sum((p_t[i + 1, ...] - p_t[i, ...]) ** 2) / np.product(p_t[0, ...].shape) for i in range(len(p_t) - 1)]

    min_err_idx = np.argmin(diff)

    x = np.arange(-8, 8, 1)
    y = np.arange(-8, 8, 1)
    x_len, y_len = len(x), len(y)
    # p_0 = np.zeros((x1_len, x2_len))
    # p_0 = p_0.flatten()
    # x_v, y_v = np.meshgrid(x, y, indexing='ij')
    #
    # v_x, v_y = cef_B([x_v, y_v]), cef_C([x_v, y_v])

    fig2, ax = plt.subplots(1, 1, figsize=(20, 20))

    potential = - np.log(p_t[-1, ...] + 1.01 * np.abs(p_t[-1, ...].min()))
    ax.imshow(potential.T, cmap='coolwarm', origin='lower', extent=(x1[0], x1[-1], x2[0], x2[-1]),
              aspect='equal', interpolation='spline16')
    # ax.plot(x1, [0] * x1_len, '--r')
    # ax.plot(x1, - (omega ** 2 * x1 - omega ** 2 * gamma * x1 ** 3) / (2 * eta * omega), '--g')
    ax.plot(x1, toggle.null_cline_tetr(x1), '--r')
    ax.plot(toggle.null_cline_laci(x2), x2, '--g')
    # ax.quiver(x_v, y_v, v_x, v_y, scale_units='x')
    ax.set_xlim(x1[0], x1[-1])
    ax.set_ylim(x2[0], x2[-1])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    fig2.show()

    # %% potential 3D
    # fig3, ax2 = plt.subplots(subplot_kw={"projection": '3d'}, figsize=(25, 25))
    # ax2.view_init(elev=45, azim=-60)
    # # surf = ax2.plot_surface(x1_v[0:500, 0:500], x2_v[0:500, 0:500],
    # #                         potential[0:500, 0:500], cmap='coolwarm', linewidth=0)
    # surf = ax2.plot_surface(x1_v, x2_v,
    #                         potential, cmap='coolwarm', linewidth=0)
    # # ax2.set_zlim(2.2, 9.5)
    #
    # ax2.set_zscale('log')
    # # ax2.set_xscale('log')
    # # ax2.set_yscale('log')
    #
    # # ax2.zaxis._set_scale('log')
    # fig3.show()

    import matplotlib.tri as mtri
    import sciplot as splt

    splt.whitegrid()
    fig2, ax2 = plt.subplots(subplot_kw={"projection": '3d'}, figsize=(25, 25))
    ax2.view_init(elev=25, azim=-135)
    # surf = ax2.plot_surface(x1_v[0:500, 0:500], x2_v[0:500, 0:500],
    #                         potential[0:500, 0:500], cmap='coolwarm', linewidth=0)
    # surf = ax2.plot_surface(x1_v, x2_v,
    #                         p_t[-1, ...], cmap='coolwarm', linewidth=0)

    mask = (x1_v < x1.max()) & (x2_v < x2.max())
    sparse_poiint = 5
    tri = mtri.Triangulation(x1_v[mask][::sparse_poiint].flatten(), x2_v[mask][::sparse_poiint].flatten())
    surf = ax2.plot_trisurf(x1_v[mask][::sparse_poiint].flatten(), x2_v[mask][::sparse_poiint].flatten(),
                            potential[mask][::sparse_poiint].flatten(),
                            triangles=tri.triangles,
                            cmap='coolwarm')

    cset = ax2.contour(x1_v, x2_v, potential, offset=potential.min() - 50,
                       zdir='z', cmap='coolwarm')
    # nuclines
    # ax2.plot(x1, toggle.null_cline_tetr(x1), '--r', zs=potential.min() - 50, zdir='z')
    # ax2.plot(toggle.null_cline_laci(x2), x2, '--g', zs=potential.min() - 50, zdir='z')
    ax2.set_zlim(potential.min() - 50)
    ax2.set_xlim(0, x1.max())
    ax2.set_ylim(0, x2.max())
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')

    fig2.show()

    # %% marginal probability

    p_ssx1 = np.sum(p_t[-1], axis=1)
    p_ssx2 = np.sum(p_t[-1], axis=0)

    u_ssx1 = -np.log(p_ssx1 + 1.001 * np.abs(p_ssx1.min()))
    u_ssx2 = -np.log(p_ssx2 + 1.001 * np.abs(p_ssx2.min()))

    fig3, ax3 = plt.subplots(1, 1)

    ax3.plot(x2, u_ssx2)
    ax3.set_xscale('log')
    # ax3.plot(x1, p_ssx1)
    fig3.show()

    # %%

    fig4, ax4 = plt.subplots(1, 1)
    ax4.plot(time[1:], diff)
    ax4.set_yscale('log')
    ax4.set_xlabel('time')
    ax4.set_ylabel('$<\Delta X^{2}> $')
    fig4.show()

