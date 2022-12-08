# -*- coding: utf-8 -*-

"""
Cython script for boost speed of PDE solve.
 @author: Pan M. CHU
 @Email: pan_chu@outlook.com
"""


import numpy as np
cimport numpy as np
# from tqdm import tqdm
from scipy.sparse.linalg import spsolve
# […]

import cython


# Own modules
np.import_array()

def multi_var_guassian(np.ndarray x, np.ndarray cov, np.ndarray mean):
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
    cdef float p_x

    x -= mean
    p_x = (1. / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov))) *
            np.exp(-(x.T.dot(np.linalg.solve(cov, x))) / 2))[0][0]
    return p_x


def crate_init_p(np.ndarray x1, np.ndarray x2, np.ndarray cov_x12, np.ndarray mean_x12):
    '''
    Two dimensional Gaussian distribution
    Parameters
    ------------
    x1: ndarray
        values for 1st dimension vertices.
    x2: ndarray
        values for 2nd dimension vertices.
    cov_x12: 2x2 array 
        covariance matrix
    mean_x12: (2, 1) array
        exception

    '''
    cdef Py_ssize_t length_x = len(x1)
    cdef np.ndarray p0 = np.zeros(length_x)
    cdef np.ndarray x_prime = np.zeros((2, 1))
    cdef np.ndarray c_p_cov = cov_x12
    cdef np.ndarray c_p_mean = mean_x12
    for x_index in range(length_x):
        x_prime[0, 0], x_prime[1, 0] = x1[x_index], x2[x_index]
        p0[x_index] = multi_var_guassian(x=x_prime, cov=c_p_cov, mean=c_p_mean)
    return p0


def evolve_ADI(np.ndarray time, np.ndarray p0, float save_step,
                list A_1, list A_2, list b_1, list b_2):
    # cdef list t_for_save
    cdef np.float_t t_i
    # t_for_save = [t_i for t_i in time if t_i % save_step == 0.]
    cdef int lengthTime = len(time)
    cdef Py_ssize_t index_time
    cdef np.float_t saveTime=0.

    cdef int saveSize = 0
    for index_time in range(lengthTime):
        if time[index_time] >= saveTime:
            saveSize +=1
            saveTime +=save_step



    # cdef int len_t_for_save = len(t_for_save)
    cdef int x1_len = p0.shape[0]
    cdef int x2_len = p0.shape[1]
    cdef unsigned int save_index = 0

    cdef np.ndarray[ndim=2, dtype=np.float64_t] p_tilde = np.zeros((x1_len, x2_len))
    cdef np.ndarray[ndim=2, dtype=np.float64_t] p_current
    cdef np.ndarray[ndim=2, dtype=np.float64_t] u_1 = np.zeros((x2_len, x1_len))
    cdef np.ndarray[ndim=2, dtype=np.float64_t] u_2 = np.zeros((x1_len, x2_len))
    cdef np.ndarray[ndim=3, dtype=np.float64_t] p_t = np.zeros((saveSize, x1_len, x2_len))

    p_current = p0.copy()
    cdef int len_of_iteration = len(time) - 1
    cdef int update_step = len_of_iteration // 100 + 1
    cdef int progress = 0
    saveTime = 0
    for index_time in range(len_of_iteration):
        if time[index_time] >= saveTime:
            p_t[save_index, :, :] = p_current.copy()
            save_index += 1
            saveTime += save_step

        if index_time % update_step == 0:
            print("%i Precg." % progress)
            progress += 1
        # p = p_t[t]
        for index in range(x1_len):
            # u_1[:, index] = b_1[index].dot(p_t[t][index, :].reshape(-1, 1)).flatten()
            # u_1[:, index] = np.matmul(b_1[index], p_current[index, :].reshape(-1,1))
            u_1[:, index] = b_1[index].dot(p_current[index, :])


        for index in range(x2_len):
            # p_tilde[:, index] = spsolve(A_1[index], u_1[index, :].reshape(-1, 1))
            p_tilde[:, index] = spsolve(A_1[index], u_1[index, :],
                                        use_umfpack=True)

        for index in range(x2_len):
            # u_2[:, index] = b_2[index].dot(p_tilde[:, index].reshape(-1, 1)).flatten()
            u_2[:, index] = b_2[index].dot(p_tilde[:, index])
            # u_2[:, index] = np.matmul(b_2[index], p_tilde[:, index].reshape(-1, 1))

        for index in range(x1_len):
            # p_t[t + 1, index, :] = spsolve(A_2[index], u_2[index, :].reshape(-1, 1))
            p_current[index, :] = spsolve(A_2[index], u_2[index, :],
                                          use_umfpack=True)
        for i in range(x1_len):
            for j in range(x2_len):
                if p_current[i, j] < 0.:
                    p_current[i, j] = 0.

    return  p_t
