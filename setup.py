# -*- coding: utf-8 -*-

"""
This Script for build Cython extension
"""


from distutils.core import setup
from Cython.Build import cythonize

import numpy
setup(ext_modules=cythonize('pde_solver_cpy.pyx'),
      include_dirs=[numpy.get_include()])