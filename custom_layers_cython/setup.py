from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [
  Extension('dense_cython', ['dense_cython.pyx'],
    include_dirs = [numpy.get_include()]),
]

setup(
    ext_modules = cythonize(extensions),
)

