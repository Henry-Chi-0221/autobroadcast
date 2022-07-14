
from setuptools import Extension,setup
#from distutils.core import setup
from Cython.Build import cythonize
import numpy
ext_modules = [
    Extension(
        "cython_simple",
        ["cython_simple.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]
setup(
    ext_modules  = cythonize(ext_modules) ,
    include_dirs = [numpy.get_include()]
)