from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        Extension(
            "regressionC",           
            sources=["regressionC.pyx"],
            include_dirs=[numpy.get_include()]
        )
    ),
)
