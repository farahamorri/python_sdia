from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize

extensions = [
    Extension("knn_cy", ["knn_cy.pyx"], include_dirs=[numpy.get_include()]),
    Extension("knn_cy_opt", ["knn_cy_opt.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name="knn_cy",
    ext_modules=cythonize(extensions, annotate=True, language_level="3"),
)
