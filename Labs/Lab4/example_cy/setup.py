import os
from distutils.core import setup

from Cython.Build import cythonize

os.environ["CC"] = "gcc"

setup(
    ext_modules=cythonize(
        ["example_cy/helloworld.pyx", "example_cy/primes.pyx"], annotate=True, language_level="3"
    ),
)
