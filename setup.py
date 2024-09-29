import os
import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

cpp_files = glob.glob(os.path.join("qmb", "*.cpp"))

ext_modules = [Pybind11Extension(
    f"qmb.{os.path.splitext(os.path.basename(cpp_file))[0]}",
    [cpp_file],
    cxx_std=17,
) for cpp_file in cpp_files]

setup(ext_modules=ext_modules)
