# setup.py
from setuptools import setup, find_packages

setup(
    name="stab2ppb",
    version="0.2",
    packages=find_packages(), # 会自动将 utils, stab, ppb 等文件夹识别为包
)