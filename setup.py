# setup.py

from setuptools import setup, find_packages

setup(
    name="ThermoElasticSim",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        # 其他依赖包
    ],
)
