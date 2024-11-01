# setup.py

from setuptools import setup, find_packages

setup(
    name="ThermoElasticSim",
    version="3.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
    ],
)
