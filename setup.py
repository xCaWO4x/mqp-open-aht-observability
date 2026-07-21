from setuptools import setup, find_packages

setup(
    name="open-aht-observability",
    version="0.1.0",
    description="GPL observability and auxiliary inference experiments in LBF",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "torch>=2.0",
        "gymnasium>=0.26",
        "lbforaging>=2.1",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "tensorboard>=2.13",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "scipy>=1.10",
    ],
)
