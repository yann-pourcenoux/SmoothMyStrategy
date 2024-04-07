"""Setup file for Finance package."""

from setuptools import find_packages, setup

DS_PKGS = [
    "pandas~=2.2.1",
    "yfinance~=0.2.37",
    "statsmodels~=0.14.1",
]
FIN_PKG = [
    "stockstats~=0.6.2",
    "QuantStats",  # The version comes from the URL in dependency_links
]
RL_PKGS = [
    "tensordict~=0.3.1",
    "torchrl~=0.3.1",
    "torch==2.2.2",
]
STYLE_PKGS = [
    "black~=23.7.0",
    "docformatter~=1.7.5",
    "isort~=5.12.0",
    "pre-commit~=3.3.3",
    "prettier~=0.0.7",
    "ruff~=0.0.280",
]
TEST_PKGS = [
    "pytest~=8.1.1",
]
UTILS_PKGS = [
    "GitPython~=3.1.42",
    "hydra-core~=1.3.2",
    "icecream~=2.1.3",
    "jupyter~=1.0.0",
    "loguru~=0.7.2",
    "pydantic~=2.6.4",
]
VIZ_PKGS = [
    "plotly~=5.20.0",
    "seaborn~=0.13.2",
    "wandb~=0.16.4",
]
PKGS = DS_PKGS + FIN_PKG + RL_PKGS + STYLE_PKGS + TEST_PKGS + UTILS_PKGS + VIZ_PKGS

setup(
    name="Finance",
    version="0.0.0",
    description="Private package to have fun around finance.",
    author="Yann POURCENOUX",
    author_email="yann.pourcenoux@gmail.com",
    url="https://gitlab.com/yannpourcenoux/Finance",
    dependency_links=["https://github.com/anchorblock/quantstats"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=PKGS,
)
