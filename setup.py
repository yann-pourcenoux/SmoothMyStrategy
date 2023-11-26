from setuptools import find_packages, setup

STYLE_PKGS = [
    "black~=23.7.0",
    "docformatter~=1.7.5",
    "isort~=5.12.0",
    "pre-commit~=3.3.3",
    "prettier~=0.0.7",
    "ruff~=0.0.280",
]
UTILS_PKGS = [
    "GitPython~=3.1.35",
    "jupyter~=1.0.0",
    "pydantic~=2.5.2",
]
DS_PKGS = [
    "pandas~=2.1.0",
    "yfinance~=0.2.28",
    "statsmodels~=0.14.0",
    "trendln~=0.1.10",
]
ML_PKGS = [
    "xgboost~=2.0.0",
    "scikit-learn~=1.3.0",
]
VIZ_PKGS = [
    "seaborn~=0.12.2",
]

TEST_PKGS = ["pytest~=7.4.0"]


setup(
    name="Finance",
    version="0.0.0",
    description="Private package to have fun around finance.",
    author="Yann POURCENOUX",
    author_email="yann.pourcenoux@gmail.com",
    url="https://gitlab.com/yannpourcenoux/Finance",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=STYLE_PKGS + TEST_PKGS + UTILS_PKGS + DS_PKGS + ML_PKGS + VIZ_PKGS,
)
