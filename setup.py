from setuptools import setup  # type: ignore

STYLE_PKGS = [
    "black~=23.7.0",
    "docformatter~=1.7.5",
    "isort~=5.12.0",
    "mypy~=1.4.1",
    "pre-commit~=3.3.3",
    "prettier~=0.0.7",
    "ruff~=0.0.280",
]


setup(
    name="Finance",
    version="0.0.0",
    description="Private package to have fun around finance.",
    author="Yann POURCENOUX",
    author_email="yann.pourcenoux@gmail.com",
    url="https://gitlab.com/yannpourcenoux/Finance",
    install_requires=STYLE_PKGS,
)
