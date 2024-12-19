from setuptools import setup

setup(
    name="pyfrbus",
    version="1.1.0",
    install_requires=[
        "pandas",
        "scipy",
        "numpy==1.21.0",
        "black",
        "flake8",
        "mypy",
        "typing_extensions",
        "scikit-umfpack==0.3.3",
        "multiprocess",
        "sympy==1.12",
        "symengine",
        "matplotlib",
        "lxml",
        "networkx",
        "torch",
    ],
    packages=["pyfrbus"],
)
