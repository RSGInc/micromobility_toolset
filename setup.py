from setuptools import setup, find_packages

setup(
    name="micromobility_toolset",
    version="0.2.1",
    description="Micromobility Travel Modeling Toolkit",
    author="contributing authors",
    author_email="blake.rosenthal@rsginc.com",
    license="BSD-3",
    url="https://github.com/RSGInc/micromobility_toolset",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3 :: Only",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy >= 1.16.1",
        "pandas >= 1.0.1",
        "pyyaml >= 5.3.1",
        "python-igraph == 0.8.3",
    ],
)
