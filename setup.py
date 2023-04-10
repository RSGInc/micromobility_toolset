from setuptools import setup, find_packages

setup(
    name="micromobility_toolset",
    version="0.3.2",
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
    python_requires=">=3.8",
    install_requires=[
        "pandas >= 1.5",
        "pyarrow",
        "numpy >= 1.24.1",
        "pyyaml",
        "python-igraph",
    ],
)
