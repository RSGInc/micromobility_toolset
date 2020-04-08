from setuptools import setup, find_packages

setup(
    name='micromobility_toolset',
    version='0.1',
    description='Micromobility Travel Modeling Toolkit',
    author='contributing authors',
    author_email='blake.rosenthal@rsginc.com',
    license='BSD-3',
    url='https://github.com/WFRCAnalytics/Micromobility-Toolset',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'activitysim >= 0.9.2',
        'numpy >= 1.16.1',
        'pandas >= 1.0.1',
    ]
)
