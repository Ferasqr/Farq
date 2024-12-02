from setuptools import setup, find_packages

setup(
    name="farq",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "rasterio>=1.2.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.18.0",
        "pandas>=1.3.0",
    ],
    author="Feras",
    description="A Python library for raster change detection and analysis",
    url="https://github.com/yourusername/farq",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 