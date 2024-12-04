from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="farq",
    version="0.1.2",
    author="Feras",
    author_email="ferasqr@yahoo.com",
    description="A Python library for raster change detection and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ferasqr/farq",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "rasterio>=1.2.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'psutil>=5.0.0',
        ],
        'dev': [
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.900',
            'build>=0.7.0',
            'twine>=3.4.2',
        ],
    }
) 