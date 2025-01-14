from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setup function
setup(
    name="farq",  # Package name on PyPI
    version="0.1.5.1",  # Package version
    author="Feras Alqrinawi",
    author_email="ferasqr@yahoo.com",
    description="A Python library for raster change detection and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Markdown format for PyPI
    url="https://github.com/ferasqr/farq",  # Repository URL
    packages=find_packages(include=["farq", "farq.*"]),  # Include package and sub-packages
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "rasterio>=1.2.0",
        "scipy>=1.6.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
)
