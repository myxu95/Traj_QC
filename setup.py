#!/usr/bin/env python3
"""
Setup script for Traj_QC package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Traj_QC - Trajectory Quality Control System"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="traj-qc",
    version="0.1.0",
    author="Traj_QC Team",
    author_email="",
    description="A Python package for GROMACS molecular dynamics trajectory analysis and quality assessment",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Traj_QC",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "traj-qc=traj_qc_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "traj_qc": ["config/*.yaml"],
    },
    keywords="molecular dynamics, gromacs, trajectory analysis, quality assessment, bioinformatics, computational chemistry",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/Traj_QC/issues",
        "Source": "https://github.com/yourusername/Traj_QC",
        "Documentation": "https://traj-qc.readthedocs.io/",
    },
) 