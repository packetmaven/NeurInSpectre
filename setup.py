#!/usr/bin/env python3
"""
NeurInSpectre: AI Security Intelligence Platform
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="neurinspectre",
    version="1.0.0",
    author="packetmaven",
    description="AI Security Interpretability Platform with cutting-edge threat detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neurinspectre",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.1",
        "torchvision>=0.18.1",
        "torchaudio>=2.3.1",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0,<1.4.0",
        "pandas>=2.0.0",
        "matplotlib>=3.6.0,<3.8.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "dash>=2.14.1",
        "click>=8.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.0",
        "transformers>=4.53.1",
        "accelerate>=0.29.0",
        "cryptography>=41.0.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "full": [
            "torchattacks>=3.4.0",
            "adversarial-robustness-toolbox>=1.15.0",
            "foolbox>=3.3.0",
            "scapy>=2.5.0",
            "yara-python>=4.3.0",
            "pefile>=2023.2.7",
            "entropy>=0.1.5",
            "statsmodels>=0.14.0",
        ],
        "research": [
            "angr>=9.2.0",
            "capstone>=5.0.0",
            "keystone-engine>=0.9.2",
            "unicorn>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neurinspectre=neurinspectre.cli.__main__:main",
            "neurinspectre-obfgrad=neurinspectre.security.visualization.obfuscated_gradient_visualizer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "neurinspectre": [
            "attack_data/*.npy",
            "attack_data/*.json",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/neurinspectre/neurinspectre/issues",
        "Source": "https://github.com/neurinspectre/neurinspectre",
        "Documentation": "https://github.com/neurinspectre/neurinspectre/wiki",
    },
    keywords=[
        "ai-security",
        "machine-learning",
        "cybersecurity",
        "threat-detection",
        "adversarial-ml",
        "red-team",
        "blue-team",
        "security-analysis",
        "gradient-analysis",
        "neural-networks",
    ],
) 
