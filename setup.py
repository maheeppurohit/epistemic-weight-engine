from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ewe-gate",
    version="0.2.0",
    author="Maheep Purohit",
    author_email="purohitmaheep@gmail.com",
    description="Epistemic Weight Engine — Adaptive pre-update gating for noise-robust AI learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maheeppurohit/epistemic-weight-engine",
    project_urls={
        "Paper": "https://doi.org/10.5281/zenodo.18940011",
        "Bug Tracker": "https://github.com/maheeppurohit/epistemic-weight-engine/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    keywords=[
        "noisy labels",
        "robust learning",
        "label noise",
        "epistemic weighting",
        "approval bias",
        "RLHF",
        "neural networks",
        "machine learning",
    ],
)
