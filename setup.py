from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ewe-gate",
    version="0.1.0",
    author="Maheep Purohit",
    author_email="purohitmaheep@gmail.com",
    description="Epistemic Weight Engine — Pre-update gating for noise-robust AI learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maheeppurohit/epistemic-weight-engine",
    project_urls={
        "Paper": "https://doi.org/10.5281/zenodo.18940011",
        "Bug Tracker": "https://github.com/maheeppurohit/epistemic-weight-engine/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "full": [
            "torchvision>=0.15.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pandas>=2.0.0",
        ]
    },
    keywords=[
        "machine learning",
        "deep learning",
        "noisy labels",
        "robust learning",
        "epistemic weight",
        "signal reliability",
        "pytorch",
    ],
)
