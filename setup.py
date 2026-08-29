from setuptools import setup, find_packages

setup(
    name="torchdire",
    version="1.0.0",
    description="Query-Graph Flow Diffusion (QGFD) Ecosystem for PyTorch",
    author="Raj Boopathi",
    url="https://github.com/rajboopathiking/TorchDire.git",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "peft>=0.4.0",
        "trl>=0.9.0",
        "accelerate>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "transformers>=4.30.0",
            "datasets>=2.0.0",
            "peft>=0.4.0",
            "trl>=0.9.0",
            "accelerate>=0.20.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
