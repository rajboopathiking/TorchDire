from setuptools import setup, find_packages

setup(
    name="TorchDire",
    version="0.1.19",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0,<2.3.0",
        "PyYAML>=6.0",
        "tqdm",
        "numpy<2",
        "matplotlib",
        "seaborn",
        "pandas",
        "torchvision>=0.15.0",
        "onnx",
        "onnxruntime",
        "onnxruntime-tools",
        "timm"
    ],
    python_requires=">=3.8",
    description="A PyTorch-based library with YAML automation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rajboopathiking/TorchDire",
    author="Boopathi Raj",
    author_email="rajboopathiking@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords=["pytorch", "deep learning", "automation", "yaml"]
)
