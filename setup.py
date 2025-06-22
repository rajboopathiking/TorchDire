from setuptools import setup, find_packages

setup(
    name="torchdire",
    version="0.1.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0,<2.3.0",
        "PyYAML>=6.0",
        "tqdm",
        "numpy<2",           # <-- Pin numpy version here
        "matplotlib",
        "seaborn",
        "pandas",
        "torchvision",
        "onnx",
        "onnxruntime",
        "onnxruntime-tools",
        "timm"
    ],
    description="A PyTorch-based library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rajboopathiking/TorchDire",
)
