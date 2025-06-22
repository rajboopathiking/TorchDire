from setuptools import setup, find_packages

setup(
    name="torchdire",
    version="0.1.18",
    packages=find_packages(),
    install_requires=[
        "torch",
        "PyYAML",
        "tqdm",
        "numpy",        
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
