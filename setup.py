from setuptools import setup, find_packages

setup(
    name="torchdire",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch>=2.0.0", "PyYAML>=6.0", "tqdm", "numpy"],
    description="A PyTorch-based library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rajboopathiking/TorchDire",
)