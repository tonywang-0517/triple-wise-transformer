from setuptools import setup, find_packages

setup(
    name='twa',
    version='0.1.0',
    description='Triple Wise Attention Implementation for Slot-based Video Reconstruction',
    author='Tony Wang',
    packages=find_packages(include=['twa*']),
    install_requires=[
        "torch>=2.0",
        "transformers",
        "tqdm",
        "opencv-python",
        "einops",
        "omegaconf",
        "numpy",
        "matplotlib",
        "pandas",
        "easydict",
        "flash-attn",
        "timm",
    ],
    python_requires=">=3.8",
)
