from setuptools import setup, find_packages

setup(
    name="ze",
    packages=find_packages(),
    install_requires=[
        'kornia',
        'lightning',
        'lmdb',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'torch',
        'torchvision',
        'wandb'
    ]
)