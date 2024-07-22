from setuptools import setup, find_packages

setup(
    name="contrastive-ops",
    packages=find_packages(),
    install_requires=[
        'lightning',
        'lmdb',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'torch',
        'torchvision'
    ]
)
