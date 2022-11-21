import os
from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open(os.path.join("lanro_gym", "VERSION"), "r") as f:
    __version__ = f.read().strip()

setup(
    name="lanro_gym",
    description="Gymnasium multi-goal environments for goal-conditioned and language-conditioned deep reinforcement learning build with PyBullet",
    author="Frank Röder",
    author_email="frank.roeder@tuhh.de",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/frankroeder/lanro-gym",
    packages=[package for package in find_packages() if package.startswith("lanro_gym")],
    package_data={ "lanro_gym": ["VERSION"] },
    include_package_data=True,
    version=__version__,
    install_requires=["gymnasium~=0.26", "pybullet", "numpy"],
    extras_require={
        "dev": ["pytest", "yapf", "ipdb", "glfw"]
    },
    python_requires=">=3.7",
    classifiers=[
        "Operating System :: OS Independent", "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7", "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", "Programming Language :: Python :: 3.10"
    ],
)
