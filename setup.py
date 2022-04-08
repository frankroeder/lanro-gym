from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="lanro_gym",
    description="OpenAI Gym multi-goal environments for goal-conditioned and language-conditioned deep reinforcement learning build with PyBullet",
    author="Frank Röder",
    author_email="frank.roeder@tuhh.de",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/frankroeder/lanro-gym",
    packages=find_packages(),
    package_data={},
    include_package_data=True,
    version=open("LANRO_VERSION").read().strip(),
    install_requires=["gym", "pybullet", "numpy"],
    extras_require={"dev": ["pytest", "yapf", "ipdb", "glfw"]},
    python_requires=">=3.6",
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ],
)
