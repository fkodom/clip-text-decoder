import os
from distutils.core import setup
from subprocess import getoutput

import setuptools


def get_version_tag() -> str:
    try:
        version = os.environ["CLIP_TEXT_DECODER_VERSION"]
    except KeyError:
        version = getoutput("git describe --tags --abbrev=0")

    return version


setup(
    name="clip-text-decoder",
    version=get_version_tag(),
    author="Frank Odom",
    author_email="frank.odom.iii@gmail.com",
    url="https://github.com/fkodom/clip-text-decoder",
    packages=setuptools.find_packages(exclude=["tests"]),
    description="Generate text captions for images from their CLIP embeddings.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "dill",
        "numpy",
        "pickle5",
        "pytorch-lightning",
        "spacy",
        "torch>=1.8",
        "torchtext",
        "clip @ git+https://github.com/openai/CLIP.git",
    ],
    extras_require={
        "test": [
            "black",
            "flake8",
            "isort",
            "pytest",
            "pytest-cov",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)