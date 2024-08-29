#!/usr/bin/env python
# coding: utf-8

import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

dependencies = [
    "pyyaml>=5.1",
    "pytorch-lightning",
]

setup(
    name="traintrack-stttrkx",
    description="Command line utility to configure and run the machine learning pipelines of the stttrkx project.",
    version="0.1.0",
    install_requires=dependencies,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "traintrack=traintrack.main:main",
        ]
    },
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
)
