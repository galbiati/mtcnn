#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# By: Gianni Galbiati

# Standard libraries
from setuptools import find_packages, setup

# External libraries

# Internal libraries


with open('requirements.txt', 'r') as f:
    requirements = f.readlines()


setup(
    name='mtcnn',
    version='0.1.0',
    author='Gianni Galbiati',
    author_email='galbiatig@gmail.com',
    packages=find_packages(),
    license='LICENSE.txt',
    description='Pure PyTorch MTCNN',
    long_description=open('README.md').read(),
    include_package_data=True,
    install_requires=requirements
)
