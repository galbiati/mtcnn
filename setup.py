#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# By: Gianni Galbiati

# Standard libraries
from setuptools import find_packages, setup

# External libraries

# Internal libraries


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
)