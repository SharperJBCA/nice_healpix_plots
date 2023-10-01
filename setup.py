# install the mollweide package to python environment
# python setup.py install
#
import os
import sys
from setuptools import setup, find_packages


setup(
    name='mollweide',
    version='0.1',
    description='A package to plot healpix maps in mollweide projection',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='none',
    author='none',
    author_email='none',
    license='none',
    packages=['mollweide'])

