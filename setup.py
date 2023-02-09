#!/usr/bin/env python
from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='Greenwood',
    version='0.01',
    description='Forest management application to alarm authorities of illegal logging, poaching and track wild elephant activity by utilizing sound analysis.',
    author='Safa Yousif',
    author_email='safa.yousif@outlook.com',
    url='safaa.dev/greenwood',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'greenwood=greenwood.main:main'
        ],
    },
)