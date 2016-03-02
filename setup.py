# -*- coding: utf-8 -*-
from distutils.core import setup

setup(
    name='platoon',
    version='0.5.0',
    author='MILA',
    packages=['platoon'],
    scripts=['scripts/platoon-launcher'],
    url='https://github.com/mila-udem/platoon/',
    license='MIT',
    description='Experimental multi-GPU mini-framework for Theano',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'cffi', 'pyzmq', 'posix_ipc', 'six']
)
