#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:10:42 2017

@author: tsakai
"""

from setuptools import setup, find_packages
from spmlib import __version__

def main():
    setup(
        name             = 'spmlib',
        version          = __version__,
        license          = 'MIT',
        description      = 'Sparse Modeling Library',
        author           = 'Tomoya Sakai, Dr. and his lab. members',
        keywords         = ['sparse coding', 'compressed sensing', 'matching pursuit algorithm', 'proximity operator', 'optimization', 'random projection', 'matrix completion', 'stable principal component pursuit', 'online RPCA'],
        author_email     ='tsakai@cis.nagasaki-u.ac.jp',
        url              = 'https://github.com/tsakailab/spmlib',
        zip_safe         = False,
        install_requires = ['numpy', 'scipy', 'matplotlib', 'future'],
        packages         = find_packages(),
        classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules'        
    ],
    )


if __name__ == '__main__':
    main()
