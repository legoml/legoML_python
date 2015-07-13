#!/usr/bin/env bash

# Install Anaconda add-ons
conda update conda
conda install Accelerate
conda install IOPro
conda install MKL
conda install Numba
conda install LibPython MinGW
conda update numba
conda update numpy
conda update scikit-learn
conda update scipy
conda update sympy
conda update tk

# install Packaging & Distributing tools
pip install --upgrade pip
pip install --upgrade setuptools

pip install --upgrade ArgParse
pip install --upgrade AIMA
pip install --upgrade Decorator
pip install --upgrade DistCan
pip install --upgrade FrozenDict
pip install --upgrade FrozenOrderedDict
pip install --upgrade GGPlot
pip install --upgrade GraphViz
pip install --upgrade H5Py
pip install --upgrade Keras   # depends on Theano
pip install --upgrade Lea
pip install --upgrade LibPGM
pip install --upgrade LSHash
pip install --upgrade Milk
pip install --upgrade Naked
pip install --upgrade NearPy
pip install --upgrade NetworkX
pip install --upgrade NeuroLab
pip install --upgrade Nose
pip install --upgrade NumExpr
pip install --upgrade Orange
pip install --upgrade PDRandom
pip install --upgrade PkgInfo
pip install --upgrade ProbPy
pip install --upgrade Py-Expression-Eval
pip install --upgrade PyBrain   # *** note that this package is no longer actively developed & maintained ***
pip install --upgrade PyDiscreteProbability
pip install --upgrade PyDot
pip install --upgrade PyDot2
pip install --upgrade PyOpenCL
pip install --upgrade PyOperators

pip uninstall PyParsing   # to install an older version compatible with PyDot (see Stack Overflow thread:
# http://stackoverflow.com/questions/15951748/pydot-and-graphviz-error-couldnt-import-dot-parser-loading-of-dot-files-will)
pip install -Iv https://pypi.python.org/packages/source/p/pyparsing/pyparsing-1.5.7.tar.gz#md5=9be0fcdcc595199c646ab317c1d9a709

pip install --upgrade PyTest
pip install --upgrade PyTools
pip install --upgrade PyYAML
pip install --upgrade Redis
pip install --upgrade Requests
pip install --upgrade SciKit-NeuralNetwork   # depends on Theano
pip install --upgrade Sparkit-Learn
# pip install --upgrade Theano
# Bleading-edge: want to use this to get ongoing bug fixes
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
pip install --upgrade Twine
pip install --upgrade Wheel


#pip install PyLearn2
#pip install PyCUDA   # may not work correctly on Windows
# pip install PyMVPA
# pip install MLPy
# pip install PyMC   # needs Fortran compiler
