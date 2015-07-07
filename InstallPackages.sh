#!/usr/bin/env bash

# Install Anaconda add-ons
conda update conda
conda install Accelerate
conda install IOPro
conda install MKL
conda install Numba
conda install LibPython MinGW
conda update scikit-learn
conda update tk

# Install additional packages from PyPI (Python Packages Index)
pip install --upgrade pip

pip install AIMA
pip install DistCan
pip install FrozenDict
pip install FrozenOrderedDict
pip install GGPlot
pip install GraphViz
pip install H5Py
pip install Lea
pip install LibPGM
pip install LSHash
pip install Milk
pip install NearPy
pip install NetworkX
pip install NeuroLab
pip install PDRandom
pip install ProbPy
pip install Py-Expression-Eval
pip install PyBrain   # *** note that this package is no longer actively developed & maintained ***
pip install PyDiscreteProbability
#pip install PyLearn2
pip install Redis



pip uninstall PyParsing   # to install an older version compatible with PyDot (see Stack Overflow thread:
# http://stackoverflow.com/questions/15951748/pydot-and-graphviz-error-couldnt-import-dot-parser-loading-of-dot-files-will)
pip install -Iv https://pypi.python.org/packages/source/p/pyparsing/pyparsing-1.5.7.tar.gz#md5=9be0fcdcc595199c646ab317c1d9a709
pip install PyDot
pip install PyDot2

#pip install PyCUDA   # may not work correctly on Windows

pip install Theano
# Bleading-edge: pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

pip install Keras   # depends on Theano
pip install SciKit-NeuralNetwork   # depends on Theano

pip install Orange

# pip install PyMVPA
# pip install MLPy
# pip install PyMC   # needs Fortran compiler
