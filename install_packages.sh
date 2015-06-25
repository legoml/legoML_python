#!/usr/bin/env bash

# Install Anaconda Add-Ons
conda update conda
conda install Accelerate
conda install IOPro
conda install MKL
conda install mingw libpython

# Install Additional Python Packages
pip install --upgrade pip

pip install FrozenDict
pip install GraphViz
pip install LSHash
pip install NearPy
pip install NetworkX
pip install Py-Expression-Eval
pip install Redis

pip uninstall PyParsing
pip install -Iv https://pypi.python.org/packages/source/p/pyparsing/pyparsing-1.5.7.tar.gz#md5=9be0fcdcc595199c646ab317c1d9a709
pip install PyDot

pip install PyCUDA

pip install Theano
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
