#!/usr/bin/env bash


# ANACONDA ADD-ONS
# ________________
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


# PACKAGING & DISTRIBUTING TOOLS
# ______________________________
pip install --upgrade pip
pip install --upgrade SetupTools


# PYPI PACKAGES without complex dependencies
# __________________________________________
pip install --upgrade AIMA
pip install --upgrade ArgParse
pip install --upgrade AutoGrad
pip install --upgrade BigML
pip install --upgrade BigMLer
pip install --upgrade Bokeh
pip install --upgrade Boto
pip install --upgrade Climate
pip install --upgrade Deap
pip install --upgrade Decorator
pip install --upgrade DistCan
pip install --upgrade ELM
pip install --upgrade FrozenDict
pip install --upgrade FrozenOrderedDict
pip install --upgrade GGPlot
pip install --upgrade GNumPy
pip install --upgrade H5Py
pip install --upgrade JobLib
pip install --upgrade Lea
pip install --upgrade LibPGM
pip install --upgrade LockFile
pip install --upgrade LSHash
pip install --upgrade MEmPaMal
pip install --upgrade Milk
pip install --upgrade Naked
pip install --upgrade NatSort
pip install --upgrade NearPy
pip install --upgrade NetworkX
pip install --upgrade Nose
pip install --upgrade NumExpr
pip install --upgrade Optunity
pip install --upgrade Orange
pip install --upgrade PDRandom
pip install --upgrade Pillow
pip install --upgrade PkgInfo
pip install --upgrade Plac
pip install --upgrade Poster
pip install --upgrade ProbPy
pip install --upgrade ProtoBuf
pip install --upgrade Py-Expression-Eval
pip install --upgrade PyBrain   # *** note that this package is no longer actively developed & maintained ***
pip install --upgrade PyDiscreteProbability
pip install --upgrade PyOperators
pip install --upgrade PyTest
pip install --upgrade PyTools
pip install --upgrade PyYAML
pip install --upgrade Redis
pip install --upgrade Requests
pip install --upgrade SciKit-Image
pip install --upgrade Six
pip install --upgrade SKData
pip install --upgrade Toolz
pip install --upgrade Twine
pip install --upgrade Unicode
pip install --upgrade Wheel

# pip install MLPy
# pip install PyMC   # needs Fortran compiler
# pip install PyMVPA




# THEANO & THEANO-RELATED
# _______________________


# Theano's dependencies
# _____________________
pip install --upgrade GraphViz
# pip install PyCUDA   # may not work correctly on Windows
pip install --upgrade PyDot
pip install --upgrade PyDot2
pip install --upgrade PyOpenCL

pip uninstall PyParsing   # to install an older version compatible with PyDot (see Stack Overflow thread:
# http://stackoverflow.com/questions/15951748/pydot-and-graphviz-error-couldnt-import-dot-parser-loading-of-dot-files-will)
pip install -Iv https://pypi.python.org/packages/source/p/pyparsing/pyparsing-1.5.7.tar.gz#md5=9be0fcdcc595199c646ab317c1d9a709


# Theano itself
# _____________
# pip install --upgrade Theano
# Bleading-edge: want to use this to get ongoing bug fixes
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git


# OPTIMIZATION (some dependent on Theano)
# ____________
pip install --upgrade Downhill
pip install --upgrade Gradient-Optimizers



# NEURAL NETWORKS / DEEP LEARNING-FOCUSED
# (PyPI search keywords: "ann", "deep learning", "neural", "theano")
# _________________________________________________________________
pip install --upgrade ANNarchy

pip install git+git://github.com/mila-udem/blocks.git  -r https://raw.githubusercontent.com/mila-udem/blocks/master/requirements.txt

pip install --upgrade Chainer
# pip install --upgrade DeepCL   # very complex dependencies
pip install --upgrade DeepDish
# pip install --upgrade DeepDist   # work-in-progress, cannot be installed yet
# pip install --upgrade DeepLearning   # work-in-progress, cannot be installed yet
pip install --upgrade Deepy

pip install --upgrade GDBN
pip install --upgrade Graph-Tool-NN
pip install --upgrade Hebel

# pip install --upgrade FANN2   # Exception: Couldn't find swig2.0 binary!
pip install --upgrade Keras
# pip install --upgrade Lasagne   # work-in-progress, cannot be installed yet
pip install --upgrade LMJ.RBM
pip install --upgrade Mang
# pip install --upgrade Monte   # work-in-progress, cannot be installed yet
pip install --upgrade NervanaNEON
pip install --upgrade Neural
pip install --upgrade NeuralPy
pip install --upgrade NeuroLab
# pip install --upgrade NLP   # work-in-progress, cannot be installed yet
pip install --upgrade NN
pip install --upgrade NNToolkit
# pip install --upgrade Nodes   # work-in-progress, cannot be installed yet
pip install --upgrade NOLearn
# pip install --upgrade PUG-ANN   # this is just to install a bunch of relevant supporting packages
# pip install --upgrade Peach   # work-in-progress, cannot be installed yet
pip install --upgrade PyBrain2
pip install --upgrade PyDNN

#pip install PyLearn2 Not available

pip install --upgrade PythonBrain
pip install --upgrade Recurrent-JS-Python
pip install --upgrade SciKit-NeuralNetwork
pip install --upgrade Synapyse
# pip install --upgrade Syntaur   # work-in-progress, cannot be installed yet
pip install --upgrade Theanets



# SPARK & SPARK-RELATED


# Spark-dependent

pip install --upgrade Sparkit-Learn
