#!/usr/bin/env bash


# PACKAGING & DISTRIBUTING TOOLS
# ______________________________
conda update conda
conda update conda-env
conda update pip
conda update setuptools


# PYPI PACKAGES without complex dependencies
# __________________________________________
conda install Accelerate
conda update accelerate
pip install --upgrade AIMA
pip install --upgrad ANNOY
pip install --upgrade AppNope
pip install --upgrade ArgParse
pip install --upgrade AutoGrad
<<<<<<< HEAD
pip install --upgrade AutoSlugnstp
pip install --upgrade AWSCLI
pip install --upgrade AWSEBCLI
conda install Basemap
conda update basemap
=======
pip install --upgrade AutoSlug
pip install --upgrade AWSEBCLI
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade BigML
pip install --upgrade BigMLer
conda install Bokeh
conda update bokeh
conda install Boto
conda update boto
<<<<<<< HEAD
pip install --upgrade Cartopy
pip install --upgrade Cement
pip install --upgrade Click
pip install --upgrade CLIGJ
pip install --upgrade Climate
conda install CUDAToolkit
conda update cudatoolkit
pip install --upgrade CURL
pip install --upgrade Cycler
pip install --upgrade cypari
pip install --upgrade DataSet-Examples
pip install --upgrade Deap
conda install Decorator
conda update decorator
pip install --upgrade Descartes
=======
pip install --upgrade Cement
pip install --upgrade Click
pip install --upgrade Climate
conda install CUDAToolkit
conda update cudatoolkit
pip install --upgrade cypari
pip install --upgrade Deap
conda install Decorator
conda update decorator
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade DistCan
conda install Django
conda update django
pip install --upgrade Django-Cities-Light
pip install --upgrade Django-AutoComplete-Light
pip install --upgrade Django-Bootstrap-Markdown
pip install --upgrade Django-Classy-Tags
pip install --upgrade Django-DateTime-Widget
pip install --upgrade Django-Easy-Maps
<<<<<<< HEAD
=======

>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade Django-Easy-Timezones
pip install --upgrade Django-Extra-Views
pip install --upgrade Django-Generic-M2M
pip install --upgrade Django-Haystack
pip install --upgrade Django-GM2M
pip install --upgrade Django-Guardian
pip install --upgrade Django-ImageKit
pip install --upgrade Django-Keyboard-Shortcuts
pip install --upgrade Django-Markdown-Deux
pip install --upgrade Django-User-Accounts
pip install --upgrade Django-Userena
pip install --upgrade DjangoRestFramework
pip install --upgrade Docker-Py
pip install --upgrade DockerPty
pip install --upgrade DoCopt
pip install --upgrade Easy-Thumbnails
pip install --upgrade ElasticSearch
pip install --upgrade ELM
pip install --upgrade FileChunkIO
conda install Fiona
pip install --ugrade FreeType
pip install --upgrade FrozenDict
pip install --upgrade FrozenOrderedDict
pip install --upgrade FXRays
pip install --upgrade GeoPy
<<<<<<< HEAD
conda install Geos
conda update geos
=======
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade GGPlot
pip install --upgrade GitDB
pip install --upgrade GitPython
pip install --upgrade GNumPy
<<<<<<< HEAD
pip install --upgrade GoogleMaps
pip install --upgrade Graph-Tool
pip install --upgrade H2O
conda install H5Py
conda update h5py
pip install --upgrade HDF5
=======
pip install --upgrade Graph-Tool
conda install H5Py
conda update h5py
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade HTML2Text
pip install --upgrade Ibis
conda install IOPro
conda update iopro
pip install --upgrade IPyKernel
conda install IPython
conda update ipython
pip install --upgrade IPython-GenUtils
conda install Jinja2
conda update jinja2
pip install --upgrade JmesPath
pip install --upgrade JobLib
pip install --upgrade Jupyter
pip install --upgrade Jupyter-Client
pip install --upgrade Jupyter-Core
<<<<<<< HEAD
pip install --upgrade Kartograph
pip install --upgrade KRB5
pip install --upgrade LCMS
=======
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade Lea
pip install --upgrade LibGDAL
pip install --upgrade LibNetCDF
pip install --upgrade LibPGM
# conda install LibPython
# conda update LibPython
<<<<<<< HEAD
pip install --upgrade LineCache
=======
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade LivereLoad
pip install --upgrade LockFile
pip install --upgrade LSHash
conda install Markdown
conda update markdown
conda install Markdown2
conda update markdown2
conda install MatPlotLib
conda update matplotlib
# pip install --upgrade MEmPaMal   # why does this re-install SciKit-Learn?
pip install --upgrade Milk
#pip install --upgrade MinGW
#conda update MinGW
pip install --upgrade Mistune
pip install --upgrade MkDocs
conda install MKL
conda update mkl
conda install MKL-RT
conda update mkl-rt
conda install MKL-Service
conda update mkl-service
conda install MKLFFT
conda update mklfft

# conda install MySQL-Python   # configure paths first: http://stackoverflow.com/questions/21440230/install-mysql-python-windows
# for virtualenv: http://stackoverflow.com/questions/12498317/install-mysql-python-in-virtualenv-on-windows-7
# grab pre-compiled wheel: http//www.lfd.uci.edu/~gohlke/pythonlibs

<<<<<<< HEAD
pip install --upgrade MRJob
=======
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade Naked
pip install --upgrade NatSort
pip install --upgrade NearPy
conda install NetworkX
conda update networkx
conda install Nose
conda update nose
conda install Numba
conda update numba
conda install NumbaPro
conda update numbapro
conda install NumbaPro_CUDALib
conda update numbapro_cudalib
conda install NumExpr
conda update numexpr
conda install NumPy
conda update numpy
<<<<<<< HEAD
pip install --upgrade OpenSSL
=======
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade Optunity
pip install --upgrade Orange
pip install --upgrade PANNS
pip install --upgrade Path.Py
pip install --upgrade PathSpec
pip install --upgrade PExpect
pip install --upgrade PDRandom
pip install --upgrade PickleShare
<<<<<<< HEAD
conda install PIL
conda update pil
=======
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade PilKit
conda install Pillow
conda update pillow
pip install --upgrade PkgInfo
pip install --upgrade Plac
pip install --upgrade PLink
pip install --upgrade Pluggy
pip install --upgrade Poster
pip install --upgrade ProbPy
conda install Proj4
conda update proj4
pip install --upgrade ProtoBuf

<<<<<<< HEAD
conda install psycopg2
#pip install git+https://github.com/nwcell/psycopg2-windows.git@win64-py27#egg=psycopg2   # install PostgreSQL first
# or grab pre-compiled wheel: http//www.lfd.uci.edu/~gohlke/pythonlibs

pip install --upgrade Py
=======
#pip install git+https://github.com/nwcell/psycopg2-windows.git@win64-py27#egg=psycopg2   # install PostgreSQL first
# or grab pre-compiled wheel: http//www.lfd.uci.edu/~gohlke/pythonlibs

>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade Py-Expression-Eval
pip install --upgrade Py4J
pip install --upgrade PyBrain   # *** note that this package is no longer actively developed & maintained ***
pip install --upgrade PyDiscreteProbability
pip install --upgrade PyOperators
pip install --upgrade PyPNG
<<<<<<< HEAD
pip install --upgrade PyQT
conda install PySAL
conda update pysal
pip install --upgrade PySolr
pip install --upgrade PyTest

pip install git+https://github.com/mitya57/python-markdown-math

pip install --upgrade PyTools
pip install --upgrade PyTZ
pip install --upgrade PyYAML
pip install --upgrade QT
=======
pip install --upgrade PySolr
pip install --upgrade PyTest
pip install --upgrade Python-Markdown-Math
pip install --upgrade PyTools
pip install --upgrade PyTZ
pip install --upgrade PyYAML
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
conda install Redis
conda update redis
conda install Requests
conda update requests
pip install --upgrade Rodeo
pip install --upgrade SciKit-Image
conda install SciKit-Learn
conda update scikit-learn
conda install SciPy
conda update scipy
<<<<<<< HEAD
conda install Shapely
conda update shapely
pip install --upgrade SimpleGeneric
pip install --upgrade SimpleJSON
pip install --upgrade SIP
=======
pip install --upgrade SimpleGeneric
pip install --upgrade SimpleJSON
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
conda install Six
conda update six
pip install --upgrade SKData
pip install --upgrade SMMap
pip install --upgrade SNAPPy
<<<<<<< HEAD
pip install --upgrade Sparkit-Learn
=======
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade Spherogram
conda install Sphinx
conda update sphinx
pip install --upgrade South
conda install SymPy
conda update sympy
<<<<<<< HEAD
pip install --upgrade Tabulate
pip install --upgrade Test_Helper
pip install --upgrade Testify
=======
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade TextTable
pip install --upgrade Toolz
conda install Tornado
conda update tornado
pip install --upgrade Tox
<<<<<<< HEAD
pip install --upgrade TraceBack
=======
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade Traitlets
pip install --upgrade Twine
conda update tk
pip install --upgrade Unicode
conda install UniDecode
conda update UniDecode
<<<<<<< HEAD
pip install --upgrade UnitTest
=======
>>>>>>> aa2a581e66e92def76389c5cba2b772819dc3f28
pip install --upgrade URLLib3
conda install VirtualEnv
conda update virtualenv
pip install --upgrade VirtualEnvWrapper
pip install --upgrade WebSocket-Client
pip install --upgrade Wheel

# pip install MLPy
# pip install PyMC   # needs Fortran compiler
# pip install PyMVPA

pip install --upgrade Distance


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
# pip install --upgrade ANNarchy

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
pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
pip install --upgrade Theanets



# SPARK & SPARK-RELATED


# Spark-dependent

pip install --upgrade Sparkit-Learn


## extra packages: ipywidgets-4.0.2 jupyter-console-4.0.2 nbconvert-4.0.0 nbformat-4.0.0 notebook-4.0.4 qtconsole-4.0.1