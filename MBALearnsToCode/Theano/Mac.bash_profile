export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_ROOT="/usr/local/cuda/bin"
export LD_LIBRARY_PATH="/usr/local/cuda/lib"
export THEANO_FLAGS="device=gpu,force_device=False,floatX=float32,reoptimize_unpickled_function=False,blas.ldflags=-L/Applications/anaconda/lib/,cuda.root=/usr/local/cuda/bin/"
