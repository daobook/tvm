import os

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_CUDNN_LIB_CHECKING'] = '0'
os.environ["PATH"] += ":/usr/local/cuda/bin"
os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib64"
