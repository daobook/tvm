import os
from . import cudnn

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_CUDNN_LIB_CHECKING'] = '0'

