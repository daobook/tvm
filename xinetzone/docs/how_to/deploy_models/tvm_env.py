# 加载自定义库
import sys
from pathlib import Path
from importlib import import_module

TVM_ROOT = Path(__file__).absolute().resolve().parents[3]
# TVM_ROOT = Path('/media/pc/data/4tb/lxw/tvm')
TVM_PATH = str(TVM_ROOT/'python')
VTA_PATH = str(TVM_ROOT/'vta/python')
print(TVM_PATH, VTA_PATH)

for path in [TVM_PATH, VTA_PATH]:
    if path not in sys.path:
        sys.path.extend([path])

tvm = import_module('tvm')
vta = import_module('vta')
