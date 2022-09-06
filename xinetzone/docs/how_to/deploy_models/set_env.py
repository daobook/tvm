# 加载自定义库
import sys
from pathlib import Path
from importlib import import_module

TVM_ROOT = Path(__file__).absolute().parents[4]
MOD_PATH = str(TVM_ROOT/'xinetzone/src')
# print(MOD_PATH)

if MOD_PATH not in sys.path:
    sys.path.extend([MOD_PATH])

tvmx = import_module('tvmx')
tvmx.set_tvm(TVM_ROOT)
