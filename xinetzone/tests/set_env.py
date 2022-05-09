# 加载自定义库
import sys
from pathlib import Path
from importlib import import_module

ROOT = Path(__file__).absolute().resolve()
TVM_ROOT = str(ROOT.parents[2])
MOD_PATH = str(ROOT.parents[1]/'src') #str(TVM_ROOT/'xinetzone/src')
# print(MOD_PATH)

if MOD_PATH not in sys.path:
    sys.path.extend([MOD_PATH])

tvmx = import_module('tvmx')
tvmx.set_tvm(TVM_ROOT)
# tvm, vta = tvmx.import_tvm(TVM_ROOT)