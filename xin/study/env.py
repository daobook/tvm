# 加载自定义库
import sys
from pathlib import Path
from importlib import import_module

DOC_ROOT = Path(__file__).absolute().parents[2]
MOD_PATH = str(DOC_ROOT/'xinetzone/src')
# print(MOD_PATH)

if MOD_PATH not in sys.path:
    sys.path.extend([MOD_PATH])

tvmx = import_module('tvmx')
# 设定 TVM 项目的根目录
# TVM_ROOT = Path('/media/pc/data/4tb/lxw/study/tvm')
TVM_ROOT = Path(__file__).absolute().resolve().parents[2]
# print(TVM_ROOT)
tvm, vta = tvmx.import_tvm(TVM_ROOT)
# 查看 TVM 和 VTA 路径
print(f'{tvm}\n{vta}')