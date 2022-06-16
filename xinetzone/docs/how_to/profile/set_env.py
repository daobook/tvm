# 加载自定义库
from platform import python_version
import sys
import os
from pathlib import Path
from importlib import import_module

ROOT = Path(__file__).absolute().resolve().parents[4]
TVM_ROOT = str(ROOT)
MOD_PATH = str(ROOT/'xinetzone/src')
_python_version = python_version()
print(f'Python: {_python_version}')

# TVM_ROOT = "/media/pc/data/4tb/lxw/tvm"
# TVM_ROOT = "/media/pc/data/4tb/lxw/tvm0old"
# if _python_version in ['3.8.10', "3.9.12"]:
#     TVM_ROOT = "/media/pc/data/4tb/lxw/tvm"
# else:
#     TVM_ROOT = "/media/pc/data/4tb/lxw/books/tvm"
# TVM_ROOT = "/media/pc/data/4tb/lxw/tvm"
# TVM_ROOT = "/media/pc/data/4tb/zzy/zzy/npu/tvm"
# TVM_ROOT = "/media/pc/data/4tb/lxw/tvm310"
# TVM_ROOT = "/media/pc/data/4tb/lxw/books/tvm"
os.environ['VTA_HW_PATH'] = os.environ.get('VTA_HW_PATH', f'{TVM_ROOT}/3rdparty/vta-hw')
# print(TVM_ROOT)

if MOD_PATH not in sys.path:
    sys.path.extend([MOD_PATH])

tvmx = import_module('tvmx')
tvmx.set_tvm(TVM_ROOT)
# tvm, vta = tvmx.import_tvm(TVM_ROOT)