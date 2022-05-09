import sys
from pathlib import Path
from importlib import import_module


def set_tvm(tvm_root):
    tvm_root = Path(tvm_root)
    TVM_PATH = str(tvm_root/'python')
    VTA_PATH = str(tvm_root/'vta/python')
    sys.path.extend([TVM_PATH, VTA_PATH])
    # for path in [TVM_PATH, VTA_PATH]:
    #     if path not in sys.path:
    #         sys.path.extend([path])


def import_tvm(tvm_root):
    set_tvm(tvm_root)
    # import 模块
    tvm = import_module('tvm')
    vta = import_module('vta')
    return tvm, vta


if __name__ == '__main__':
    # TVM_ROOT = Path('/media/pc/data/4tb/lxw/study/tvm')
    TVM_ROOT = Path(__file__).absolute().parents[3]
    tvm, vta = import_tvm(TVM_ROOT)
    print(f'{tvm}\n{vta}')
