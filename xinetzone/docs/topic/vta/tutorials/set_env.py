from pathlib import Path
from tvm_book.tvm.env import set_tvm
TVM_ROOT = Path(__file__).absolute().parents[5]
# print(TVM_ROOT)
set_tvm(TVM_ROOT)