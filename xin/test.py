import sys
from pathlib import Path
# -- Project information -----------------------------------------------------
project = 'TVM'
copyright = '2022, xinetzone'
author = 'xinetzone'

# The full version, including alpha/beta/rc tags
release = ''
tvm_path = Path(__file__).absolute().parents[1]
sys.path.extend([(tvm_path/"python").as_posix(),
                 (tvm_path/"vta/python").as_posix(),
                 (tvm_path/"docs").as_posix(),
                 ])
print(tvm_path)