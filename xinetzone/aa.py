import tvm
from tvm import te  # te 代表张量表达式（tensor expression）


def vector_add(n):  # 保存到 d2ltvm
    """TVM expression for vector add"""
    A = te.placeholder((n,), name='a')
    B = te.placeholder((n,), name='b')
    C = te.compute(A.shape,
                   lambda i: A[i] + B[i], name='c')
    return A, B, C

n = 100
A, B, C = vector_add(n)
s = te.create_schedule(C.op)
mod = tvm.build(s, [A, B, C])