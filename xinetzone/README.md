# 项目配置

1. 构建 TVM 项目，生成 TVM 编译器（`libtvm.so`）和 TVM 运行时库（`libtvm_runtime.so`）：

    ```shell
    invoke make
    ```

    如果想要启用 CUDA，则需要运行：

    ```shell
    invoke make --use-cuda
    ```

2. （可选）将 TVM 和 VTA 的 Python 库添加到环境中。

    ```shell
    . env.sh
    ```

3. （可选）直接载入 TVM 和 VTA 的 Python 库到 Python 脚本：

    ```python
    from pathlib import Path
    import tvmx

    # 设定 TVM 项目的根目录
    # TVM_ROOT = Path('/media/pc/data/4tb/lxw/study/tvm')
    TVM_ROOT = Path('.').absolute().parents[1]
    tvm, vta = tvmx.import_tvm(TVM_ROOT)
    # 查看 TVM 和 VTA 路径
    print(f'{tvm}\n{vta}')
    ```