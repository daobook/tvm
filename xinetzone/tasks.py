'''暂时仅仅提供 Linux 平台
'''
from pathlib import Path
# import shutil
from invoke import task

from d2py.tools.write import site

ROOT = str(Path('..').absolute().resolve())

@task
def init(ctx,
         name='xinetzone',
         email='xinzone@outlook.com'):
    
    # 安装这些最小的共享库（Linux）
    ctx.run('sudo apt-get update')
    package_cmd = ('sudo apt-get install -y git '
                   'gcc g++ libtinfo-dev zlib1g-dev '
                   'build-essential cmake libedit-dev libxml2-dev')
    ctx.run(package_cmd)
    ctx.run('sudo apt install clang clangd llvm liblldb-dev')
    ctx.run(f'pip install -r {ROOT}/xin/requirements.txt')
    # Git
    ctx.run(f'git config user.name {name}')
    ctx.run(f'git config user.eamil {email}')
    ctx.run('git submodule init')
    ctx.run('git submodule update')


@task
def config(ctx, cuda=False):
    BUILD = f'{ROOT}/build'
    if Path(BUILD).exists:
        ctx.run(f'rm -rf {BUILD}')
            #shutil.rmtree(BUILD, ignore_errors=True)
    ctx.run(f'mkdir {BUILD}')
    ctx.run(f'export VTA_HW_PATH={ROOT}/3rdparty/vta-hw')
    origin = f'{ROOT}/cmake/config.cmake'
    target = f'{ROOT}/build/config.cmake'
    with open(origin) as fp:
        content = fp.read()
    with open(target, 'w') as fp:
        content = content.replace('set(USE_LLVM OFF)', 'set(USE_LLVM ON)')
        # content = content.replace('set(USE_VTA_FSIM OFF)', 'set(USE_VTA_FSIM ON)')
        # content = content.replace('set(USE_RELAY_DEBUG OFF)', 'set(USE_RELAY_DEBUG ON)')
        # content = content.replace("set(USE_PIPELINE_EXECUTOR OFF)", "set(USE_PIPELINE_EXECUTOR ON)")
        if cuda:
            content = content.replace('set(USE_CUDA OFF)', 'set(USE_CUDA ON)')
            content = content.replace('set(USE_MICRO OFF)', 'set(USE_MICRO ON)')
            content = content.replace('set(USE_CUBLAS OFF)', 'set(USE_CUBLAS ON)')
            content = content.replace('set(USE_CUDNN OFF)', 'set(USE_CUDNN ON)')
            content = content.replace("set(USE_PAPI OFF)", "set(USE_PAPI ON)")
            content = content.replace("set(USE_PROFILER OFF)", "set(USE_PROFILER ON)")
            content = content.replace("set(USE_UMA OFF)", "set(USE_UMA ON)")
            # content = content.replace('set(USE_NNPACK OFF)', 'set(USE_NNPACK ON)')
        fp.write(content)

@task
def make(ctx):
    BUILD = f'{ROOT}/build'
    with ctx.cd(BUILD):
        ctx.run('cmake ..')
        ctx.run("make -j$(nproc)")

@task
def update(ctx):
    '''仅仅适用于 Linux'''
    with ctx.cd(ROOT):
        # ctx.run('ls')
        ctx.run('rm -rf xin/docs/')
        ctx.run('cp -r docs/ xin/docs/')
        ctx.run('rm -rf xin/docs/arch')
        ctx.run('rm -rf xin/docs/_build')
        ctx.run('rm -rf xin/docs/_staging/')
        ctx.run('rm -rf xin/docs/locales/')
        ctx.run('rm -rf xin/docs/index.rst xin/docs/genindex.rst')
        ctx.run("rm -rf xin/docs/how_to/profile")
        ctx.run("rm -rf xin/docs/reference/api/python")
        ctx.run('cp -r xinetzone/docs/** xin/docs/')
        ctx.run('cp -r xinetzone/docs/tutorial xin/docs/')
        ctx.run('cp -r xinetzone/docs/how_to xin/docs/')
        ctx.run('cp -r xinetzone/docs/topic xin/docs/')
        ctx.run("cp tests/scripts/ci.py xin/tests/scripts/ci.py")

@task
def ln_env(ctx,
           root="/media/pc/data/4tb/lxw/books/tvm",
           target="/media/pc/data/4tb/lxw/anaconda3/envs/tvm39",
           python_version="3.9"):
    '''将 TVM 库添加到 Python 环境
    '''
    # so_files = ["libtvm_runtime.so", "libvta_fsim.so", "libtvm.so"]
    # for so_file in so_files:
    #     ctx.run(f'ln -s {root}/{so_file} {target}/{so_file}')
    ctx.run(f'ln -s {root}/python/tvm '
            f'{target}/lib/python{python_version}/site-packages/tvm')
    ctx.run(f'ln -s {root}/vta/python/vta '
            f'{target}/lib/python{python_version}/site-packages/vta')
    ctx.run(f'ln -s {root}/3rdparty '
            f'{target}/lib/3rdparty')

namespace = site(source=f'{ROOT}/xin', target=f'{ROOT}/xinetzone/site/html')
namespace.add_task(init)
namespace.add_task(config)
namespace.add_task(make)
namespace.add_task(update)
namespace.add_task(ln_env)
