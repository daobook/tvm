from pathlib import Path
from invoke import Collection, task
from _docs import xin


@task
def init(ctx):
    '''
    python -m venv .env
    source .env/bin/activate
    '''
    ctx.run('conda install numpy decorator attrs tornado psutil xgboost')
    ctx.run('conda install scikit-learn-intelex')
    pip_cmd = 'pip install'
    # pip_cmd += ' -i https://pypi.tuna.tsinghua.edu.cn/simple'
    ctx.run(f'{pip_cmd} -r xin/requirements.txt')
    ctx.run(f'{pip_cmd} tensorflow onnx mxnet torch')


@task
def xin_init(ctx):
    '''仅仅适用于 Linux'''
    ctx.run('rm -rf xin/docs/')
    ctx.run('cp -r docs/ xin/docs/')
    ctx.run('cp -r _toc/how_to xin/docs/')
    ctx.run('cp -r _toc/topic xin/docs/')
    ctx.run('cp -r _toc/tutorial xin/docs/')
    ctx.run('rm -rf xin/docs/_staging/')
    ctx.run('rm -rf xin/docs/index.rst xin/docs/genindex.rst')


@task
def install(ctx):
    ctx.run('sudo apt-get update')
    ctx.run('sudo apt-get install -y python3 python3-dev '
            'python3-setuptools gcc libtinfo-dev zlib1g-dev '
            'build-essential cmake libedit-dev libxml2-dev')


@task
def edit_config(ctx):
    '''仅仅用于 deploy.yml'''
    BUILD = Path('build')
    if not BUILD.exists():
        BUILD.mkdir()
    ctx.run('cp cmake/config.cmake build')
    ctx.run("echo 'set(USE_VTA_FSIM ON)' >> build/config.cmake")
    with open('build/config.cmake') as fp:
        text = fp.read()
    text = text.replace('USE_RELAY_DEBUG OFF', 'USE_RELAY_DEBUG ON')
    text = text.replace('USE_LLVM OFF', 'USE_LLVM ON')
    text = text.replace('USE_MICRO OFF', 'USE_MICRO ON')
    with open('build/config.cmake', 'w') as fp:
        fp.write(text)
    # ctx.run("echo 'set(USE_RELAY_DEBUG ON)' >> build/config.cmake")
    # ctx.run("echo 'set(USE_LLVM ON)' >> build/config.cmake")


@task
def export(ctx):
    ctx.run('export TVM_HOME=/home/runner/work/tvm/tvm/')
    ctx.run('export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}')
    ctx.run('export PYTHONPATH=/home/runner/work/tvm/vta/python:${PYTHONPATH}')
    ctx.run('export TVM_LOG_DEBUG="ir/transform.cc=1;relay/ir/transform.cc=1"')


ns = Collection(xin, init, xin_init, install, edit_config, export)
