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
    ctx.run('rm -rf xin/docs/_build')
    ctx.run('rm -rf xin/docs/_staging/')
    ctx.run('rm -rf xin/docs/index.rst xin/docs/genindex.rst')


@task
def install(ctx):
    ctx.run('sudo apt-get update')
    ctx.run('sudo apt-get install -y python3 python3-dev '
            'python3-setuptools gcc libtinfo-dev zlib1g-dev '
            'build-essential cmake libedit-dev libxml2-dev')
    ctx.run('sudo apt install clang-12 clangd-12 llvm-12 liblldb-12-dev')

@task
def cuda(ctx):
    '''安装 CUDA'''
    cmds = ['sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub',
        """sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'""",
        'sudo apt-get update',
        'sudo apt-get install -y cuda-toolkit-11-0']
    for cmd in cmds:
        ctx.run(cmd)

@task
def docker(ctx):
    '''安装docker 及 nvidia-docker 2
    
    测试：`docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark`
    '''
    cmds = ['curl https://get.docker.com | sh',
    'distribution=$(. /etc/os-release;echo $ID$VERSION_ID)',
    'curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -',
    'curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list'
    'curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container-experimental.list | sudo tee /etc/apt/sources.list.d/libnvidia-container-experimental.list'
    'sudo apt-get update',
    'sudo apt-get install -y nvidia-docker2',
    'sudo service docker stop',
    'sudo service docker start']
    for cmd in cmds:
        ctx.run(cmd)


@task
def edit_config(ctx):
    '''仅仅用于 deploy.yml'''
    BUILD = Path('build')
    if not BUILD.exists():
        BUILD.mkdir()
    ctx.run('cp cmake/config.cmake build')
    ctx.run("echo 'set(USE_VTA_FSIM ON)' >> build/config.cmake")
    text = Path('build/config.cmake').read_text()
    text = text.replace('USE_RELAY_DEBUG OFF', 'USE_RELAY_DEBUG ON')
    text = text.replace('USE_LLVM OFF', 'USE_LLVM ON')
    text = text.replace('USE_MICRO OFF', 'USE_MICRO ON')
    with open('build/config.cmake', 'w') as fp:
        fp.write(text)
    # ctx.run("echo 'set(USE_RELAY_DEBUG ON)' >> build/config.cmake")
    # ctx.run("echo 'set(USE_LLVM ON)' >> build/config.cmake")


@task
def export(ctx):
    # ROOT = '/home/xinet/study/'
    ctx.run(f'export TVM_HOME=/home/runner/work/tvm/tvm/')
    ctx.run('export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}')
    ctx.run('export PYTHONPATH=$TVM_HOME/vta/python:${PYTHONPATH}')
    ctx.run('export TVM_LOG_DEBUG="ir/transform.cc=1;relay/ir/transform.cc=1"')


ns = Collection(xin, init, xin_init, install, edit_config, export)
