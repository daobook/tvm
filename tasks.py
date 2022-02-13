from invoke import Collection, task
from _docs import xin


@task
def init(ctx):
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
    ctx.run('export TVM_HOME=/home/runner/work/tvm/tvm/')
    ctx.run('export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}')


ns = Collection(xin, init, xin_init)
