import torch
from torchvision.models import quantization as models

from tvm_env import tvm, vta
from vta.top import graph_pack
from tvm import autotvm, relay


def tvm_mod(model_name, input_data):
    tune_path = f'./{model_name}_tune.log'
    model = getattr(models, model_name)(pretrained=True,
                                        quantize=False).eval().cpu()

    scripted_model = torch.jit.trace(model, input_data).eval()
    # 转成 relay 格式，输入层名字可任意指定
    input_shape = tuple(input_data.shape)
    shape_list = [("data", input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params, tune_path


def comple(target, tune_path, relay_prog, model_name, env, params):
    # 编译(交叉编译,输出动态库)
    # 输出指令 debug_flag=2
    with autotvm.tophub.context(target, [tune_path]):
        with vta.build_config(opt_level=3, disable_vectorize=False, disabled_pass={"AlterOpLayout"}):
            #Relay先会寻找 AutoTVM 是否有预先tune好的参数记录
            lib = relay.build(relay_prog, target=env.target,
                              params=params, target_host=env.target_host)
        lib.export_library("./"+model_name+".so", cc="arm-linux-gnueabihf-gcc")
        print("export lib:", lib)


def tvm_pass(mod, params, tune_path, model_name):
    # vta 环境
    env = vta.get_env()
    target = env.target
    with tvm.transform.PassContext(opt_level=3):
        #量化
        with relay.quantize.qconfig(global_scale=8.0,
                                    skip_conv_layers=[],
                                    skip_dense_layer=False):
            mod = relay.quantize.quantize(mod, params=params)
        print("quantize mod:", mod['main'])
        #Perform graph packing and constant folding for VTA target
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH
        )

        #得到模型最优配置
        mod = tvm.IRModule.from_expr(relay_prog)
        tasks = autotvm.task.extract_from_program(
            mod,
            params=params,
            ops=(relay.op.get("nn.conv2d"), relay.op.get("nn.dense")),
            target=target,
            target_host=env.target_host,
        )

        print('tasks:', tasks)
        autotvm.utils.find_config(tasks, tune_path)

        comple(target, tune_path, relay_prog, model_name, env, params)


def run(model_name, input_data):
    mod, params, tune_path = tvm_mod(model_name, input_data)
    print("mod:", mod['main'])
    tvm_pass(mod, params, tune_path, model_name)


if __name__ == '__main__':
    model_names = ['resnet18']  # ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0']
    input_shape = (1, 3, 224, 224)
    input_data = torch.randn(input_shape)

    for model_name in model_names:
        run(model_name, input_data)
        break
