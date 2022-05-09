from torchvision import models
from torch import jit
import torch

import set_env  # 加载 TVM/VTA 环境
import tvm
from tvm import relay

from utils import find_topk, get_topk_correct
from bn_fuse import fuse_module


def batch_fn(batch):
    return batch[0].data.numpy(), batch[1].data.numpy()


def calibrate_dataset(val_loader,
                      calibration_samples=10,
                      batch_size=1):
    #val_loader = get_val_loader()
    for i, batch in enumerate(val_loader):
        xs, ys = batch_fn(batch)
        if i * batch_size >= calibration_samples:
            break
        yield {"xs": xs,
               'ys': ys}


class TorchModel:
    def __init__(self,
                 model_name='resnet18',
                 pretrained=True, **kwargs):
        self.model_name = model_name
        self.pretrained = pretrained

    @property
    def model(self):
        mod = getattr(models, self.model_name)
        return mod(pretrained=self.pretrained)

    def scripted_model(self,
                       batch_size,
                       shape=(3, 224, 224),
                       fuse_model=False):
        input_shape = (batch_size, *shape)
        input_data = torch.randn(input_shape)
        model = self.model
        if fuse_model:
            fuse_module(model)
            # print(model)
        return jit.trace(model, input_data), input_shape

    def _scripted_model(self,
                        batch_size,
                        shape=(3, 224, 224),
                        fuse_model=False):
        input_shape = (batch_size, *shape)
        model = self.model
        if fuse_model:
            model.fuse_model()
        return jit.script(model), input_shape


class RelayFunction:
    @staticmethod
    def get_relay_model(model_name='resnet18',
                        pretrained=True,
                        batch_size=1,
                        shape=(3, 224, 224),
                        fuse_model=False,
                        **kwargs):
        torch_model = TorchModel(model_name=model_name,
                                 pretrained=pretrained, **kwargs)

        scripted_model, input_shape = torch_model.scripted_model(batch_size,
                                                                 shape,
                                                                 fuse_model=fuse_model)
        scripted_model = scripted_model.eval()
        shape_list = [("input", input_shape)]
        # 返回 mod, params
        return relay.frontend.from_pytorch(scripted_model, shape_list)

    @staticmethod
    def quantize(mod, params,
                 data_aware,
                 val_loader,
                 calibration_samples=10,
                 batch_size=1):
        if data_aware:
            with relay.quantize.qconfig(calibrate_mode="kl_divergence",
                                        weight_scale="max"):
                dataset = calibrate_dataset(val_loader,
                                            calibration_samples,
                                            batch_size)
                mod = relay.quantize.quantize(mod, params, dataset)
        else:
            with relay.quantize.qconfig(calibrate_mode="global_scale",
                                        global_scale=8.0):
                mod = relay.quantize.quantize(mod, params)
        return mod

    @staticmethod
    def run_tvm_model(mod, params, target="llvm"):
        dev = tvm.device(target, 0)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        return runtime

    @staticmethod
    def inference(runtime, xs):
        runtime.set_input('input', xs)
        runtime.run()
        return runtime.get_output(0)  # .asnumpy()

    # @staticmethod
    # def timeit(runtime, val_loader):
    #     '''计算平均推理时间'''
    #     time_start = time.time()
    #     repeat = 100
    #     for _, (xs, _) in enumerate(val_loader, 1):
    #         for _ in range(repeat):
    #             RelayFunction.inference(runtime, xs)
    #         time_end = time.time()
    #         return (time_end - time_start)/repeat

    @staticmethod
    def topk(runtime, val_loader, print_freq=100):
        # test_nums = len(val_loader)

        num = 0
        top1_correct = 0
        top5_correct = 0
        # correct = np.array([0]*5)
        for k, (xs, ys) in enumerate(val_loader):
            output = RelayFunction.inference(runtime, xs)
            if isinstance(output, tvm.runtime.ndarray.NDArray):
                output = output.asnumpy() # 或者 output.numpy()
            _, topk_indices = find_topk(output, k=5)
            ys = ys.numpy()
            top1_correct += get_topk_correct(
                ys, topk_indices, k=1)
            top5_correct += get_topk_correct(
                ys, topk_indices, k=5)
            # correct += (topk_indices.T == ys.numpy()).T.sum(axis=0)
            num += len(ys)
            # topk = correct/num
            if k % print_freq == 0:
                print(f'Test: [{k}/{num}] \t'
                      f'Acc@1 {top1_correct/num:.4f} \t'
                      f'Acc@5 {top5_correct/num:.4f}')
        return top1_correct/num, top5_correct/num

    @staticmethod
    def relay(val_loader,
              batch_size=4,
              shape=(3, 224, 224),
              model_name='resnet18',
              pretrained=True,
              tvm_quantize=False,
              data_aware=False,
              calibration_samples=500,
              fuse_model=False):
        # 获取 Relay 模块和参数
        mod, params = RelayFunction.get_relay_model(model_name=model_name,
                                                    pretrained=pretrained,
                                                    batch_size=batch_size,
                                                    shape=shape,
                                                    fuse_model=fuse_model)

        if tvm_quantize:
            mod = RelayFunction.quantize(mod, params,
                                         data_aware,
                                         val_loader,
                                         calibration_samples=calibration_samples,
                                         batch_size=batch_size)
        return mod, params


def run_inference(mod, val_loader, target="llvm"):
    dev = tvm.device(target)
    model = relay.create_executor("vm", mod, dev, target).evaluate()

    for i, batch in enumerate(val_loader):
        data, label = batch_fn(batch)
        prediction = model(data)
        if i > 10:  # 仅在本教程中的几个示例上运行推理
            break
    return model
