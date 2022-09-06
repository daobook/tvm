import time
import logging
import os
import numpy as np


import torch

from torchvision.models.quantization import resnet18 as qresnet18

import set_env
import tvm
from tvm import relay


def find_topk(array, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = array.size
    else:
        axis_size = array.shape[axis]
    assert 1 <= k <= axis_size

    array = np.asanyarray(array)
    if largest:
        index_array = np.argpartition(array, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(array, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(array, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices





def pytorch_quantization(model, val_loader, calibration_samples=500):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    # calibration
    print("pytorch calibration begin---------------------------->>")
    for i, (input, _) in enumerate(val_loader, 1):
        if i < calibration_samples:
            output = model(input)
        else:
            print("pytorch calibration end----------------------------->>")
            break
    torch.quantization.convert(model, inplace=True)
    return model


def get_model(val_loader,
              pre_quantization=False,
              batch_size=1,
              calibration_samples=500):
    if pre_quantization:
        #导入pytorch官方resnet18量化模型 https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/resnet.py
        print('load pytorh quantization resnet18--------------------->>')
        model = qresnet18(pretrained=True).eval()
        model = pytorch_quantization(model, val_loader, calibration_samples)
        #print(model)
    else:
        print('load pytorh resnet18--------------------->>')
        model = models.resnet18(pretrained=True).eval()

    input_shape = (batch_size, 3, 224, 224)
    shape_list = [("input", input_shape)]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params


def quantize(mod, params, data_aware,
             val_loader,
             calibration_samples=10,
             batch_size=1):
    if data_aware:
        with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="max"):
            dataset = calibrate_dataset(val_loader,
                                        calibration_samples,
                                        batch_size)
            mod = relay.quantize.quantize(mod, params, dataset)
    else:
        with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    return mod


def run_inference(mod, val_loader, target="llvm"):
    dev = tvm.device(target)
    model = relay.create_executor("vm", mod, dev, target).evaluate()

    for i, batch in enumerate(val_loader):
        data, label = batch_fn(batch)
        prediction = model(data)
        if i > 10:  # 仅在本教程中的几个示例上运行推理
            break
    return model


def run_tvm_model(mod, params, target="llvm"):
    dev = tvm.device(target, 0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return runtime


def main(
        data_aware=True,
        pre_quantization=False,
        calibration_samples=500,
        print_freq=100,):
    val_loader = get_val_loader()
    mod, params = get_model(val_loader,
                            pre_quantization=pre_quantization,
                            calibration_samples=calibration_samples)
    if not pre_quantization:
        mod = quantize(mod, params, data_aware,
                       val_loader=val_loader,
                       calibration_samples=calibration_samples)
    runtime = run_tvm_model(mod, params)
    #print(runtime.benchmark(dev, number=1, repeat=100))

    test_nums = len(val_loader)
    top1_correct = 0
    top5_correct = 0
    print('llvm inference-------------->>')
    for i, (input, label) in enumerate(val_loader, 1):
        runtime.set_input('input', input)
        runtime.run()
        output = runtime.get_output(0).asnumpy()

        # find topk index
        _, preds = find_topk(output, 5)
        print(preds, label)
        if label.item() == preds[0][0]:
            top1_correct += 1

        if label.item() in preds[0]:
            top5_correct += 1

        if i % print_freq == 0:
            print('Test: [{}/{}] \t'
                  'Acc@1 {:.4f} \t'
                  'Acc@5 {:.4f}'.format(
                      i, test_nums, top1_correct / i, top5_correct / i))

    top1 = top1_correct / test_nums
    top5 = top5_correct / test_nums
    print(' * Acc@1 {:.4f} Acc@5 {:.4f}'
          .format(top1, top5))

    time_start = time.time()
    repeat = 100
    for i, (input, label) in enumerate(val_loader, 1):
        for r in range(repeat):
            runtime.set_input('input', input)
            runtime.run()
            output = runtime.get_output(0).asnumpy()
        time_end = time.time()
        print("平均推理时间：", (time_end - time_start)/repeat)
        exit()



# pre_quantization = False
# calibration_samples = 500
# target = "cuda"
# data_aware = True


# mod, params = get_model(val_data,
#                         pre_quantization=pre_quantization,
#                         calibration_samples=calibration_samples)
# quantized_mod = quantize(mod, params, data_aware,
#                val_loader=val_data, 
#                calibration_samples=calibration_samples)

# model = run_inference(quantized_mod, val_data, target=target)