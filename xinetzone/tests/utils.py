import numpy as np
import torch


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


def get_topk_correct(ys, topk_indices, k=5):
    assert len(ys.shape) == 1
    correct = (topk_indices.T == ys)
    return sum(correct[:k].sum(axis=0) >= 1)


def get_top1_top5(model, val_loader, print_freq=1000):
    num = 0
    top1_correct = 0
    top5_correct = 0
    for k, (xs, ys) in enumerate(val_loader):
        output = model(xs)
        if isinstance(output, torch.Tensor):
            output = output.detach().numpy()
        _, topk_indices = find_topk(output, k=5)
        ys = ys.numpy()
        top1_correct += get_topk_correct(ys, topk_indices, k=1)
        top5_correct += get_topk_correct(ys, topk_indices, k=5)
        num += len(ys)
        if k % print_freq == 0:
            print(f'Test: batch/num: [{k}/{num}] \t'
                f'Acc@1 {top1_correct/num:.4f} \t'
                f'Acc@5 {top5_correct/num:.4f}')
    return top1_correct/num, top5_correct/num