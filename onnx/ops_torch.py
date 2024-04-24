
import torch
import numpy as np
from typing import Dict

from onnx.onnx_pb2 import TensorProto

data_type_map = {k: v for k, v in TensorProto.DataType.items()}
data_type_id_map = {v: k for k, v in data_type_map.items()}


def gather_nd(x: torch.Tensor, indices: torch.Tensor, batch_dims: int):
    # Note the data rank - will be reused multiple times later
    x = x.numpy()
    indices = indices.numpy()
    data_rank = len(x.shape)

    # The list of data/indice shape of batch_dims.
    batch_dims_shape = []

    # The number of elements in the batch_dims for data/indice array.
    batch_dims_size = 1

    # Check the shape of indice and data are identical for batch dims.
    for i in range(batch_dims):
        batch_dims_shape.append(indices.shape[i])
        batch_dims_size *= indices.shape[i]

    # Compute output of the op as below.
    # Compute shape of output array.
    output_shape = (
        batch_dims_shape + list(indices.shape)[batch_dims:-1]
        if (indices.shape[-1] == data_rank - batch_dims)
        else batch_dims_shape
        + list(indices.shape)[batch_dims:-1]
        + list(x.shape)[batch_dims + indices.shape[-1] :]
    )

    # Placeholder for output data.
    output_data_buffer = []

    # Flatten 'indices' to 2D array.
    reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])

    # Flatten 'data' to array of shape
    # (batch_dim_size, data.shape[batch_dimes:]).
    reshaped_data = x.reshape((batch_dims_size,) + x.shape[batch_dims:])

    # Gather each scalar value from 'data'.
    for batch_dim in range(reshaped_indices.shape[0]):
        for outer_dim in range(reshaped_indices.shape[1]):
            gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
            output_data_buffer.append(reshaped_data[(batch_dim, *gather_index)])
    return torch.from_numpy(np.asarray(output_data_buffer, dtype=x.dtype)).reshape(output_shape)

def SequenceConstruct(*inputs, args):
    return list(inputs)

def Constant(*inputs, args: Dict[str, any]):
    if "value" in args:
        return torch.Tensor([args["value"]])
    elif "value_float" in args:
        return torch.Tensor([args["value_float"]])
    elif "value_int" in args:
        return torch.Tensor([args["value_int"]]).int()
    elif "value_ints" in args:
        return torch.Tensor(args["value_ints"]).int()
    elif "value_floats" in args:
        return torch.Tensor(args["value_floats"])
    else:
        raise NotImplementedError("Constant value not found")

def Cast(*inputs, args):
    dtype = data_type_id_map[args["to"]]
    if dtype == "FLOAT":
        return inputs[0].float()
    elif dtype == "INT64":
        return inputs[0].int()
    else:
        raise NotImplementedError(
            f"Data Type Not implemented for Cast: {dtype}"
        )
    
def Slice(*inputs, args):
    slices = [slice(0, a) for a in inputs[0].shape]
    for s, e, a in zip(inputs[1].tolist(), inputs[2].tolist(), inputs[3].tolist()):
        slices[a] = slice(s, e)
    return inputs[0][tuple(slices)]

def CastLike(*inputs, args):
    return inputs[0].type(inputs[1].dtype)

def Conv(*inputs, args):
    return torch.nn.functional.conv3d(*inputs, dilation=args["dilations"], padding=args["pads"][:3], stride=args["strides"], groups=args["group"])

def Size(*inputs, args):
    return torch.Tensor(list(inputs[0].shape)).int()

def Identity(*inputs, args):
    return None if inputs[0] is None else inputs[0].clone()

def Equal(*inputs, args):
    return torch.eq(*inputs)

def Not(*inputs, args):
    return torch.logical_not(inputs[0])

def MaxPool(*inputs, args):
    (res, indicies) =  torch.nn.functional.max_pool3d_with_indices(*inputs, kernel_size=args["kernel_shape"], return_indices=True, stride=args["strides"], ceil_mode=bool(args["ceil_mode"] if "ceil_mode" in args else 0), dilation=args["dilations"], padding=args["pads"][:3] if "pads" in args else 0)

    return res, indicies.int()

def Range(*inputs, args):
    return torch.arange(*[i.item() for i in inputs]).type(inputs[0].dtype)

def Sub(*inputs, args):
    return torch.sub(*inputs)

def Transpose(*inputs, args):
    return torch.permute(*inputs, args["perm"])

def Mul(*inputs, args):
    return torch.mul(*inputs)

def Add(*inputs, args):
    return torch.add(*inputs)

def Shape(*inputs, args):
    return torch.Tensor(list(inputs[0].shape)).int()

def Unsqueeze(*inputs, args):
    return torch.unsqueeze(inputs[0], dim=inputs[1].int().item())

def Expand(*inputs, args):
    return torch.ones(inputs[1].tolist(), dtype=inputs[0].dtype) * inputs[0]

def Concat(*inputs, args):
    return torch.concatenate(inputs, dim=args["axis"])

def GatherND(*inputs, args):
    return gather_nd(inputs[0], inputs[1], args["batch_dims"])

def ConcatFromSequence(*inputs, args):
    return torch.concatenate(*inputs, dim=args["axis"])

def Squeeze(*inputs, args):
    return torch.squeeze(inputs[0])

def LeakyRelu(*inputs, args):
    return torch.nn.functional.leaky_relu(*inputs, negative_slope=args["alpha"])

def Max(*inputs, args):
    if len(inputs) == 2:
        return torch.max(*inputs)
    else:
        res = inputs[0]
        for i in range(1, len(inputs)):
            res = torch.max(res, inputs[i])
        return res
        