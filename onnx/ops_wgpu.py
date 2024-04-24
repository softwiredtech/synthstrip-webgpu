import wgpu
import torch
import math
import torch

from onnx.util import tensor_type_cast, item_size
from onnx.onnx_pb2 import TensorProto

import numpy as np
from typing import Any, List

MAX_DIM_SIZE = 65535

adapater = wgpu.gpu.request_adapter(power_preference="high-performance")
device = adapater.request_device(required_limits={"maxStorageBufferBindingSize": 2147483644, "maxBufferSize": 2147483644})

mem_to_wgput = {
    "f": "f32",
    "i": "i32",
    "l": "i32", # TODO: This is long but wgpu does not support
}

mem_to_np = {
    "f": np.float32,
    "i": np.int32,
}

def run_wgpu(src: str, inputs: List[memoryview], out_size: List[int], size: List[int]) -> memoryview:
    module = device.create_shader_module(code=src)
    out_size = [out_size] if not isinstance(out_size, list) else out_size
    bufs = [device.create_buffer_with_data(data=vec, usage=wgpu.BufferUsage.STORAGE) for vec in inputs]
    outs = [device.create_buffer(size=x, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC) for x in out_size]
    binding_layout = [{
            "binding": i,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage
            }
        } for i in range(len(inputs))] 
    for _ in range(len(out_size)):
        binding_layout.append(
            {
                "binding": len(binding_layout),
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage
            }
            }
        )
    bindings = [{ "binding": i, "resource": {"buffer":x, "offset": 0, "size": x.size} } for i, x in enumerate(bufs)]
    for out in outs:
        bindings.append(
            {
                "binding": len(bindings),
                "resource": {"buffer": out, "offset": 0, "size": out.size}
            }
        )
    bind_group_layout = device.create_bind_group_layout(entries=binding_layout)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)
    pipeline = device.create_compute_pipeline(layout=pipeline_layout, compute={"module": module, "entry_point": "main"})
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 0)
    compute_pass.dispatch_workgroups(*size)
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    return device.queue.read_buffer(outs[0]) if len(outs) == 1 else [device.queue.read_buffer(out) for out in outs]

def add_scalar(vec: memoryview, scalar: Any, ):
    shader_source = f"""
    @group(0) @binding(0)
    var<storage,read> in1: array<{mem_to_wgput[vec.format]}>;

    @group(0) @binding(1)
    var<storage,read_write> out: array<{mem_to_wgput[vec.format]}>;

    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
        let i: u32 = index.x;
        out[i] = in1[i] + {scalar};
    }}
    """
    return run_wgpu(shader_source, [vec], vec.nbytes, [vec.shape[0], 1, 1]).cast(vec.format, shape=vec.shape)

def mul_scalar(vec: memoryview, scalar: Any):
    shader_source = f"""
    @group(0) @binding(0)
    var<storage,read> in1: array<{mem_to_wgput[vec.format]}>;

    @group(0) @binding(1)
    var<storage,read_write> out: array<{mem_to_wgput[vec.format]}>;

    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
        let i: u32 = index.x;
        out[i] = in1[i] * {scalar};
    }}
    """
    return run_wgpu(shader_source, [vec], vec.nbytes, [vec.shape[0], 1, 1]).cast(vec.format, shape=vec.shape)

def sub_scalar(vec: memoryview, scalar: Any):
    shader_source = f"""
    @group(0) @binding(0)
    var<storage,read> in1: array<{mem_to_wgput[vec.format]}>;

    @group(0) @binding(1)
    var<storage,read_write> out: array<{mem_to_wgput[vec.format]}>;

    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
        let i: u32 = index.x;
        out[i] = in1[i] * {scalar};
    }}
    """
    return run_wgpu(shader_source, [vec], vec.nbytes, [vec.shape[0], 1, 1]).cast(vec.format, shape=vec.shape)

def sub_vec_broadcast(vec: memoryview, vec2: memoryview, dim: int):
    # TODO: This is hacky, and might not solve all cases.
    size = math.prod(vec.shape[dim+1:])
    last_size = 1
    if size > 65535:
        last_size = vec.shape[-1]
        size //= vec.shape[-1]
    offset = f"gindex.x * {vec.strides[dim] // vec.itemsize} + gindex.y"
    if last_size != 1:
        offset += f" * {vec.strides[-2] // vec.itemsize} + gindex.z"
    shader_source = f"""
    @group(0) @binding(0)
    var<storage,read> in1: array<{mem_to_wgput[vec.format]}>;
    
    @group(0) @binding(1)
    var<storage,read> in2: array<{mem_to_wgput[vec2.format]}>;
    
    @group(0) @binding(2)
    var<storage,read_write> out: array<{mem_to_wgput[vec.format]}>;
    
    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) gindex: vec3<u32>) {{
        let i = {offset};
        out[i] = in1[i] - in2[gindex.x];
    }}
    
    """
    return run_wgpu(shader_source, [vec, vec2], vec.nbytes, [vec.shape[dim], size, last_size]).cast(vec.format, shape=vec.shape)

def leaky_relu(vec: memoryview, alpha: float):
    x = math.prod(vec.shape)
    y = 1
    z = 1
    pos = 1
    while x > 65535:
        x //= vec.shape[-pos]
        y *= vec.shape[-pos]
        pos += 1
    offset = "gindex.x"
    if y > 1:
        offset += f" * {vec.strides[-(pos)] // vec.itemsize} + gindex.y"
    shadersource = f"""
    @group(0) @binding(0)
    var<storage,read> in1: array<{mem_to_wgput[vec.format]}>;
    
    @group(0) @binding(1)
    var<storage,read_write> out: array<{mem_to_wgput[vec.format]}>;
    
    @compute
    @workgroup_size(1, 1, 1)
    fn main(@builtin(global_invocation_id) gindex: vec3<u32>) {{
        let i = {offset};
        out[i] = select(in1[i], in1[i] * {alpha}, in1[i] < 0.0);
    }}
    """
    # print(shadersource, [x, y, z])
    return run_wgpu(shadersource, [vec], vec.nbytes, [x, y, z]).cast(vec.format, shape=vec.shape)

def equal_op(x: memoryview, y: memoryview):
    assert x.shape == y.shape, "Equal operation requires same shape"
    shadersource = f"""
    @group(0) @binding(0)
    var<storage,read> in1: array<{mem_to_wgput[x.format]}>;
    
    @group(0) @binding(1)
    var<storage,read> in2: array<{mem_to_wgput[y.format]}>;
    
    @group(0) @binding(2)
    var<storage,read_write> out: array<i32>;
    
    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) gindex: vec3<u32>) {{
        let i = gindex.x;
        out[i] = i32(in1[i] == in2[i]);
    }}
    """
    return run_wgpu(shadersource, [x, y], x.nbytes, [math.prod(x.shape), 1, 1]).cast(x.format, shape=x.shape)

def get_index_not_one(x: List[int]):
    last = -1
    for i, v in enumerate(x):
        if v != 1:
            last = i
        if v == 1 and last != -1:
            return last
    return last

def expand_max(x: memoryview, y: memoryview):
    xshape = expand_shape(x.shape, y.ndim)
    xdim = get_index_not_one(xshape)
    yshape = expand_shape(y.shape, x.ndim)
    ydim = get_index_not_one(yshape)
    fshape = final_shape(xshape, yshape)
    # print(xdim, ydim)
    shader_source = f"""
    @binding(0) @group(0)
    var<storage, read> x: array<{mem_to_wgput[x.format]}>;
    
    @binding(1) @group(0)
    var<storage, read> y: array<{mem_to_wgput[y.format]}>;
    
    @binding(2) @group(0)
    var<storage, read_write> out: array<{mem_to_wgput[x.format]}>;
    
    @compute
    @workgroup_size(1, 1)
    fn main(@builtin(global_invocation_id) gindex: vec3<u32>) {{
        out[gindex.x * {yshape[ydim]} + gindex.y] = max(x[gindex.x], y[gindex.y]);
    }}
    """
    return run_wgpu(shader_source, [x, y], math.prod(fshape) * x.itemsize, [math.prod(xshape[:xdim+1]), yshape[ydim]]).cast(x.format, shape=fshape)

def cast_op(x: memoryview, target: str):
    target = target if target != tensor_type_cast[TensorProto.INT64] else tensor_type_cast[TensorProto.INT32]
    shader_source = f"""
    @binding(0) @group(0)
    var<storage, read> x: array<{mem_to_wgput[x.format]}>;
    
    @binding(1) @group(0)
    var<storage, read_write> out: array<{mem_to_wgput[target]}>;
    
    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) gindex: vec3<u32>) {{
        out[gindex.x] = {mem_to_wgput[target]}(x[gindex.x]);
    }}
    """
    return run_wgpu(shader_source, [x], math.prod(x.shape) * item_size[target], [math.prod(x.shape), 1, 1]).cast(target, shape=x.shape)

def get_strides(x: List[int]):
    res = [1]
    for i in reversed(x):
        res.append(res[-1] * i)
    res.pop()
    return res[::-1]

def max_pool_3d(x: memoryview, kernel_shape: List[int],  ceil_mode: bool = False, strides: List[int] = None, pads: List[int] = None, dilations: List[int] = None):
    ceil_mode = ceil_mode if ceil_mode is not None else False
    pads = pads if pads is not None else [0] * len(kernel_shape) * 2
    fix = lambda x: int(math.floor(x) if not ceil_mode else math.ceil(x))
    out_shape = list(x.shape[:-3]) + [fix((s + 2 * p - d * (ks - 1) -1) / st + 1) for s, p, d, ks, st in zip(x.shape[-3:], pads, dilations, kernel_shape, strides)]
    # local_size, global_size
    out_strides = get_strides(out_shape)
    in_strides = get_strides(x.shape)
    local_size = [out_shape[-4], 1, 1]
    global_size = [out_shape[-3], out_shape[-2], out_shape[-1]]
    shader_source = f"""
    @binding(0) @group(0)
    var<storage, read> in: array<{mem_to_wgput[x.format]}>;
    
    @binding(1) @group(0)
    var<storage, read_write> out: array<{mem_to_wgput[x.format]}>;
    
    @binding(2) @group(0)
    var<storage, read_write> out_indicies: array<u32>;
    
    @compute
    @workgroup_size({local_size[0]}, {local_size[1]}, {local_size[2]})
    fn main(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{
        var c = lindex.x;
        var x = gindex.x;
        var y = gindex.y;
        var z = gindex.z;

        var ix = x * {strides[0]};
        var iy = y * {strides[1]};
        var iz = z * {strides[2]};
        
        var res = in[c * {in_strides[1]} + ix * {in_strides[2]} + iy * {in_strides[3]} + iz];
        var pos = ix * {in_strides[2]} + iy * {in_strides[3]} + iz;
        var ioffset = 0u;
        for (var i = 0u; i < {kernel_shape[0]}; i++) {{
            for (var j = 0u; j < {kernel_shape[1]}; j++) {{
                for (var k = 0u; k < {kernel_shape[2]}; k++) {{
                    ioffset = c * {in_strides[1]} + (ix + i) * {in_strides[2]} + (iy + j) * {in_strides[3]} + (iz + k);
                    if (in[ioffset] > res) {{
                        pos = (ix + i) * {in_strides[2]} + (iy + j) * {in_strides[3]} + (iz + k);
                        res = in[ioffset];
                    }}
                }}
            }}
        }}
        let outpos = c * {out_strides[1]} + x * {out_strides[2]} + y * {out_strides[3]} + z;
        out_indicies[outpos] = pos;
        out[outpos] = res;
    }}
    """
    vals, indicies = run_wgpu(shader_source, [x], [math.prod(out_shape) * x.itemsize, math.prod(out_shape) * 4], global_size)
    return vals.cast(x.format, shape=out_shape), indicies.cast("i", shape=out_shape)

def conv_3d(x: memoryview, w: memoryview, b: memoryview, strides: List[int], pads: List[int], dilations: List[int]):
    out_shape = [x.shape[0], w.shape[0]] + [int((s + 2 * p - d * (ks - 1) - 1) / st + 1) for (s, p, d, ks, st) in zip(x.shape[-3:], pads, dilations, w.shape[-3:], strides)]
    out_strides = get_strides(out_shape)
    in_strides = get_strides(x.shape)
    w_strides = get_strides(w.shape)
    local_size = [w.shape[0], 1, 1]
    global_size = [out_shape[-3], out_shape[-2], out_shape[-1]]
    src_shader = f"""
    @binding(0) @group(0)
    var<storage, read> in: array<{mem_to_wgput[x.format]}>;
    
    @binding(1) @group(0)
    var<storage, read> w: array<{mem_to_wgput[w.format]}>;
    
    @binding(2) @group(0)
    var<storage, read> b: array<{mem_to_wgput[b.format]}>;
    
    @binding(3) @group(0)
    var<storage, read_write> out: array<{mem_to_wgput[x.format]}>;
    
    @compute
    @workgroup_size({local_size[0]}, {local_size[1]}, {local_size[2]})
    fn main(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {{
        var cout = lindex.x;
        var x = gindex.x;
        var y = gindex.y;
        var z = gindex.z;
        var res = 0.0;
        for(var c = 0u; c < {w.shape[1]}; c++) {{
            for(var i = 0u; i < {w.shape[2]}; i++) {{
                for(var j = 0u; j < {w.shape[3]}; j++) {{
                    for (var k = 0u; k < {w.shape[4]}; k++) {{
                        var ix = x * {strides[0]} + i * {dilations[0]} - {pads[0]};
                        var iy = y * {strides[1]} + j * {dilations[1]} - {pads[1]};
                        var iz = z * {strides[2]} + k * {dilations[2]} - {pads[2]};
                        if (ix < 0 || ix >= {x.shape[-3]} || iy < 0 || iy >= {x.shape[-2]} || iz < 0 || iz >= {x.shape[-1]}) {{
                            continue;
                        }}
                        res += in[c * {in_strides[1]} + ix * {in_strides[2]} + iy * {in_strides[3]} + iz] * w[cout * {w_strides[0]} + c * {w_strides[1]} + i * {w_strides[2]} + j * {w_strides[3]} + k];
                    }}
                }}
            }}
        }}
        out[cout * {out_strides[1]} + x * {out_strides[2]} + y * {out_strides[3]} + z] = res + b[cout];
    }}
    """
    return run_wgpu(src_shader, [x, w, b], [math.prod(out_shape) * x.itemsize], global_size).cast(x.format, shape=out_shape)

def to_mem(x: torch.Tensor):
    return memoryview(x.contiguous().numpy())

def from_mem(x):
    res = torch.from_numpy(np.array(x))
    return res 

def Sub(*_inputs, args):
    assert len(_inputs) == 2, "Sub takes exactly 2 inputs"
    if _inputs[0].numel() == 1 and _inputs[1].numel() == 1:
        return _inputs[0] - _inputs[1]
    inputs = list(map(to_mem, _inputs))
    if any([math.prod(x.shape) == 1 for x in inputs]):
        scalar = inputs[0].tolist() if len(inputs[0]) == 1 else inputs[1].tolist()
        vec = inputs[1] if len(inputs[0]) == 1 else inputs[0]
        return from_mem(sub_scalar(vec, scalar))
    else:
        # check if it will broadcast
        x, y = inputs
        if x.ndim == y.ndim:
            point = 0
            for i in range(x.ndim):
                if x.shape[i] == y.shape[i]:
                    continue
                point = i
                break
            return from_mem(sub_vec_broadcast(x, y, point - 1))
        print("SUB: ", [x.shape for x in inputs], point)
        exit(1)
        
def Mul(*_inputs, args):
    assert len(_inputs) == 2, "Mul takes exactly 2 inputs"
    if _inputs[0].numel() == 1 and _inputs[1].numel() == 1:
        return _inputs[0] * _inputs[1]
    inputs = list(map(to_mem, _inputs))
    if any([math.prod(x.shape) == 1 for x in inputs]):
        scalar = inputs[0][0] if len(inputs[0]) == 1 else inputs[1][0]
        vec = inputs[1] if len(inputs[0]) == 1 else inputs[0]
        return from_mem(mul_scalar(vec, scalar))
    else:
        print("MUL", [x.shape for x in inputs])
        exit(1)

def Add(*_inputs, args):
    inputs = list(map(to_mem, _inputs))
    assert len(inputs) == 2, "Add takes exactly 2 inputs"
    if any([math.prod(x.shape) == 1 for x in inputs]):
        scalar = inputs[0][0] if len(inputs[0]) == 1 else inputs[1][0]
        vec = inputs[1] if len(inputs[0]) == 1 else inputs[0]
        return from_mem(add_scalar(vec, scalar))
    else:
        raise NotImplementedError("Add for vectors not implemented")

def Not(*_inputs, args):
    # NOTE: This is a hacky implementation
    if (_inputs[0].numel() < 1024):
        return torch.logical_not(_inputs[0])
    raise NotImplementedError("Not implemented in wgpu")

def Equal(*_inputs, args):
    inputs = list(map(to_mem, _inputs))
    return from_mem(equal_op(*inputs))

def Max(*_inputs, args):
    inputs = list(map(to_mem, _inputs))
    x = inputs[0]
    for i in range(1, len(inputs)):
        y = inputs[i]
        x = expand_max(x, y)
    return from_mem(x)

def Cast(*_inputs, args):
    inputs = list(map(to_mem, _inputs))
    if inputs[0].format == tensor_type_cast[args["to"]]:
        return from_mem(inputs[0])
    return from_mem(cast_op(inputs[0], tensor_type_cast[args["to"]]))

def MaxPool(*_inputs, args):
    inputs = list(map(to_mem, _inputs))
    (res, indicies) = max_pool_3d(inputs[0], **args)
    return from_mem(res), from_mem(indicies)
    
def expand_shape(shape: List[int], size: int):
    shape = shape if isinstance(shape, list) else list(shape)
    while len(shape) < size:
        shape.insert(0, 1)
    return shape

def final_shape(xshape: List[int], yshape: List[int]):
    xshape = expand_shape(xshape, len(yshape))
    yshape = expand_shape(yshape, len(xshape))
    res = []
    for i in range(len(xshape)):
        if xshape[i] == yshape[i]:
            res.append(xshape[i])
        elif xshape[i] == 1 or yshape[i] == 1:
            res.append(max(xshape[i], yshape[i]))
        else:
            raise ValueError("Cannot broadcast")
    return res

def Conv(*_inputs, args):
    assert len(_inputs) == 3, "only bias is supported"
    assert args["group"] == 1, "only group=1 is supported"
    inputs = list(map(to_mem, _inputs))
    del args["group"]
    return from_mem(conv_3d(*inputs, **args))

def get_min(x: int, max: int):
    count = 1
    while x < max:
        x += x
        count += 1
    return count

def LeakyRelu(*_inputs, args):
    assert len(_inputs) == 1, "LeakyRelu takes exactly 1 input"
    inputs = list(map(to_mem, _inputs))
    return from_mem(leaky_relu(inputs[0], args["alpha"]))