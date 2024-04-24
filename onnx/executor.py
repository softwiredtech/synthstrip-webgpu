from typing import Dict, Tuple, Any
import torch
import os
import numpy as np

import onnx.ops_torch as ops_torch
import onnx.ops_wgpu as ops_wgpu

from onnx.util import tensor_type_cast
from onnx.onnx_pb2 import (
    ModelProto,
    TensorProto,
    TypeProto,
    AttributeProto,
    FunctionProto,
    GraphProto
)

missing_ops = set()

DEBUG=int(os.getenv("DEBUG", "0")) > 0


def format_inputs(x: Dict[str, torch.Tensor]):
    return [(k, (v.shape, v.dtype) if isinstance(v, torch.Tensor) else len(v) if isinstance(v, list) else v) for (k,v) in x.items()]

def parse_tensor_proto(tensor: TensorProto):
    if tensor.data_type not in tensor_type_cast:
        raise NotImplementedError(
            f"Data Type Not implemented for tensor: {tensor.name}, data_type: {tensor.data_type}"
        )
    assert len(tensor.raw_data), "Only support raw data for now"
    temp = torch.from_numpy(np.array(memoryview(tensor.raw_data).cast(tensor_type_cast[tensor.data_type], shape=tuple(tensor.dims))))

    if temp.dtype == torch.int64:
        temp = temp.type(torch.int32)
    return temp

def parse_value_info_proto(x: TypeProto):
    if x.WhichOneof("value") is None:
        return None
    assert x.WhichOneof("value") == "tensor_type", f"Only support tensor type for now got {x.WhichOneof('value')}"
    assert x.tensor_type.elem_type == 1, f"Only support float for now got {x.tensor_type.elem_type}"
    return torch.zeros(
        size=[s.dim_value for s in x.tensor_type.shape.dim], dtype=torch.float32
    )


def parse_attribute(attr: AttributeProto, parent_args=None):
    if attr.type == AttributeProto.AttributeType.FLOAT:
        return float(attr.f) if attr.ref_attr_name == "" else parent_args[attr.ref_attr_name]
    elif attr.type == AttributeProto.AttributeType.INT:
        return int(attr.i) if attr.ref_attr_name == "" else parent_args[attr.ref_attr_name]
    elif attr.type == AttributeProto.AttributeType.STRING:
        return str(attr.s) if attr.ref_attr_name == "" else parent_args[attr.ref_attr_name]
    elif attr.type == AttributeProto.AttributeType.GRAPH:
        return attr.g if attr.ref_attr_name == "" else parent_args[attr.ref_attr_name]
    elif attr.type == AttributeProto.AttributeType.TENSOR:
        return parse_tensor_proto(attr.t) if attr.ref_attr_name == "" else parent_args[attr.ref_attr_name]
    elif attr.type == AttributeProto.AttributeType.FLOATS:
        return [float(x) for x in attr.floats] if attr.ref_attr_name == "" else parent_args[attr.ref_attr_name]
    elif attr.type == AttributeProto.AttributeType.INTS:
        return [int(x) for x in attr.ints] if attr.ref_attr_name == "" else parent_args[attr.ref_attr_name]
    else:
        raise NotImplementedError(f"Attribute Type Not implemented: {attr.type}")


def parse_function(func: FunctionProto):
    nodes = [
        {
            "name": n.name,
            "attributes": n.attribute,
            "input": n.input,
            "output": n.output,
            "op_type": n.op_type,
        }
        for n in func.node
    ]
    return {
        "attribute": func.attribute,
        "input": func.input,
        "output": func.output,
        "nodes": nodes,
    }

class OnnxExecutor:
    def __init__(self, file: str):
        self.file = file
        self.model = ModelProto()
        self.tensors = None
        self.funcs = None

    def init(self):
        with open(self.file, "rb") as f:
            self.model.ParseFromString(f.read())
        self.load_tensors_dict()
        self.load_functions()

    def load_functions(self):
        if self.funcs is not None:
            return
        self.funcs = {func.name: parse_function(func) for func in self.model.functions}

    def load_tensors_dict(self):
        if self.tensors is not None:
            return
        self.tensors = {i.name: parse_tensor_proto(i) for i in self.model.graph.initializer}

    def get_inputs(self) -> Dict[str, torch.Tensor]:
        return {x.name: parse_value_info_proto(x.type) for x in self.model.graph.input}

    def run_func(self, func, inputs: Dict[str, torch.Tensor], args, parent_args=None) -> Tuple[torch.Tensor]:
        interDict = {name: val for (name, val) in zip(func["input"], inputs.values())}
        for node in func["nodes"]:
            inputs = {x: interDict[x] for x in node["input"]}
            attributes = {x.name: parse_attribute(x, args) for x in node["attributes"]}
            if DEBUG:
                print(
                    f"\t\tRunning {node['op_type']} inputs={format_inputs(inputs)} kwargs={attributes} outputs={node['output']}"
                )
            if node["op_type"] in self.funcs:
                output = self.run_func(self.funcs[node["op_type"]], inputs, attributes, parent_args=args)
            else:
                output = self.run_op(node["op_type"], inputs if node["op_type"] != "If" else {**inputs, **interDict}, attributes, args)
            assert len(output) == len(
                node["output"]
            ), f"Output length mismatch: {node['op_type']} {len(output)} != {len(node['output'])}"
            for name, val in zip(node["output"], output):
                interDict[name] = val
        return tuple([interDict[x] for x in func["output"]])


    def run_op(self, op: str, indict: Dict[str, torch.Tensor], args, parent_args=None) -> Tuple[torch.Tensor]:
        res = None
        inputs = list(indict.values())
        if op == "If":
            if inputs[0].item():
                res = self.run_graph(args["then_branch"], indict, parent_args)
            else:
                res = self.run_graph(args["else_branch"], indict, parent_args)
        elif hasattr(ops_wgpu, op):
            res = getattr(ops_wgpu, op)(*inputs, args=args)
        elif hasattr(ops_torch, op):
            if op not in missing_ops:
                missing_ops.add(op)
            res = getattr(ops_torch, op)(*inputs, args=args)
        else:
            raise NotImplementedError(f"Op Not implemented: {op}")

        return res if isinstance(res, tuple) else (res,)

    def run_graph(self, graph: GraphProto, tensors: Dict[str, torch.Tensor], attrs=None):
        for node in graph.node:
            inputs = {x: tensors[x] for x in node.input}
            assert len(node.input) == len(inputs)
            attributes = (
                {x.name: parse_attribute(x, attrs) for x in node.attribute}
                if len(node.attribute)
                else {}
            )

            # TODO: Implement a run and get the outputs        # Running logic
            if node.op_type in self.funcs:
                if DEBUG:
                    print(f"Running {node.op_type} inputs={format_inputs(inputs)} kwargs={attributes} outputs={node.output}")
                output = self.run_func(self.funcs[node.op_type], inputs, attributes)
            else:
                if DEBUG:
                    print(
                    f"Running {node.op_type} inputs={format_inputs(inputs)} kwargs={attributes} outputs={node.output}"
                )
                output = self.run_op(node.op_type, inputs, attributes)

            assert len(output) == len(
                node.output
            ), f"Output length mismatch: {len(output)} != {len(node.output)}"
            for name, x in zip(node.output, output):
                tensors[name] = x

        outputs = []
        for out in graph.output:
            expected = parse_value_info_proto(out.type)
            assert out.name in tensors, f"Output not found: {out.name}"
            got = tensors[out.name]
            if expected is None:
                outputs.append(got)
                continue
            assert expected.shape == got.shape, f"Shape mismatch: {expected.shape} != {got.shape}"
            assert expected.dtype == got.dtype, f"Data Type mismatch: {expected.dtype} != {got.dtype}"
            # print(f"Output: {out.name} Shape: {got.shape} Data Type: {got.dtype}")
            # np.testing.assert_allclose(expected.numpy(), got.numpy())
            outputs.append(got)
        return tuple(outputs)
    
    def missing(self):
        return missing_ops
    
    def __call__(self, args: Dict[str, Any]):
        self.tensors.update(args)
        return self.run_graph(self.model.graph, self.tensors)