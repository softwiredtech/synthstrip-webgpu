from onnx.onnx_pb2 import TensorProto 

tensor_type_cast = {
    TensorProto.BOOL: "?",
    TensorProto.UINT8: "B",
    TensorProto.UINT16: "H",
    TensorProto.UINT32: "I",
    TensorProto.UINT64: "Q",
    TensorProto.INT8: "b",
    TensorProto.INT16: "h",
    TensorProto.INT32: "i",
    TensorProto.INT64: "q",
    TensorProto.FLOAT16: "e",
    TensorProto.FLOAT: "f",
}

item_size = {
    "i": 4,
    "f": 4,
}