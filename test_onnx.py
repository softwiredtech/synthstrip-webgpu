
from onnx.executor import OnnxExecutor
import numpy as np
import torch

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ONNX model")
    parser.add_argument("--model", type=str, help="ONNX model file")
    args = parser.parse_args()
    input_t = torch.from_numpy(np.load("./input_tensor.npy"))
    expected = torch.from_numpy(np.load("./out_tensor.npy"))

    model = OnnxExecutor(args.model)
    model.init()
    outputs = model({"l_x_": input_t})
    
    print(f"FINISHED and got {len(outputs)}")
    np.testing.assert_allclose(outputs[0].numpy(), expected.numpy())