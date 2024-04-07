import numpy as np
import onnxruntime as ort

if __name__ == "__main__":
    input_t = np.load("./input_tensor.npy")
    out_t = np.load("./out_tensor.npy")
    ort_session = ort.InferenceSession("./bet.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: input_t}
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(out_t, ort_outs[0].squeeze(), atol=1e-4, rtol=1e-4)
    print("All tests passed!")