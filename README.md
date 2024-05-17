# SynthStrip WebGPU

This project is a work-in-progress port of the [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) neural network based brain extraction tool to WebGPU.

## Planned porting steps

- Export synthstrip model to onnx
- Use [wonnx](https://github.com/webonnx/wonnx) to run the onnx model through webgpu
- Port `surfa.Volume`, `surfa.load_volumes`, and `surfa.stack` for pre/post-processing
- Deploy a demo site that takes in a full NIfTI volume of the head, and outputs the brain extrected nifti volume

## Usage

Configure python environment

```shell
python -m venv venv
source ./venv/bin/activate
pip install -r ./requirements.txt
```

Run command

```shell
python ./synth_clean.py -i ./test_t1.nii -o ./out.nii -m ./mask.nii -d ./sdt.nii
```
