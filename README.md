# SynthStrip WebGPU

This project is a port of the [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) neural network based brain extraction tool to WebGPU.

## Porting steps

- Export synthstrip model to onnx
- Use [wonnx](https://github.com/webonnx/wonnx) to run the onnx model through webgpu
- Port `surfa.Volume`, `surfa.load_volumes`, and `surfa.stack` for pre/post-processing
- Deploy a demo site that takes in a full NIfTI volume of the head, and outputs the brain extrected nifti volume
- The demo site doesn't have to do rendering, just NIfTI I/O

NOTE: [surfa](https://github.com/freesurfer/surfa) might not work from `pip install`, for me it only worked through `git clone`, then pip install from the local repo

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
