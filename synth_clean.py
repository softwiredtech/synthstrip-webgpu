import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import surfa as sf
from typing import List
from surfa.image.interp import interpolate
import scipy
import math

def _distance(x: sf.image.Volume):
    sampling = x.geom.voxsize[:x.basedim]
    dt = lambda z: scipy.ndimage.distance_transform_edt(1 - z, sampling=sampling)
    return _stack([x.new(dt(_framed_data(x)[..., i])) for i in range(x.nframes)])

def extend_sdt(sdt: sf.image.framed.Volume, border=1) -> sf.image.framed.Volume:
    if border < int(sdt.max()):
        return sdt

    # Find bounding box.
    mask = sdt < 1
    keep = np.nonzero(mask)
    low = np.min(keep, axis=-1)
    upp = np.max(keep, axis=-1)

    # Add requested border.
    gap = int(border + 0.5)
    low = (max(i - gap, 0) for i in low)
    upp = (min(i + gap, d - 1) for i, d in zip(upp, mask.shape))

    # Compute EDT within bounding box. Keep interior values.
    ind = tuple(slice(a, b + 1) for a, b in zip(low, upp))
    out = np.full_like(sdt, fill_value=100)
    out[ind] = _distance(sf.Volume(mask[ind]))
    out[keep] = sdt[keep]

    return sdt.new(out)


class StripModel(nn.Module):
    def __init__(
        self,
        nb_features: int = 16,
        nb_levels: int = 7,
        feat_mult: int = 2,
        max_features: int = 64,
        nb_conv_per_level: int = 2,
        max_pool: int = 2,
        return_mask=False,
    ):
        super().__init__()
        # build feature list automatically
        feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
        feats = np.clip(feats, 1, max_features)
        # extract any surplus (full resolution) decoder convolutions
        enc_nf = np.repeat(feats[:-1], nb_conv_per_level) 
        dec_nf = np.repeat(np.flip(feats), nb_conv_per_level)
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        self.pooling = [nn.MaxPool3d(s) for s in max_pool]
        self.upsampling = [
            nn.Upsample(scale_factor=s, mode="nearest") for s in max_pool
        ]

        # configure encoder (down-sampling path)
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if level < (self.nb_levels - 1):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(prev_nf, nf))
            prev_nf = nf

        # final convolutions
        if return_mask:
            self.remaining.append(ConvBlock(prev_nf, 2, activation=None))
            self.remaining.append(nn.Softmax(dim=1))
        else:
            self.remaining.append(ConvBlock(prev_nf, 1, activation=None))

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x.squeeze()


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride=1,
        activation="leaky",
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2) if activation == "leaky" else None

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        return out if self.activation is None else self.activation(out)


def _reshape(x: sf.image.Volume, shape: List[int]):
    shape = shape[:x.basedim]

    if np.array_equal(x.baseshape, shape):
        return x

    tdelta = (torch.Tensor(shape) - torch.Tensor(x.baseshape)) / 2
    tlow = tdelta.floor().int()
    thigh = tdelta.ceil().int()

    tc_low = tlow.clip(0)
    tc_high = thigh.clip(0)
    # TODO: Clean this up
    tpadding = ([z for y in zip(tc_low.tolist(), tc_high.tolist()) for z in y])
    tc_data = torch.nn.functional.pad(torch.from_numpy(_framed_data(x).T), tpadding, mode='constant')
    tc_data = tc_data.permute(*torch.arange(tc_data.ndim - 1, -1, -1))

    tcropping = tuple([slice(a, b) for a, b in zip((thigh.neg().clip(0)).int().tolist(), (torch.Tensor(list(tc_data.shape[:3])) - tlow.neg().clip(0)).int().tolist())])
    tc_data = tc_data[tcropping]
    return tc_data.squeeze()

# Framed data is just the data with a last frame dimension
# if data is shape (3, 3, 3) then framed data is shape (3, 3, 3, 1)
def _framed_data(x: sf.image.Volume):
    arr = x.data
    for _ in range(x.basedim + 1 - x.data.ndim):
        arr = np.expand_dims(arr, axis=-1)
    return arr

def _bbox(x: sf.image.Volume):
    mask = _framed_data(x).max(-1) > 0
    if not np.any(mask):
        return tuple([slice(0, s) for s in mask.shape])
    from scipy.ndimage import find_objects
    return find_objects(mask)[0]

from surfa.transform.geometry import ImageGeometry

def _rotation_matrix_to_orientation(matrix: np.array) -> str:
    matrix = matrix[:3, :3]
    orientation = ''
    for i in range(3):
        sag, cor, ax = matrix[:, i]
        if np.abs(sag) > np.abs(cor) and np.abs(sag) > np.abs(ax):
            orientation += 'R' if sag > 0 else 'L'
        elif np.abs(cor) > np.abs(ax):
            orientation += 'A' if cor > 0 else 'P'
        else:
            orientation += 'S' if ax > 0 else 'I'
    return orientation

def _orientation_to_rotation_matrix(orientation):
    matrix = np.zeros((3, 3))
    for i, c in enumerate(orientation.upper()):
        matrix[:3, i] -= [c == x for x in 'LPI']
        matrix[:3, i] += [c == x for x in 'RAS']
    return matrix

def _orientation(x: sf.image.Volume, orientation: str):
    trg_orientation = orientation.upper()
    src_orientation = _rotation_matrix_to_orientation(x.geom.vox2world.matrix).upper()
    if trg_orientation == src_orientation:
        return x
    
    # extract world axes
    get_world_axes = lambda aff: np.argmax(np.absolute(np.linalg.inv(aff)), axis=0)
    trg_matrix = _orientation_to_rotation_matrix(trg_orientation)
    src_matrix = _orientation_to_rotation_matrix(src_orientation)
    world_axes_trg = get_world_axes(trg_matrix[:x.basedim, :x.basedim])
    world_axes_src = get_world_axes(src_matrix[:x.basedim, :x.basedim])

    voxsize = np.asarray(x.geom.voxsize)
    voxsize = voxsize[world_axes_src][world_axes_trg]

    # initialize new
    data = x.data.copy()
    affine = x.geom.vox2world.matrix.copy()

    # align axes
    affine[:, world_axes_trg] = affine[:, world_axes_src]
    for i in range(x.basedim):
        if world_axes_src[i] != world_axes_trg[i]:
            data = np.swapaxes(data, world_axes_src[i], world_axes_trg[i])
            swapped_axis_idx = np.where(world_axes_src == world_axes_trg[i])
            world_axes_src[swapped_axis_idx], world_axes_src[i] = world_axes_src[i], world_axes_src[swapped_axis_idx]

    # align directions
    dot_products = np.sum(affine[:3, :3] * trg_matrix[:3, :3], axis=0)
    for i in range(x.basedim):
        if dot_products[i] < 0:
            data = np.flip(data, axis=i)
            affine[:, i] = - affine[:, i]
            affine[:3, 3] = affine[:3, 3] - affine[:3, i] * (data.shape[i] - 1)

    # update geometry
    target_geom = ImageGeometry(
        shape=data.shape[:3],
        vox2world=affine,
        voxsize=voxsize)
    return x.new(data, target_geom)
import sys

def interp(source: np.ndarray, method: str, target_shape: List[int], affine=None, fill = 0):
    if not source.flags.c_contiguous and not source.flags.f_contiguous:
        source = np.asarray(source, order='F')
    swap_byteorder = sys.byteorder == 'little' and '>' or '<'
    source = source.byteswap().newbyteorder() if source.dtype.byteorder == swap_byteorder else source
    if method == 'nearest':
        return _interpolate_nearest(source, target_shape, fill_value=fill, affine=affine)
    elif method == 'linear':
        return interp_3d_contiguous_linear(source, target_shape, fill_value=fill, affine=affine)

def interp_3d_contiguous_linear(source: np.ndarray, target_shape: List[int], fill_value, affine=None):
    # dimensions of the source image
    sx_max_idx = source.shape[0] - 1
    sy_max_idx = source.shape[1] - 1
    sz_max_idx = source.shape[2] - 1
    frames = source.shape[3]

    # target image
    x_max = target_shape[0]
    y_max = target_shape[1]
    z_max = target_shape[2]

    # intermediate variables
    x, y, z, f = 0, 0, 0, 0
    v = 0.0
    sx, sy, sz = 0.0, 0.0, 0.0
    ix, iy, iz = 0, 0, 0
    sx_low, sy_low, sz_low = 0, 0, 0
    sx_high, sy_high, sz_high = 0, 0, 0
    dsx, dsy, dsz = 0.0, 0.0, 0.0
    w0, w1, w2, w3, w4, w5, w6, w7 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # allocate the target image
    target = np.zeros([x_max, y_max, z_max, frames], dtype=np.float32, order='F')
    target_view = target

    # extract affine matrix values
    mat00, mat01, mat02, mat03 = 0.0, 0.0, 0.0, 0.0
    mat10, mat11, mat12, mat13 = 0.0, 0.0, 0.0, 0.0
    mat20, mat21, mat22, mat23 = 0.0, 0.0, 0.0, 0.0
    if affine is not None:
        mat00 = affine[0, 0]
        mat01 = affine[0, 1]
        mat02 = affine[0, 2]
        mat03 = affine[0, 3]
        mat10 = affine[1, 0]
        mat11 = affine[1, 1]
        mat12 = affine[1, 2]
        mat13 = affine[1, 3]
        mat20 = affine[2, 0]
        mat21 = affine[2, 1]
        mat22 = affine[2, 2]
        mat23 = affine[2, 3]

    # loop over each voxel in the target image
    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):
                ix = x
                iy = y
                iz = z

                if affine is not None:
                    sx = (mat00 * ix) + (mat01 * iy) + (mat02 * iz) + mat03
                    sy = (mat10 * ix) + (mat11 * iy) + (mat12 * iz) + mat13
                    sz = (mat20 * ix) + (mat21 * iy) + (mat22 * iz) + mat23
                else:
                    sx = ix
                    sy = iy
                    sz = iz

                # get low and high coords
                sx_low = int(math.floor(sx))
                sy_low = int(math.floor(sy))
                sz_low = int(math.floor(sz))

                # check coordinate limits
                if sx_low < 0 or sx_low > sx_max_idx or \
                   sy_low < 0 or sy_low > sy_max_idx or \
                   sz_low < 0 or sz_low > sz_max_idx:
                    for f in range(frames):
                        target_view[x, y, z, f] = fill_value
                    continue

                # make sure high value does not exceed limit
                sx_high = sx_low
                sy_high = sy_low
                sz_high = sz_low
                if sx_low != sx_max_idx:
                    sx_high += 1
                if sy_low != sy_max_idx:
                    sy_high += 1
                if sz_low != sz_max_idx:
                    sz_high += 1

                # get coordinate diff
                sx -= sx_low
                sy -= sy_low
                sz -= sz_low
                dsx = 1.0 - sx
                dsy = 1.0 - sy
                dsz = 1.0 - sz

                # compute weights
                w0 = dsx * dsy * dsz
                w1 = sx  * dsy * dsz
                w2 = dsx * sy  * dsz
                w3 = dsx * dsy * sz
                w4 = sx  * dsy * sz
                w5 = dsx * sy  * sz
                w6 = sx  * sy  * dsz
                w7 = sx  * sy  * sz

                # interpolate for each frame
                for f in range(frames):
                    v = w0 * source[sx_low , sy_low , sz_low , f] + \
                        w1 * source[sx_high, sy_low , sz_low , f] + \
                        w2 * source[sx_low , sy_high, sz_low , f] + \
                        w3 * source[sx_low , sy_low , sz_high, f] + \
                        w4 * source[sx_high, sy_low , sz_high, f] + \
                        w5 * source[sx_low , sy_high, sz_high, f] + \
                        w6 * source[sx_high, sy_high, sz_low , f] + \
                        w7 * source[sx_high, sy_high, sz_high, f]
                    target_view[x, y, z, f] = v

    return target

def _interpolate_nearest(source: np.array, target_shape: List[int], fill_value, affine = None):
    # dimensions of the source image
    sx_max = source.shape[0]
    sy_max = source.shape[1]
    sz_max = source.shape[2]
    frames = source.shape[3]

    # target image
    x_max = target_shape[0]
    y_max = target_shape[1]
    z_max = target_shape[2]

    # intermediate variables
    x, y, z, f = 0, 0, 0, 0
    sx, sy, sz = 0., 0., 0.
    sx_idx, sy_idx, sz_idx = 0, 0, 0

    target = np.zeros([x_max, y_max, z_max, frames], dtype=np.float32, order='F')
    target_view = target
    # extract affine matrix values
    mat00, mat01, mat02, mat03 = 0.0, 0.0, 0.0, 0.0
    mat10, mat11, mat12, mat13 = 0.0, 0.0, 0.0, 0.0
    mat20, mat21, mat22, mat23 = 0.0, 0.0, 0.0, 0.0
    if affine is not None:
        mat00 = affine[0, 0]
        mat01 = affine[0, 1]
        mat02 = affine[0, 2]
        mat03 = affine[0, 3]
        mat10 = affine[1, 0]
        mat11 = affine[1, 1]
        mat12 = affine[1, 2]
        mat13 = affine[1, 3]
        mat20 = affine[2, 0]
        mat21 = affine[2, 1]
        mat22 = affine[2, 2]
        mat23 = affine[2, 3]

    # loop over each voxel in the target image
    for z in range(z_max):
        for y in range(y_max):
            for x in range(x_max):

                if affine is not None:
                    sx = (mat00 * x) + (mat01 * y) + (mat02 * z) + mat03
                    sy = (mat10 * x) + (mat11 * y) + (mat12 * z) + mat13
                    sz = (mat20 * x) + (mat21 * y) + (mat22 * z) + mat23
                else:
                    sx = x
                    sy = y
                    sz = z

                # check coordinate limits
                if sx < 0 or sx >= sx_max or \
                   sy < 0 or sy >= sy_max or \
                   sz < 0 or sz >= sz_max:
                    for f in range(frames):
                        target_view[x, y, z, f] = fill_value
                    continue

                # round to nearest voxel
                sx_idx = int(round(sx))
                sy_idx = int(round(sy))
                sz_idx = int(round(sz))
                if sx_idx == sx_max: sx_idx -= 1
                if sy_idx == sy_max: sy_idx -= 1
                if sz_idx == sz_max: sz_idx -= 1

                # sample each frame
                for f in range(frames):
                    target_view[x, y, z, f] = source[sx_idx, sy_idx, sz_idx, f]

    return target


def _resize(x: sf.image.Volume):
    _shape = x.baseshape if len(x.baseshape) >= 3 else x.baseshape + [1] * (3 - len(x.baseshape)) 
    target_shape = tuple([math.ceil((gv * bs) / 1.) for gv, bs in zip(x.geom.voxsize, _shape)])

    target_geom = ImageGeometry(
        shape=target_shape,
        voxsize=1.,
        rotation=x.geom.rotation,
        center=x.geom.center)
    affine = x.geom.world2vox @ target_geom.vox2world
    # surfa/image/interp.pyx
    interped = interpolate(source=_framed_data(x), target_shape=target_shape, method="nearest", affine=affine.matrix)
    # interped = interp(_framed_data(x), "nearest", target_shape, affine=affine.matrix)
    # np.testing.assert_allclose(interped, cinterped, atol=1e-6, rtol=1e-6)
    # print(interped.shape, cinterped.shape)
    return x.new(interped, target_geom)

def _conform(x: sf.image.Volume):
    x = _orientation(x, "LIA")
    x = _resize(x)
    return x.astype(np.float32)


def _resample_like(x: sf.image.Volume, target: sf.image.Volume, fill = 0):
    affine = x.geom.world2vox @ target.geom.vox2world
    interped = interpolate(source=_framed_data(x), target_shape=target.geom.shape, method='linear', affine=affine.matrix, fill=fill)
    # interped = interp(_framed_data(x), "linear", target.geom.shape, affine=affine.matrix, fill=fill)
    # np.testing.assert_allclose(interped, cinterped, atol=1e-4, rtol=1e-5)

    return x.new(interped, target.geom)

def _connected_components(x: sf.image.Volume):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
        cc = [x.new(scipy.ndimage.label(_framed_data(x)[..., i])[0]) for i in range(x.nframes)]
        return _stack(cc)

def _connected_component_mask(x: sf.image.Volume, k=1, fill=False):
    cc = _connected_components(x)
    bincounts = [torch.bincount(torch.from_numpy(_framed_data(cc))[..., i].flatten())[1:] for i in range(cc.nframes)]
    topk = [(-bc).argsort()[:k] + 1 for bc in bincounts]
    mask = [torch.isin(torch.from_numpy(_framed_data(cc))[..., i], topk[i]) for i in range(x.nframes)]
    if fill:
        mask = [scipy.ndimage.binary_fill_holes(m.numpy()) for m in mask]
    return _stack([x.new(m) for m in mask])

def _stack(arrays: List[sf.image.Volume]):
    return arrays[0].new(np.concatenate([_framed_data(arr) for arr in arrays], axis=-1))

@torch.no_grad()
def run(in_image: str, modelfile: str = "./synthstrip.1.pt", saving: bool = False):
    model = StripModel()
    model.load_state_dict(torch.load(modelfile, map_location="cpu")["model_state_dict"])

    # load input volume
    image: sf.image.Volume = sf.load_volume(in_image)

    dist = []
    mask = []
    for f in range(image.nframes):
        frame = image.new(_framed_data(image)[..., f])
        conformed = _conform(frame)
        conformed = conformed[_bbox(conformed)]
        metadata = conformed.metadata

        target_shape = ((torch.Tensor(conformed.shape[:3]) / 64).ceil().type(torch.int32) * 64).clip(192, 320)
        conformed = _reshape(conformed, target_shape.tolist())
        x = conformed.data[None, None]
        x -= x.min()
        x = (x / x.quantile(.99)).clip(0, 1)
        if saving:
            np.save("./input_tensor.npy", x.numpy())
            onnx_program = torch.onnx.dynamo_export(model, x)
            onnx_program.save("bet.onnx")

        sdt = model(x).cpu().numpy()

        if saving:
            np.save("./out_tensor.npy", sdt)

        # POST PROCESSING
        sdt = extend_sdt(sf.image.framed.Volume(sdt, metadata=metadata), border=args.border)
        sdt = sdt.resample_like(image, fill=100)
        dist.append(sdt)
        mask.append(_connected_component_mask((sdt < args.border), k=1, fill=True))
    # combine frames and end line
    dist = sf.stack(dist)
    mask = sf.stack(mask)
    print("done")

    # write the masked output
    if args.out:
        image[mask == 0] = np.min([0, image.min()])
        image.save(args.out)
        print(f"Masked image saved to: {args.out}")

    # write the brain mask
    if args.mask:
        image.new(mask).save(args.mask)
        print(f"Binary brain mask saved to: {args.mask}")

    # write the distance transform
    if args.sdt:
        image.new(dist).save(args.sdt)
        print(f"Distance transform saved to: {args.sdt}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Robust, universal skull-stripping for brain images of any type."
    )
    p.add_argument(
        "-i", "--image", metavar="FILE", required=True, help="input image to skullstrip"
    )
    p.add_argument("-o", "--out", metavar="FILE", help="save stripped image to file")
    p.add_argument(
        "-m", "--mask", metavar="FILE", help="save binary brain mask to file"
    )
    p.add_argument(
        "-d", "--sdt", metavar="FILE", help="save distance transform to file"
    )
    p.add_argument(
        "-b",
        "--border",
        default=1,
        type=float,
        help="mask border threshold in mm, defaults to 1",
    )
    p.add_argument(
        "--no-csf", action="store_true", help="exclude CSF from brain border"
    )
    p.add_argument("--model", metavar="FILE", help="alternative model weights")
    p.add_argument("-s", "--saving", action="store_true", help="save the model to onnx format")
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        p.print_help()
        exit(1)
    args = p.parse_args()
    if not args.out and not args.mask and not args.sdt:
        print("Must provide at least one -o, -m, or -d output flag.")
        exit(1)

    torch.set_grad_enabled(False)

    run(args.image, saving=args.saving)
