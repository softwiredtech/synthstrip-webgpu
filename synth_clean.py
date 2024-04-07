import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import surfa as sf


def extend_sdt(sdt: sf.image.framed.Volume, border=1):
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
    out[ind] = sf.Volume(mask[ind]).distance()
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

@torch.no_grad()
def run(in_image: str, modelfile: str = "./synthstrip.1.pt", saving: bool = False):
    model = StripModel()
    model.load_state_dict(torch.load(modelfile, map_location="cpu")["model_state_dict"])

    # load input volume
    image = sf.load_volume(in_image)
    print(f"Input image read from: {in_image}")

    # loop over frames (try not to keep too much data in memory)
    print(f"Processing frame (of {image.nframes}):", end=" ", flush=True)
    dist = []
    mask = []
    for f in range(image.nframes):
        # PREPROCESSING
        frame = image.new(image.framed_data[..., f])
        # conform, fit to shape with factors of 64
        conformed = frame.conform(
            voxsize=1.0, dtype="float32", method="nearest", orientation="LIA"
        ).crop_to_bbox()
        target_shape = np.clip(
            np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320
        )
        conformed = conformed.reshape(target_shape)
        conformed -= conformed.min()
        conformed = (conformed / conformed.percentile(99)).clip(0, 1)
        # END PREPROCESSING

        # Input 
        input_tensor = torch.from_numpy(conformed.data[None, None])
        
        if saving:
            np.save("./input_tensor.npy", input_tensor.numpy())
            onnx_program = torch.onnx.dynamo_export(model, input_tensor)
            onnx_program.save("bet.onnx")

        sdt = model(input_tensor).cpu().numpy()
        if saving:
            np.save("./out_tensor.npy", sdt)

        # POST PROCESSING
        sdt = extend_sdt(conformed.new(sdt), border=args.border)
        sdt = sdt.resample_like(image, fill=100)
        dist.append(sdt)
        mask.append((sdt < args.border).connected_component_mask(k=1, fill=True))

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
        sf.system.fatal("Must provide at least one -o, -m, or -d output flag.")
    torch.set_grad_enabled(False)

    run(args.image, saving=args.saving)
