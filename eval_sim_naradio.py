#!/usr/bin/env python3

import os
import argparse
# from pathlib import Path
import numpy as np
# from tqdm import tqdm
import torch
# from torchvision import transforms
from gradslam.datasets.icl import ICLDataset
# from PIL import Image
# import torch.nn.functional as F
from rayfronts.image_encoders.naradio import NARadioEncoder
import yaml
import cv2
import time
import pickle as pkl


"""
eval_sim_naradio.py

Loop over images using the ICLDataset class from gradslam.datasets.icl.
Downscale the images to a given target dimension, extract vision embeddings
with NARadioEncoder, upscale embeddings back to the target resolution,
encode multiple text queries, compute cosine similarity maps, and save
similarity_{i}.npy files for each text query per image.

Usage:
    python eval_sim_naradio.py \
        --dataset_root /path/to/icl/dataset \
        --output_dir /path/to/output \
        --target_size 256 \
        --texts "a red chair|a blue table" \
        --device cuda
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", required=True, help="Root path to ICL dataset")
    parser.add_argument("--split", default="train", help="Split name for ICLDataset if supported")
    parser.add_argument("--export_path", required=True, help="Directory to save similarities")
    parser.add_argument("--target_size", required=False, type=int, nargs=2, help="(width, height)")
    parser.add_argument("-q", "--query", type=str, nargs='*',
                        help="optional query with similarity visualisation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # parser.add_argument("--max_images", type=int, default=None, help="Limit number of images (for debugging)")
    args = parser.parse_args()

    # device = torch.device(args.device)

    args.query = ["chair", "coffee machine", "detergent", "faucet", "milk", "paper towel", "sink", "socket", "sponge", "sugar", "table top", "towel"]

    config_path = os.path.join(args.log_path, "icl.yaml")
    with open(config_path, "r") as f:
        config_dict = yaml.full_load(f)
    basedir, sequence = os.path.split(args.log_path)
    dataset = ICLDataset(
        config_dict = config_dict,
        basedir = basedir,
        sequence = sequence,
        desired_height = config_dict["camera_params"]["image_height"],
        desired_width = config_dict["camera_params"]["image_width"],
        stride = 1,
    )

    if args.export_path is not None and args.query is not None:
        for qu in args.query:
            # export_path_sim = os.path.join(args.export_path, qu.replace(' ', '_'))
            os.makedirs(os.path.join(args.export_path, qu.replace(' ', '_')), exist_ok=True)

    # load encoder
    # encoder = NARadioEncoder()
    # encoder = encoder.to(device)
    # encoder.eval()

    # args.target_size # (width, height)

    # args.target_size = [512, 512]

    # args.target_size = [480, 640]

    resolution = args.target_size

    if resolution is None:
        resolution = [
            config_dict["camera_params"]["image_width"],
            config_dict["camera_params"]["image_height"],
        ]

    enc = NARadioEncoder(model_version="radio_v2.5-b", lang_model="siglip",
                         input_resolution=resolution)

    classes_nyu40 = [
        "unlabeled",
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "blinds",
        "desk",
        "shelves",
        "curtain",
        "dresser",
        "pillow",
        "mirror",
        "floormat",
        "clothes",
        "ceiling",
        "books",
        "refrigerator",
        "television",
        "paper",
        "towel",
        "showercurtain",
        "box",
        "whiteboard",
        "person",
        "nightstand",
        "toilet",
        "sink",
        "lamp",
        "bathtub",
        "bag",
        "otherstructure",
        "otherfurniture",
        "otherprop",
    ]

    # prompts = args.query
    prompts = classes_nyu40

    print("prompts", prompts)

    with torch.no_grad():
        prompt_embeddings = enc.encode_prompts(prompts)
        prompt_embeddings = torch.nn.functional.normalize(prompt_embeddings, dim=-1)
        print("prompt_embeddings", prompt_embeddings.shape)

        query_embeddings = dict(zip(prompts, prompt_embeddings.cpu().numpy(), strict=True))
        # print(query_embeddings)

        with open(os.path.join(args.export_path, "embeddings-NYU40-RF.pkl"), 'wb') as f:
            pkl.dump(query_embeddings, f, protocol=pkl.HIGHEST_PROTOCOL)

    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    device = "cuda"

    time_inf = list()

    for i in range(len(dataset)):
        tstart_load = time.time()
        _color, _depth, intrinsics, _pose = dataset[i]
        img_colour = np.rint(_color.cpu().numpy()).astype(np.uint8)
        # img_depth = (_depth.cpu().numpy()*1000).astype(np.uint16)
        # K = intrinsics.cpu().numpy()[:3, :3]
        # T = np.linalg.inv(_pose.cpu().numpy())

        # print(f"time load: {(time.time() - tstart_load):.3f}s")

        # print(img_colour.shape)

        input_image = img_colour

        with torch.no_grad():
            tstart = time.time()

            tensor_image = torch.from_numpy(input_image).permute(2, 0, 1)
            tensor_image = tensor_image.to(device).float() / 255.0
            tensor_image = torch.nn.functional.interpolate(
                tensor_image.unsqueeze(0), resolution, mode="bilinear", antialias=True)

            feat_map = enc.encode_image_to_feat_map(tensor_image)
            lang_aligned_feat_map = enc.align_spatial_features_with_language(feat_map)

            lang_aligned_feat_map = torch.nn.functional.interpolate(
                lang_aligned_feat_map, resolution, mode="bilinear", antialias=True)
            lang_aligned_feat_map = lang_aligned_feat_map.squeeze(0).permute(1, 2, 0)

            lang_aligned_feat_map = torch.nn.functional.normalize(lang_aligned_feat_map, dim=-1)

            time_encode = time.time() - tstart
            # print(f"time enc: {time_encode:.3f}s")

            time_inf.append(time_encode)

            # flatten for cosim
            lang_aligned_feat_map = lang_aligned_feat_map.reshape(resolution[0] * resolution[1], -1)

            # print(lang_aligned_feat_map.shape)

            similarity = [cosine_similarity(lang_aligned_feat_map, pe).reshape(resolution[1], resolution[0], -1) for pe in prompt_embeddings]

            # print(f"time enc+sim: {(time.time() - tstart):.3f}s")

            # similarity = torch.concat(similarity, dim=-1)
            # print("similarity", similarity)
            # print("similarity", similarity.shape)

            for j, prompt in enumerate(prompts):
                exppath = os.path.join(args.export_path, prompt.replace(' ', '_'))
                os.makedirs(exppath, exist_ok=True)
                np.save(os.path.join(exppath, f"similarity_{i}.npy"), similarity[j].cpu().numpy())

            # similarity0 = similarity[0].reshape(resolution[1], resolution[0], 1152)
            # print("similarity0", similarity0.shape)

            # print(f"time total: {(time.time() - tstart):.3f}s")

    print(f"avg. time embed: {np.mean(time_inf):.3f}s")

    return


if __name__ == "__main__":
    main()
