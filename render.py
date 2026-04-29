#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_structgs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    total_time = 0.0

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if args.render_extra:
        extras_path = os.path.join(model_path, name, "ours_{}".format(iteration), "extras")
        makedirs(extras_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = time.time()
        render_pkg = render_structgs(view, gaussians, pipeline, background, args.mult, compute_extra=args.render_extra)
        rendering = render_pkg["render"]
        end_time = time.time()
        total_time += (end_time - start_time)
        gt = view.original_image[0:3, :, :].to(rendering.device)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if args.render_extra:
            # 1. Random color rendering
            num_gaussians = gaussians.get_xyz.shape[0]
            random_colors = torch.rand((num_gaussians, 3), device="cuda")
            render_pkg_random = render_structgs(view, gaussians, pipeline, background, args.mult, override_color=random_colors)
            rendering_random = render_pkg_random["render"]

            # 2. Process other maps
            depth = render_pkg["depth_map"]
            if len(depth.shape) == 2:
                depth = depth.unsqueeze(0)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)
            depth = depth.repeat(3, 1, 1) if depth.shape[0] == 1 else depth
            
            normal = render_pkg["normal_map"]
            if len(normal.shape) == 2:
                normal = normal.unsqueeze(0)
            normal = (normal + 1.0) / 2.0 # Assuming normals are in [-1, 1]
            normal = normal.repeat(3, 1, 1) if normal.shape[0] == 1 else normal
            
            error_map = torch.abs(rendering - gt)
            # error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-5)

            # 3. Create 2x2 grid
            # [Rendering_Random, Depth]
            # [Normal_Map, Error_Map]
            top_row = torch.cat([rendering_random, depth], dim=2)
            bottom_row = torch.cat([normal, error_map], dim=2)
            grid = torch.cat([top_row, bottom_row], dim=1)
            
            torchvision.utils.save_image(grid, os.path.join(extras_path, '{0:05d}'.format(idx) + ".png"))
    
    num_frames = len(views)
    avg_time = total_time / num_frames if num_frames > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"[{name}] Rendered {num_frames} frames in {total_time:.2f} seconds. Average FPS: {fps:.2f}")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_extra", action="store_true")
    parser.add_argument("--mult", type=float, default=0.5)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)