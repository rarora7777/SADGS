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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.opacity_lr = 0.05 # 0.025 
        self.scaling_lr = 0.01 # 0.005
        self.rotation_lr = 0.002 # 0.001
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016 #0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025 
        self.shfeature_lr = 0.005 
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        
        self.opacity_reset_interval = 3000
        self.opacity_reset_decay = 0.1
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.densify_grad_abs_threshold = 0.0004

        self.prune_until_iter = 25000
        self.min_weight = 0.7

        self.prune_from_iter = 6000
        self.prune_until_iter = 30_000
        self.prune_interval = 3000
        self.densify_prune_ratio = 0.45
        self.after_densify_prune_ratio = 0.01
        
        

        # fastgs parameters
        self.loss_thresh = 0.02
        self.grad_abs_thresh = 0.0002  
        self.highfeature_lr = 0.005 # 0.005
        self.lowfeature_lr = 0.0025 # 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001
        self.mult = 0.7      # multiplier for the compact box to control the tile number of each splat

        # frequency parameters
        self.lambda_l2 = 2.0
        self.lambda_tone=0.
        self.lambda_freq = 0. # Frequency-based loss weight
        self.st_levels = 4
        self.st_mode = "v1" # "v1" or "v2"
        
        self.freq_grad_threshold = 0.00002
        self.importance_score_threshold = 0.5
        self.min_contribution_threshold = 0.1
        self.importance_error_threshold = 0.06
        self.random_background = False
        self.optimizer_type = "hybrid"
        self.sample_bbox_faces = False
        self.warmup_densification = False
        self.camera_sampling = "random"
        self.compute_3d_filter = False
        
        # expansion parameters
        self.tau_expand = 1.0
        self.adaptive_clone = False # [NEW] Enable frequency-aware scale expansion before cloning
        self.expansion_speed = 0.1
        self.ks_scale_power = 1.0  # Exponent for scale division when splitting (new_scale = old_scale / k^scale_power)
        
        # initialization parameters
        self.sample_far_plane = False
        self.far_plane_dist = 10.0
        self.far_plane_res = 32
        self.densification_window_width = 200  # Number of iterations to run densification per window
        self.freq_opacity_threshold = 0.05
        self.freq_transmittance_threshold = 0.0
        self.batch_size = 1  # Number of cameras to accumulate gradients from before optimizer step
        self.split_ratio_threshold = 0.8
        self.prune_ratio_threshold = 0.8
        self.eta_compute_mode = "wavelength" # "wavelength" or "projection"
        
        # Multi-clone parameters
        self.clone_target_eta = 1.0  # Target eta for low-frequency areas (lower = more clones)
        self.scale_rotation_scheduler = False
        self.max_clones_per_axis = 8 # Maximum number of clones per axis
        self.adam_eps_order = 8



        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
