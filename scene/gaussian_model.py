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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self._gaussian_scene_num = torch.empty(0)
        self._gaussian_uid = torch.empty(0, dtype=torch.int64)
        self._next_gaussian_uid = 0

        self.max_radii2D = torch.empty(0)
        self.tmp_radii = None
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # ExploreGS: seed explore를 쓸 때 world-space big point prune 기준 (원본 3DGS = 0.1)
        self.big_point_threshold = 0.1
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._gaussian_scene_num,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._gaussian_scene_num,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

        self.optimizer.load_state_dict(opt_dict)

        # Gaussian UIDs are not checkpointed; rebuild them so their
        # length matches the restored number of Gaussians.
        self.init_gaussian_uids()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_gaussian_scene_num(self):
        return self._gaussian_scene_num
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._gaussian_scene_num = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        self.init_gaussian_uids()
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def init_gaussian_uids(self):
        """Assign stable IDs that remain attached to Gaussians through pruning."""
        n = self._xyz.shape[0]
        self._gaussian_uid = torch.arange(
            n, dtype=torch.int64, device=self._xyz.device
        )
        self._next_gaussian_uid = int(n)

    def append_gaussian_uids(self, count):
        if count <= 0:
            return
        new_ids = torch.arange(
            self._next_gaussian_uid,
            self._next_gaussian_uid + count,
            dtype=torch.int64,
            device=self._xyz.device,
        )
        self._gaussian_uid = torch.cat((self._gaussian_uid, new_ids))
        self._next_gaussian_uid += count

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append("gaussian_scene_num")
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        gaussian_scene_num = (self._gaussian_scene_num.detach().cpu().numpy())
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, gaussian_scene_num), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._gaussian_scene_num = torch.tensor(np.asarray(plydata.elements[0]["gaussian_scene_num"]), dtype=torch.float,device="cuda").reshape(-1, 1)
        self.active_sh_degree = self.max_sh_degree

        self.init_gaussian_uids()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._gaussian_scene_num =  self._gaussian_scene_num[valid_points_mask]
        self._gaussian_uid = self._gaussian_uid[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_gaussian_scene_num):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._gaussian_scene_num = torch.cat((self._gaussian_scene_num,new_gaussian_scene_num))

        num_new = new_xyz.shape[0]
        device = self._xyz.device
        self.append_gaussian_uids(num_new)
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_points = self.get_xyz.shape[0]

        # grads only contains gradients of Gaussians that existed before cloning,
        # so set the gradient of newly cloned Gaussians to 0.
        padded_grad = torch.zeros(n_points, device=self.get_xyz.device)
        grad_values = grads.reshape(-1)
        padded_grad[:grad_values.shape[0]] = grad_values

        max_scale = self.get_scaling.max(dim=1).values
        flagged = self._gaussian_scene_num[:, 0] > 0.5

        normal_split_limit = self.percent_dense * scene_extent
        flagged_split_limit = normal_split_limit  # *0.5 Flagged-specific scale threshold

        split_limit = torch.where(
            flagged,
            torch.full_like(max_scale, flagged_split_limit),
            torch.full_like(max_scale, normal_split_limit),
        )

        selected_pts_mask = (
            (padded_grad >= grad_threshold)
            & (max_scale > split_limit)
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)

        rots = build_rotation(
            self._rotation[selected_pts_mask]
        ).repeat(N, 1, 1)

        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask].repeat(N, 1)
        )

        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        # Current behavior: child Gaussians inherit the parent's flag
        new_gaussian_scene_num = self._gaussian_scene_num[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_tmp_radii,
            new_gaussian_scene_num,
        )

        # Remove split parent Gaussians while keeping new child Gaussians
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(
                N * selected_pts_mask.sum(),
                device=self.get_xyz.device,
                dtype=torch.bool,
    ),
        ))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        max_scale = self.get_scaling.max(dim=1).values
        flagged = self._gaussian_scene_num[:, 0] > 0.5

        normal_clone_limit = self.percent_dense * scene_extent
        flagged_clone_limit = normal_clone_limit  # 0.5 *  Same flagged threshold as split

        clone_limit = torch.where(
            flagged,
            torch.full_like(max_scale, flagged_clone_limit),
            torch.full_like(max_scale, normal_clone_limit),
        )

        selected_pts_mask = (
            (torch.norm(grads, dim=-1) >= grad_threshold)
            & (max_scale <= clone_limit)
        )

        # Clone copies to the exact same location as the original
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        # Current behavior: clone copies inherit the parent's flag
        new_gaussian_scene_num = self._gaussian_scene_num[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_tmp_radii,
            new_gaussian_scene_num,
        )

    @torch.no_grad()
    def random_split_explore(
        self,
        viewspace_abs_tensor,
        viewspace_point_tensor,
        max_grad,
        extent,
        N=2,
    ):
        """
        optimizer.step() 이후에 호출한다 (이번 렌더의 viewspace grad 사용).
        |grad(abs)| >= 3*max_grad 이고 |grad(abs)| >= 3*|grad(points)| 이며
        충분히 큰 Gaussian을 3DGS split 방식으로 N개로 쪼갠다.
        반환: split된 부모 개수
        """
        point_count = self._xyz.shape[0]
        if point_count == 0:
            return 0
        if N < 2:
            raise ValueError("N must be at least 2 for splitting")

        abs_grad = viewspace_abs_tensor.grad
        pt_grad = viewspace_point_tensor.grad
        if abs_grad is None or pt_grad is None:
            return 0

        grad_norm = torch.nan_to_num(torch.norm(abs_grad[:, :2], dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        real_grad_norm = torch.nan_to_num(torch.norm(pt_grad[:, :2], dim=-1), nan=0.0, posinf=0.0, neginf=0.0)

        selected_pts_mask = (
            (grad_norm >= max_grad * 3)
            & (grad_norm >= real_grad_norm * 3)
            & (self.get_scaling.max(dim=1).values > self.percent_dense * extent)
        )

        split_count = int(selected_pts_mask.sum().item())
        if split_count == 0:
            return 0

        # densification_postfix가 통계를 전역 0으로 "새로 할당"하므로(제자리 수정 아님)
        # clone 없이 참조만 잡아두면 된다.
        old_gradient_accum = self.xyz_gradient_accum
        old_denom = self.denom
        old_max_radii2D = self.max_radii2D

        # tmp_radii는 densify_and_prune 안에서만 의미가 있다.
        # 여기서는 postfix/prune_points가 길이를 맞추는 데만 필요하므로 0으로 채운다.
        self.tmp_radii = torch.zeros(point_count, dtype=self._xyz.dtype, device=self._xyz.device)

        # 3DGS split: 부모 분포 N(mu, Sigma)에서 N개 샘플, scale은 1/(0.8N)
        selected_scaling = self.get_scaling[selected_pts_mask]
        stds = selected_scaling.repeat(N, 1)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rotations = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            self._features_dc[selected_pts_mask].repeat(N, 1, 1),
            self._features_rest[selected_pts_mask].repeat(N, 1, 1),
            self._opacity[selected_pts_mask].repeat(N, 1),
            self.scaling_inverse_activation(selected_scaling.repeat(N, 1) / (0.8 * N)),
            self._rotation[selected_pts_mask].repeat(N, 1),
            torch.zeros(N * split_count, dtype=self._xyz.dtype, device=self._xyz.device),
            self._gaussian_scene_num[selected_pts_mask].repeat(N, 1),
        )

        # 기존 통계 복원 + 자식은 부모 통계 상속 (interval 기반 ADC 유지)
        self.xyz_gradient_accum = torch.cat((old_gradient_accum, old_gradient_accum[selected_pts_mask].repeat(N, 1)), dim=0)
        self.denom = torch.cat((old_denom, old_denom[selected_pts_mask].repeat(N, 1)), dim=0)
        self.max_radii2D = torch.cat((old_max_radii2D, old_max_radii2D[selected_pts_mask].repeat(N)), dim=0)

        # 부모 제거
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * split_count, dtype=torch.bool, device=self._xyz.device),
        ))
        self.prune_points(prune_filter)
        self.tmp_radii = None

        return split_count


    # ------------------------------------------------------------------
    # ExploreGS random seed (기존 Gaussian 재사용 버전)
    # ------------------------------------------------------------------
    def _new_seed_parameters(self, new_xyz, scale_factor=1.0, opacity_factor=0.5):
        """ExploreGS new_gaussian_parameter (+ 랜덤 색). 거리는 월드 원점 기준."""
        num_explore = new_xyz.shape[0]
        device = new_xyz.device

        new_features_dc = RGB2SH(torch.rand((num_explore,
                                             self._features_dc.shape[1],
                                             self._features_dc.shape[2]), device=device))
        new_features_rest = torch.zeros((num_explore,
                                         self._features_rest.shape[1],
                                         self._features_rest.shape[2]), device=device)

        new_xyz_norm = torch.norm(new_xyz, dim=1).unsqueeze(1)
        G_xyz_norm = torch.norm(self._xyz.detach(), dim=1).unsqueeze(1)
        scale_per_dist = torch.mean(self.get_scaling.detach() / G_xyz_norm)

        new_scaling = torch.ones((num_explore, 3), device=device) * \
            self.scaling_inverse_activation(scale_factor * scale_per_dist * new_xyz_norm)
        new_opacities = torch.ones((num_explore, 1), device=device) * \
            (new_xyz_norm / torch.max(new_xyz_norm)) * opacity_factor + 0.2
        new_opacities = self.inverse_opacity_activation(new_opacities)
        new_rotation = torch.randn((num_explore, 4), device=device)

        return new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation

    @torch.no_grad()
    def seed_explore(self, num_explore=100, percentage_densify=None, seed_scene_num=1.0, **kwargs):
        """
        ExploreGS seed_explore 변형: 새 위치를 뽑아 Gaussian을 추가하는 대신,
        기존 Gaussian 중 무작위 num_explore개를 골라 그 자리에서 seed 파라미터로 덮어쓴다.
        - 위치(xyz): 그대로 유지 (= seed 위치로 사용)
        - 색: 랜덤 RGB, 고차 SH = 0
        - scale: 등방성, mean(s/|x|) * |x|           (ExploreGS와 동일, 월드 원점 기준)
        - opacity: |x| / max(고른 것들의 |x|) * 0.5 + 0.2
        - rotation: 랜덤
        - 고른 Gaussian의 Adam 모멘트(xyz 제외)와 max_radii2D는 0으로 초기화
        Gaussian 개수/순서는 바뀌지 않는다.
        반환: 바뀐 개수
        """
        n = self._xyz.shape[0]
        if num_explore is None:
            if percentage_densify is None:
                raise ValueError('Either num_explore or percentage_densify must be provided.')
            num_explore = int(n * percentage_densify)
        if num_explore <= 0 or n == 0:
            return 0

        device = self._xyz.device
        idx = torch.randperm(n, device=device)[:min(int(num_explore), n)]
        seed_xyz = self._xyz.detach()[idx]

        f_dc, f_rest, opac, scal, rot = self._new_seed_parameters(seed_xyz)

        self._features_dc.data[idx] = f_dc
        self._features_rest.data[idx] = f_rest
        self._opacity.data[idx] = opac
        self._scaling.data[idx] = scal
        self._rotation.data[idx] = rot

        for param in (self._features_dc, self._features_rest, self._opacity,
                      self._scaling, self._rotation):
            state = self.optimizer.state.get(param, None)
            if state is None:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state and state[key].shape[0] == n:
                    state[key][idx] = 0.0

        # 크기가 바뀌었으니 이전 화면 크기 기록으로 prune되지 않게 초기화
        self.max_radii2D[idx] = 0.0
        self._gaussian_scene_num[idx] = float(seed_scene_num)
        return int(idx.numel())

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > self.big_point_threshold * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
