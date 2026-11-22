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
import inspect
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
# train.py와 동일하게 SH를 dc/rest로 나누지 않고 넘긴다.
# (수정된 rasterizer의 forward()는 'dc' 인자를 받지 않는다.)
SEPARATE_SH = False

try:
    from lpipsPyTorch.modules.lpips import LPIPS
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False

_lpips_model = None

# 저장소 버전에 따라 render()가 받는 인자가 다르므로, 지원하는 것만 넘긴다.
_RENDER_PARAMS = set(inspect.signature(render).parameters)

def _render_kwargs(train_test_exp, separate_sh):
    kwargs = {}
    if "use_trained_exp" in _RENDER_PARAMS:
        kwargs["use_trained_exp"] = train_test_exp
    if "separate_sh" in _RENDER_PARAMS:
        kwargs["separate_sh"] = separate_sh
    return kwargs

def _get_lpips_model():
    """LPIPS(VGG) 모델을 한 번만 만들어 재사용한다 (metrics.py와 동일한 net_type='vgg')."""
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = LPIPS("vgg", "0.1").to("cuda").eval()
    return _lpips_model

def _quantize(img):
    """save_image로 저장한 8bit PNG를 다시 읽은 것과 같은 값으로 맞춘다 (metrics.py와 수치 일치용)."""
    return torch.round(img.clamp(0.0, 1.0) * 255.0) / 255.0


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, skip_lpips=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    use_lpips = LPIPS_AVAILABLE and not skip_lpips
    if not LPIPS_AVAILABLE and not skip_lpips:
        print("lpipsPyTorch를 불러올 수 없어 LPIPS는 건너뜁니다.")

    psnrs, ssims, lpipss = [], [], []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, **_render_kwargs(train_test_exp, separate_sh))["render"]
        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # Metrics (저장된 PNG 기준과 동일하게 8bit 양자화 후 계산)
        r = _quantize(rendering).unsqueeze(0).cuda()
        g = _quantize(gt).unsqueeze(0).cuda()
        psnrs.append(psnr(r, g).mean().item())
        ssims.append(ssim(r, g).mean().item())
        if use_lpips:
            lpipss.append(_get_lpips_model()(r, g).mean().item())

    if len(psnrs) == 0:
        print(f"[{name}] 카메라가 없어 metric을 계산하지 않았습니다.")
        return

    n = len(psnrs) 
    print(f"\n[{name} | iter {iteration} | {n} views]")
    print("  SSIM : {:>12.7f}".format(sum(ssims) / n))
    print("  PSNR : {:>12.7f}".format(sum(psnrs) / n))
    if use_lpips:
        print("  LPIPS: {:>12.7f}".format(sum(lpipss) / n))
    print("  #Gaussians: {}".format(gaussians.get_xyz.shape[0]))
    print("")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, skip_lpips: bool = False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        # 구버전 3DGS 기반 저장소에는 train_test_exp 옵션이 없다.
        train_test_exp = getattr(dataset, "train_test_exp", False)
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, train_test_exp, separate_sh, skip_lpips)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, train_test_exp, separate_sh, skip_lpips)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SEPARATE_SH, args.skip_lpips)
