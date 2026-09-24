import argparse
import os
import sys
import subprocess
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import logging
import wandb

import torch
import torch.nn.functional as F
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, MeshGaussianModel
from gaussian_renderer import render
from utils.general_utils import read_obj, read_ply
from utils.smplx_deformer import SmplxDeformer
from utils.sh_utils import eval_sh
from utils.render_utils import render_mesh
from evaluation_video import encode_comparison
from collider_motion import load_raw_collider
from dataset_input import load_manifest
from material_progress import export_progress, restore_training_state, save_training_state
from fit_curves import render_body, render_material

import warp as wp
from warp_mpm.mpm_data_structure import (
    MPMStateStruct,
    MPMModelStruct,
)
from warp_mpm.mpm_solver import MPMWARP
from warp_mpm.initial_state import estimate_velocity, particle_velocities
from future_evaluation import simulate_future, load_driving_surface, score_evaluation


logger = get_logger(__name__, log_level="INFO")

def cycle(dl: torch.utils.data.DataLoader):
    while True:
        for data in dl:
            yield data

def convert_SH(
    shs_view,
    viewpoint_camera,
    pc: MeshGaussianModel,
    position: torch.tensor,
    rotation: torch.tensor = None,
):
    shs_view = shs_view.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = position - viewpoint_camera.camera_center.repeat(shs_view.shape[0], 1)
    if rotation is not None:
        dir_pp = torch.matmul(rotation, dir_pp.unsqueeze(2)).squeeze(2)

    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    return colors_precomp


class Trainer:
    def __init__(self, args, opt, pipe, run_eval):
        self.args = args
        os.environ["WANDB__SERVICE_WAIT"] = "600"

        logging_dir = os.path.join(args.output_dir, opt.save_name)
        accelerator_project_config = ProjectConfiguration(logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        set_seed(opt.seed + accelerator.process_index)
        print("process index", accelerator.process_index)
        if accelerator.is_main_process:
            output_path = os.path.join(logging_dir, f"seed{opt.seed}")
            os.makedirs(output_path, exist_ok=True)
            self.output_path = output_path

        # setup the dataset
        self.run_eval = run_eval
        self.split_idx_path = args.split_idx_path
        self.verts_start_idx = args.verts_start_idx
        self.model_path = args.model_path
        self.load_gaussians(args)
        if args.raw_collider_path:
            self.load_raw_collider()
        else:
            if args.dataset_type == "actorshq":
                self.lbs_deformer = SmplxDeformer(model_path=os.path.join(args.dataset_dir, "body_models"), gender=args.smplx_gender, num_betas=args.smplx_num_betas, use_pca=False)
            elif args.dataset_type == "4ddress":
                self.lbs_deformer = SmplxDeformer(model_path=os.path.join(args.dataset_dir, "body_models"), gender=args.smplx_gender, num_betas=10, use_pca=True)
            self.load_smplx()
            self.collider_faces = torch.from_numpy(self.lbs_deformer.smplx_model.faces.astype(int))
        if args.prescribed_surface_path:
            self.load_prescribed_surface(args.prescribed_surface_path)
        if args.body_checkpoint:
            from body_shape import load_body_shape
            load_body_shape(self, Path(args.body_checkpoint), Path(self.scene.dataset_dir), args.body_batch_size)

        bg_color = [1, 1, 1] if self.scene.white_bkgd else [0, 0, 0]
        self.bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.pipe = pipe

        train_dataloader = torch.utils.data.DataLoader(
            self.scene.train_dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            collate_fn=self.scene.collate_fn,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        self.train_dataloader = cycle(train_dataloader)

        test_dataloader = torch.utils.data.DataLoader(
            self.scene.test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            collate_fn=self.scene.collate_fn,
        )
        self.test_dataloader = accelerator.prepare(test_dataloader)

        self.iterations = opt.iterations
        self.accelerator = accelerator
        # init traiable params
        self.init_D = args.init_D
        self.init_E = args.init_E
        self.init_nu = args.init_nu
        self.init_gamma = args.init_gamma
        self.init_kappa = args.init_kappa

        self.min_D = args.min_D
        self.max_D = args.max_D
        self.min_E = args.min_E
        self.max_E = args.max_E
        self.min_H = args.min_H
        self.max_H = args.max_H

        if args.init_params_path:
            self.best_params = {k: v for k, v in np.load(args.init_params_path).items()}
            self.best_params["loss"] = 1.
            self.best_params["step"] = -1
        else:
            self.best_params = {
                "D": self.init_D,
                "E": self.init_E,
                "H": 1.0,
                "loss": 1.,
                "step": -1,
            }
            if args.random_init_params:
                self.best_params = {
                    "D": np.random.uniform(self.min_D, self.max_D),
                    "E": np.random.uniform(self.min_E, self.max_E)*100,
                    "H": 1.0,
                    "loss": 1.,
                    "step": -1,
                }
        self.last_params = {k: v for k, v in self.best_params.items()}

        self.param_ranges = {
            'D': [self.min_D, self.max_D],
            'E': [self.min_E, self.max_E],
            'H': [self.min_H, self.max_H]
        }
        
        param = {
            'D': self.best_params["D"],
            'E': self.best_params["E"] / 100.0,
            'H': self.best_params["H"],
        }

        self.torch_param = {
            'D': torch.tensor(float(param['D']), requires_grad=False),
            'E': torch.tensor(float(param['E']), requires_grad=False),
            'H': torch.tensor(float(param['H']), requires_grad=False),
        }

        self.mesh_friction_coeff = args.mesh_friction_coeff
        self.friction_angle = args.friction_angle
        
        self.setup_simulation(grid_size=args.grid_size)
        
        self.optimizer = torch.optim.Adam([{"params": [self.torch_param['D']], "lr": opt.lr_D}, {"params": [self.torch_param['E']], "lr": opt.lr_E}, {"params": [self.torch_param['H']], "lr": opt.lr_H}], lr=opt.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(self.iterations*0.5*np.pi/np.arccos(0.4)), eta_min=0.0)
        
        self.optimizer, self.scheduler = accelerator.prepare(
            self.optimizer, self.scheduler
        )

        # setup train info
        self.step = 0
        self.optimizer_reset_step = None
        self.resume_context = {
            "dataset_dir": os.path.abspath(self.scene.dataset_dir),
            "fitting_frame_ids": list(self.scene.train_frame_index),
            "model_path": os.path.abspath(args.model_path),
            "appearance_iteration": self.scene.loaded_timestep,
            "grid_size": args.grid_size, "substep": args.substep,
            "iterations": opt.iterations, "param_ranges": self.param_ranges,
            "nu": self.init_nu, "gamma": self.init_gamma, "kappa": self.init_kappa,
            "mesh_friction_coeff": self.mesh_friction_coeff, "friction_angle": self.friction_angle,
        }
        if args.resume:
            assert not run_eval and not args.init_params_path, "Resume is only for material training"
            assert accelerator.num_processes == 1, "Material resume currently requires one process"
            state = restore_training_state(Path(self.output_path), self.torch_param,
                                           self.optimizer, self.scheduler, self.resume_context,
                                           args.resume_parameters)
            self.step = state["next_step"]
            self.best_params, self.last_params = state["best"], state["last"]
            self.optimizer_reset_step = state["optimizer_reset_step"]
            # Persist the authorized one-time optimizer reset before expensive simulation.
            save_training_state(Path(self.output_path), self.step, self.torch_param,
                                self.optimizer, self.scheduler, self.best_params, self.last_params,
                                self.resume_context, self.optimizer_reset_step)
            export_progress(Path(self.output_path), self.step - 1)

        self.log_iters = opt.log_iters
        self.video_iters = opt.video_iters
        self.wandb_iters = opt.wandb_iters
        self.use_wandb = opt.use_wandb
        self.visualize = opt.visualize and self.use_wandb

        if self.visualize:
            gt_video = []
            for verts in self.train_frame_verts:
                img = render_mesh(verts.detach().cpu().numpy(), self.gaussians.faces.cpu().numpy(), self.scene.test_dataset.camera_list[-1])
                gt_video.append(img)
            self.gt_video = np.stack(gt_video, 0)
            self.best_video = None

        if self.accelerator.is_main_process:
            if opt.use_wandb:
                run = wandb.init(
                    config={**vars(args), **vars(opt), **vars(pipe)},
                    dir=self.output_path,
                    **{
                        "mode": "online",
                        "entity": opt.wandb_entity,
                        "project": opt.wandb_project,
                    },
                )
                wandb.run.log_code(".")
                wandb.run.name = opt.wandb_name
                print(f"run dir: {run.dir}")
                self.wandb_folder = run.dir
                os.makedirs(self.wandb_folder, exist_ok=True)

    def load_gaussians(self, args):
        split_idx = np.load(self.split_idx_path)

        self.num_joint_v = int(split_idx["num_joint_v"])
        self.num_joint_f = int(split_idx["num_joint_f"])

        self.reordered_cloth_v_idx = torch.tensor(split_idx["reordered_cloth_v_idx"]).int().cuda()
        self.reordered_cloth_f_idx = torch.tensor(split_idx["reordered_cloth_f_idx"]).int().cuda()

        self.reordered_human_v_idx = torch.tensor(split_idx["reordered_human_v_idx"]).int().cuda()
        self.reordered_human_f_idx = torch.tensor(split_idx["reordered_human_f_idx"]).int().cuda()

        self.joint_v_idx = self.reordered_cloth_v_idx[:self.num_joint_v]
        self.joint_f_idx = self.reordered_cloth_f_idx[:self.num_joint_f]

        self.new_cloth_faces = torch.tensor(split_idx["new_cloth_faces"]).int().cuda()
        self.new_human_faces = torch.tensor(split_idx["new_human_faces"]).int().cuda()
        
        self.gaussians = MeshGaussianModel(args.sh_degree, device="cuda")
        self.scene = Scene(args, self.gaussians, return_type="video", device="cuda", load_timestep=-1)

        verts = self.gaussians.verts_orig + self.gaussians.verts_offset
        self.train_frame_verts = verts[self.scene.train_frame_start-self.verts_start_idx:self.scene.train_frame_start-self.verts_start_idx+self.scene.train_frame_num]
        self.first_frame_verts = verts[0]

        self.initial_cloth_velocity = estimate_velocity(
            self.train_frame_verts[:, self.reordered_cloth_v_idx.long()],
            self.scene.train_frame_index,
        )

        self.train_frame_verts_velo = (self.train_frame_verts[1:] - self.train_frame_verts[:-1]) * 25.

        self.cloth_opacity = self.gaussians.get_opacity[self.reordered_cloth_f_idx]
        self.human_opacity = self.gaussians.get_opacity[self.reordered_human_f_idx]

        self.cloth_scale = self.gaussians.get_scaling[self.reordered_cloth_f_idx]
        self.human_scale = self.gaussians.get_scaling[self.reordered_human_f_idx]

        self.cloth_shs = self.gaussians.get_features[self.reordered_cloth_f_idx]
        self.human_shs = self.gaussians.get_features[self.reordered_human_f_idx]

        return
    
    def load_raw_collider(self) -> None:
        """Drive collision directly from the exported body-only motion archive."""
        assert self.args.prescribed_surface_path, "Raw colliders require the exported prescribed surface"
        assert self.scene.train_frame_start == self.verts_start_idx == 0
        root = Path(self.scene.dataset_dir)
        manifest = load_manifest(root)
        assert manifest["body_model"] == "raw"
        path = Path(self.args.raw_collider_path)
        assert path.resolve() == (root / manifest["collider_sequence"]).resolve()
        vertices, faces, velocities = load_raw_collider(path, manifest["frame_ids"])
        end = max(self.scene.train_frame_index[-1], self.scene.test_frame_index[-1])
        self.motion_frame_ids = list(range(end + 1))
        self.body_motion = torch.as_tensor(vertices[:end + 1], dtype=torch.float32, device="cuda")
        self.body_motion_velocity = torch.as_tensor(velocities[:end], dtype=torch.float32, device="cuda")
        self.collider_faces = torch.as_tensor(faces, dtype=torch.long)
        self.train_frame_collider = self.body_motion[self.scene.train_frame_index]
        self.train_frame_collider_velo = self.body_motion_velocity[self.scene.train_frame_index[:-1]]
        self.test_frame_collider = self.body_motion[self.scene.test_frame_index]
        self.test_frame_collider_velo = self.body_motion_velocity[self.scene.test_frame_index[:-1]]

    def load_smplx(self) -> None:
        train_frame_smplx = []
        for frame in tqdm(self.scene.train_frame_index, desc="Reading train frame smplx vertices..."):
            if self.scene.dataset_type == "actorshq":
                smplx_path = os.path.join(self.scene.dataset_dir, f"a{self.scene.actor}_s{self.scene.sequence}", "smplx_fitted", f"{frame:06d}.obj")
                if not os.path.exists(smplx_path):
                    smplx_path = os.path.join(self.scene.dataset_dir, f"a{self.scene.actor}_s{self.scene.sequence}", "smplx_fitted", f"{frame:06d}", "smplx_icp.obj")
                smplx_verts, _ = read_obj(smplx_path)
            elif self.scene.dataset_type == "4ddress":
                smplx_verts, _ = read_ply(os.path.join(self.scene.dataset_dir, f"4D-DRESS/{self.scene.subject:05d}_Inner/Inner/Take{self.scene.train_take}", f"SMPLX/mesh-f{frame:05d}_smplx.ply"))
            train_frame_smplx.append(smplx_verts)
        train_frame_smplx = np.stack(train_frame_smplx, 0)
        self.train_frame_collider = torch.tensor(train_frame_smplx).float().cuda()
        self.train_frame_collider_velo = (self.train_frame_collider[1:] - self.train_frame_collider[:-1]) * 25

        if self.args.prescribed_surface_path:
            assert self.scene.train_frame_start == self.verts_start_idx == 0
            end = max(self.scene.train_frame_index[-1], self.scene.test_frame_index[-1])
            self.motion_frame_ids = list(range(end + 1))
            body = []
            for frame in self.motion_frame_ids:
                path = os.path.join(self.scene.dataset_dir, f"a{self.scene.actor}_s{self.scene.sequence}",
                                    "smplx_fitted", f"{frame:06d}.obj")
                assert os.path.isfile(path), path
                vertices, _ = read_obj(path)
                body.append(vertices)
            self.body_motion = torch.as_tensor(np.stack(body), dtype=torch.float32, device="cuda")
            self.body_motion_velocity = (self.body_motion[1:] - self.body_motion[:-1]) * 25.0
            self.test_frame_collider = self.body_motion[self.scene.test_frame_index]
            self.test_frame_collider_velo = (self.test_frame_collider[1:] - self.test_frame_collider[:-1]) * 25
            return

        if self.scene.dataset_type == "actorshq":
            smplx_param_path = os.path.join(self.scene.dataset_dir, f"a{self.scene.actor}_s{self.scene.sequence}", "smplx_fitted", f"{self.verts_start_idx:06d}.pth")
            if not os.path.exists(smplx_param_path):
                smplx_param_path = os.path.join(self.scene.dataset_dir, f"a{self.scene.actor}_s{self.scene.sequence}", "smplx_fitted", f"{self.verts_start_idx:06d}", "smplx_icp_param.pth")
                smplx_param_first = {k: torch.from_numpy(v).cuda() for k, v in torch.load(smplx_param_path).items()}
            else:
                smplx_param_first = {k: v.cuda() for k, v in torch.load(smplx_param_path).items()}
            smplx_out_first = self.lbs_deformer.smplx_forward(smplx_param_first)
        elif self.scene.dataset_type == "4ddress":
            with open(os.path.join(self.scene.dataset_dir, f"4D-DRESS/{self.scene.subject:05d}_Inner/Inner/Take{self.scene.train_take}", f"SMPLX/mesh-f{self.verts_start_idx:05d}_smplx.pkl"), "rb") as smplx_pickle:
                smplx_param_first_numpy = pickle.load(smplx_pickle)
            smplx_param_first = {k: torch.from_numpy(v).cuda()[None] for k, v in smplx_param_first_numpy.items()}
            smplx_out_first = self.lbs_deformer.smplx_forward_simple(smplx_param_first)
            smplx_param_first['trans'] = smplx_param_first['transl']
            smplx_param_first['scale'] = torch.tensor(1).float().cuda()

        if self.scene.dataset_type == "actorshq":
            lbs_w = np.load(os.path.join(self.scene.dataset_dir, f"a{self.scene.actor}_s{self.scene.sequence}", f"{self.args.lbs_w}.npy"))
        elif self.scene.dataset_type == "4ddress":
            lbs_w = np.load(os.path.join(self.scene.dataset_dir, f"s{self.scene.subject}_t{self.scene.train_take}", f"{self.args.lbs_w}.npy"))
        lbs_w = torch.tensor(lbs_w).float().cuda()
        
        test_smplx_params_list = []
        for frame in tqdm(self.scene.test_frame_index, desc="Reading test frame smplx params..."):
            if self.scene.dataset_type == "actorshq":
                smplx_param_path = os.path.join(self.scene.dataset_dir, f"a{self.scene.actor}_s{self.scene.sequence}", "smplx_fitted", f"{frame:06d}.pth")
                if not os.path.exists(smplx_param_path):
                    smplx_param_path = os.path.join(self.scene.dataset_dir, f"a{self.scene.actor}_s{self.scene.sequence}", "smplx_fitted", f"{frame:06d}", "smplx_icp_param.pth")
                    smplx_param = {k: torch.from_numpy(v).cuda() for k, v in torch.load(smplx_param_path).items()}
                else:
                    smplx_param = {k: v.cuda() for k, v in torch.load(smplx_param_path).items()}
            elif self.scene.dataset_type == "4ddress":
                with open(os.path.join(self.scene.dataset_dir, f"4D-DRESS/{self.scene.subject:05d}_Inner/Inner/Take{self.scene.test_take}", f"SMPLX/mesh-f{frame:05d}_smplx.pkl"), "rb") as smplx_pickle:
                    smplx_param_numpy = pickle.load(smplx_pickle)
                smplx_param = {k: torch.from_numpy(v).cuda()[None] for k, v in smplx_param_numpy.items()}
            test_smplx_params_list.append(smplx_param)

        test_smplx_params_dict = {}
        for k, _ in test_smplx_params_list[0].items():
            if k == "scale":
                test_smplx_params_dict[k] = torch.stack([param[k] for param in test_smplx_params_list], 0)
            else:
                test_smplx_params_dict[k] = torch.cat([param[k] for param in test_smplx_params_list], 0)

        if self.scene.dataset_type == "actorshq":
            test_smplx_out = self.lbs_deformer.smplx_forward(test_smplx_params_dict)
        elif self.scene.dataset_type == "4ddress":
            test_smplx_out = self.lbs_deformer.smplx_forward_simple(test_smplx_params_dict)
            test_smplx_params_dict['trans'] = test_smplx_params_dict['transl']
            test_smplx_params_dict['scale'] = torch.tensor(1).float().cuda()

        t_verts_v, _, lbs_w = self.lbs_deformer.transform_to_t_pose(
             self.first_frame_verts.unsqueeze(0),
             smplx_out_first,
             smplx_param_first['trans'],
             smplx_param_first['scale'],
             lbs_w=lbs_w.unsqueeze(0)
        )

        deformed_verts, _ = self.lbs_deformer.transform_to_pose(
            t_verts_v[0].repeat(self.scene.test_frame_num, 1, 1),
            lbs_w.repeat(self.scene.test_frame_num, 1, 1),
            test_smplx_out,
            test_smplx_params_dict['trans'],
            test_smplx_params_dict['scale']
        )

        self.test_frame_collider = test_smplx_out["vertices"].detach()
        self.test_frame_collider_velo = (self.test_frame_collider[1:] - self.test_frame_collider[:-1]) * 25
        self.test_frame_verts = deformed_verts.squeeze().detach()
        self.test_frame_verts_velo = (self.test_frame_verts[1:] - self.test_frame_verts[:-1]) * 25.
        return

    def load_prescribed_surface(self, path: str) -> None:
        """Use the export's domain and prescribed skin/attachment observations."""
        assert torch.count_nonzero(self.gaussians.verts_offset.detach()) == 0
        with np.load(path, allow_pickle=False) as data:
            ids = data["frame_ids"]
            assert np.array_equal(ids, np.arange(len(ids)))
            assert np.allclose(np.diff(data["timestamps"]), 1.0 / 25.0)
            assert np.array_equal(data["faces"], self.gaussians.faces.cpu().numpy())
            assert np.array_equal(data["attachment_vertex_ids"], self.joint_v_idx.cpu().numpy())
            self.export_scale = float(data["simulation_scale"])
            self.export_center = torch.as_tensor(data["simulation_center_world"], device="cuda").clone()
            assert np.isfinite(self.export_scale) and self.export_scale > 0
            assert torch.isfinite(self.export_center).all()
            train_surface = torch.as_tensor(data["vertices"][self.scene.train_frame_index], device="cuda")
            assert torch.allclose(train_surface, self.train_frame_verts.detach(), atol=1e-6, rtol=0)
        self.prescribed_joint_positions, self.prescribed_human_positions = load_driving_surface(
            path, self.motion_frame_ids, self.joint_v_idx, self.reordered_human_v_idx,
        )
        if self.run_eval:
            assert self.scene.test_frame_index[0] == 0 or self.scene.test_frame_index[0] > self.scene.train_frame_index[-1]
            assert self.args.init_params_path, "Future evaluation requires fitted material parameters"
            # A body checkpoint binds the chosen D/E/H independently of the dataset
            # used to fit material. Its own source/window checks run when loaded.
            if not self.args.body_checkpoint:
                with np.load(self.args.init_params_path, allow_pickle=False) as checkpoint:
                    assert np.array_equal(checkpoint["fitting_frame_ids"], self.scene.train_frame_index)
                    assert str(checkpoint["dataset_dir"]) == os.path.abspath(self.scene.dataset_dir)
                    assert np.allclose(checkpoint["initial_cloth_velocity_world_m_s"],
                                       self.initial_cloth_velocity.cpu().numpy(), atol=1e-6, rtol=0)

    def setup_simulation(self, grid_size: int = 100) -> None:

        device = "cuda:{}".format(self.accelerator.process_index)        

        verts_whole = self.train_frame_verts[0].detach()
        verts = verts_whole[self.reordered_cloth_v_idx]
        faces = self.new_cloth_faces

        min_pos = torch.min(verts, 0)[0]
        max_pos = torch.max(verts, 0)[0]
        max_diff = torch.max(max_pos - min_pos)
        original_mean_pos = (min_pos + max_pos) / 2.0
        self.scale = 1.0 / max_diff
        if self.args.prescribed_surface_path:
            self.scale = torch.tensor(self.export_scale, dtype=torch.float32, device="cuda")
            original_mean_pos = self.export_center
        self.shift = torch.tensor([[1.0, 1.0, 1.0]]).float().cuda() - original_mean_pos * self.scale

        self.wld2sim = lambda p: p * self.scale + self.shift
        self.sim2wld = lambda p: (p - self.shift) / self.scale

        n_elements = faces.shape[0]
        n_vertices = verts.shape[0]

        mpm_init_verts = self.wld2sim(verts)
        mpm_init_elts = mpm_init_verts[faces].mean(1)

        mpm_init_traditional = torch.zeros(0, 3).float().cuda()
        mpm_traditional_vol = torch.zeros(0).float().cuda()
        
        n_traditional = mpm_init_traditional.shape[0]
        n_particles = n_traditional + n_elements + n_vertices
        
        mpm_init_pos = torch.cat([mpm_init_elts, mpm_init_traditional, mpm_init_verts], 0)

        mpm_init_dir, mpm_rest_dir, mpm_element_vol, mpm_vertex_vol = self.compute_dir_vol(mpm_init_verts, faces, thickness=1e-5)
        mpm_init_dir_inv = torch.linalg.inv(mpm_init_dir)
        mpm_rest_dir = mpm_rest_dir
        mpm_rest_dir_inv = self.compute_rest_dir_inv(mpm_rest_dir)

        mpm_init_vol = torch.cat([mpm_element_vol, mpm_traditional_vol, mpm_vertex_vol], 0)
        mpm_init_cov = torch.zeros((n_particles - n_vertices, 6))
        
        mpm_init_velo = particle_velocities(self.initial_cloth_velocity, faces, self.scale)

        wp.init()
        # wp.config.mode = "debug"
        # wp.config.verify_cuda = True

        mpm_state = MPMStateStruct()
        mpm_state.init(n_particles, n_elements, n_vertices, device=device, requires_grad=True)

        self.particle_init_position = mpm_init_pos.clone()
        self.particle_init_dir = mpm_init_dir.clone()
        self.particle_init_velo = mpm_init_velo.clone()
        self.particle_init_rest_dir = mpm_rest_dir.clone()

        self.vertices_init_position = self.wld2sim(self.first_frame_verts[self.reordered_cloth_v_idx]).clone()

        self.n_particles = n_particles
        self.n_elements = n_elements
        self.n_vertices = n_vertices
        self.n_traditional = n_traditional
        
        particle_traditional = np.zeros((self.n_particles,), dtype=np.int32)
        particle_traditional[n_elements:n_elements+n_traditional] = 1
        particle_vertices = np.zeros((self.n_particles,), dtype=np.int32)
        particle_vertices[n_elements+n_traditional:] = 1
        particle_elements = np.zeros((self.n_particles,), dtype=np.int32)
        particle_elements[:n_elements] = 1

        mpm_state.from_torch(
            self.particle_init_position.clone(),
            mpm_init_vol.to(device).clone(),
            mpm_init_dir_inv.to(device).clone(),
            mpm_rest_dir_inv.to(device).clone(),
            faces.to(device).clone(),
            particle_traditional,
            particle_vertices,
            particle_elements,
            mpm_init_cov.to(device).clone(),
            device=device,
            requires_grad=True,
            n_grid=grid_size,
            grid_lim=2.0,
        )
        mpm_model = MPMModelStruct()
        mpm_model.init(n_particles, device=device, requires_grad=True)
        mpm_model.init_other_params(n_grid=grid_size, grid_lim=2.0, device=device)

        material_params = {
            "material": "cloth",
            "g": [0.0, -9.8, 0.0],
            "density": 1.0, # kg / m^3
            "grid_v_damping_scale": 1.1,
            "friction_angle": self.friction_angle,
        }

        self.v_damping = material_params["grid_v_damping_scale"]
        self.material_name = material_params["material"]

        mesh_vertices = self.wld2sim(self.test_frame_collider[0]).cpu().numpy()
        mesh_faces = self.collider_faces.cpu().numpy()

        if (self.run_eval and not self.args.prescribed_surface_path and ("lower" in self.split_idx_path or "upper" in self.split_idx_path)):
            human_vertices = self.wld2sim(self.train_frame_verts[0, self.reordered_human_v_idx]).cpu().numpy()
            human_faces = self.new_human_faces.cpu().numpy() + mesh_vertices.shape[0]
            mesh_vertices = np.concatenate([mesh_vertices, human_vertices], 0)
            mesh_faces = np.concatenate([mesh_faces, human_faces], 0)

        mpm_solver = MPMWARP(
            n_particles, n_elements, n_vertices, n_grid=grid_size, grid_lim=2.0, mesh_vertices=mesh_vertices, mesh_faces=mesh_faces, num_joint_t=0, num_joint_v=self.num_joint_v, num_joint_f=self.num_joint_f, device=device
        )
        mpm_solver.set_parameters_dict(mpm_model, mpm_state, material_params)

        self.mpm_state, self.mpm_model, self.mpm_solver = (
            mpm_state,
            mpm_model,
            mpm_solver,
        )

        density = (
            torch.ones_like(self.particle_init_position[..., 0])
            * self.torch_param['D']
        )
        youngs_modulus = (
            torch.ones_like(self.particle_init_position[..., 0])
            * self.torch_param['E'] * 100
        )

        poisson_ratio = torch.ones_like(self.particle_init_position[..., 0]) * self.init_nu
        gamma = torch.ones_like(self.particle_init_position[..., 0]) * self.init_gamma
        kappa = torch.ones_like(self.particle_init_position[..., 0]) * self.init_kappa

        self.density = density
        self.young_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.gamma = gamma
        self.kappa = kappa

        # set density, youngs, poisson
        mpm_state.reset_density(
            density.clone(),
            torch.ones_like(density).type(torch.int),
            device,
            update_mass=True,
        )
        mpm_solver.set_E_nu_from_torch(
            mpm_model, youngs_modulus.clone(), poisson_ratio.clone(), gamma.clone(), kappa.clone(), device
        )
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device)
        mpm_solver.add_mesh_collider(mpm_solver.mesh.id, n_grid=mpm_model.n_grid, friction=self.mesh_friction_coeff)
        mpm_solver.add_particle_mover(n_grid=mpm_model.n_grid)

    def compute_rest_dir_inv(self, rest_dir):
        R11 = rest_dir[:, 0]
        R12 = rest_dir[:, 1]
        R22 = rest_dir[:, 2]
        iR11 = 1.0 / R11
        iR22 = 1.0 / R22
        iR12 = - R12 * iR11 * iR22
        return torch.stack([iR11, iR12, iR22], -1)

    def compute_rest_dir_inv_from_vf(self, vertices, faces):
        # Compute Initial Direction Matrices
        d1 = vertices[faces[:,1]] - vertices[faces[:,0]]
        d2 = vertices[faces[:,2]] - vertices[faces[:,0]]

        # Compute Inverse of Direction Matrices (QR composed)
        R11 = d1.norm(dim=1)
        R12 = (d1*d2).sum(dim=1)/R11
        R22 = (d2 - (R12/R11)[:, None] * d1).norm(dim=1)
        
        iR11 = 1.0 / R11
        iR22 = 1.0 / R22
        iR12 = - R12 * iR11 * iR22
        
        return torch.stack([iR11, iR12, iR22], -1)

    def compute_dir_vol(self, vertices: torch.Tensor, faces: torch.Tensor,
                        thickness: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute Initial Direction Matrices
        d1 = vertices[faces[:,1]] - vertices[faces[:,0]]
        d2 = vertices[faces[:,2]] - vertices[faces[:,0]]
        d3 = d1.cross(d2, dim=-1)
        d3 /= d3.norm(dim=1, keepdim=True)
        init_dir = torch.stack([d1, d2, d3], -1)

        # Compute Inverse of Direction Matrices (QR composed)
        R11 = d1.norm(dim=1)
        R12 = (d1*d2).sum(dim=1)/R11
        R22 = (d2 - (R12/R11)[:, None] * d1).norm(dim=1)
        rest_dir = torch.stack([R11, R12, R22], -1)

        # Compute Particle Volumes
        area = 0.5 * torch.norm(d1.cross(d2, dim=-1), dim=1)
        element_vol = 0.25 * thickness * area
        vertex_vol = torch.zeros(vertices.shape[0], device=vertices.device, dtype=vertices.dtype)
        vertex_vol.index_add_(0, faces.reshape(-1), element_vol[:, None].repeat(1, 3).reshape(-1))

        return init_dir, rest_dir, element_vol, vertex_vol
    
    def get_material_params(self, device):
        youngs_modulus = (
            torch.ones_like(self.particle_init_position[..., 0])
            * self.torch_param["E"] * 100
        )

        density = (
            torch.ones_like(self.particle_init_position[..., 0])
            * self.torch_param['D']
        )
        ret_poisson = self.poisson_ratio.detach().clone()
        ret_gamma = self.gamma.detach().clone()
        ret_kappa = self.kappa.detach().clone()

        return density, youngs_modulus, ret_poisson, ret_gamma, ret_kappa

    def train_one_step(self) -> None:
        accelerator = self.accelerator
        device = "cuda:{}".format(accelerator.process_index)

        log_loss_dict = {}
        wandb_dict = {}

        delta_time = 1.0 / 25
        substep_size = delta_time / self.args.substep
        num_substeps = int(delta_time / substep_size)

        geo_losses = []
        evaluated = {f"evaluated_{key}": value.item() * (100.0 if key == "E" else 1.0)
                     for key, value in self.torch_param.items()}
        for (dD, dE, dH) in [(0., 0., 0.), (0.05, 0., 0.), (0., 0.05, 0.), (0., 0., 0.005)]:
            self.mpm_solver.time = 0.0
            self.mpm_solver.time_profile.clear()
            particle_pos = self.particle_init_position.clone()
            particle_d = self.particle_init_dir.clone()
            particle_velo = self.particle_init_velo.clone()
            particle_R_inv = self.compute_rest_dir_inv_from_vf(torch.stack([self.vertices_init_position[:, 0], self.vertices_init_position[:, 1] * (self.torch_param['H'] + dH), self.vertices_init_position[:, 2]], 1), self.new_cloth_faces)

            self.mpm_state.reset_state(
                self.n_vertices,
                particle_pos.clone(),
                particle_d.clone(),
                None,
                particle_velo.clone(),
                tensor_R_inv=particle_R_inv.clone(),
                device=device,
                requires_grad=True,
            )
            self.mpm_state.set_require_grad(True)

            density, youngs_modulus, poisson, gamma, kappa = self.get_material_params(device)
            
            self.mpm_state.reset_density(
            density + dD, None, device, update_mass=True
            )
            
            self.mpm_solver.set_E_nu_from_torch(
                self.mpm_model, youngs_modulus + dE * 100, poisson.clone(), gamma.clone(), kappa.clone(), device
            )
            self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device)
            
            geo_loss = 0
            if self.visualize and dD == 0 and dE == 0 and dH == 0:
                if self.step % self.video_iters == 0 or self.step == self.iterations - 1:
                    sim_video = [self.gt_video[0]]
            for i in tqdm(range(self.scene.train_frame_num - 1), desc=f"Step {self.step}"):
                mesh_x = self.wld2sim(self.train_frame_collider[i].clone())
                mesh_v = self.train_frame_collider_velo[i].clone() * self.scale
                joint_verts_v = self.train_frame_verts_velo[i, self.joint_v_idx].clone() * self.scale
                joint_faces_v = joint_verts_v[self.new_cloth_faces[:self.num_joint_f]].mean(1).clone()

                for substep_local in range(num_substeps):
                    mesh_x_curr = mesh_x + substep_size * substep_local * mesh_v
                    self.mpm_solver.p2g2p(
                        self.mpm_model, self.mpm_state, substep_size, mesh_x=mesh_x_curr, mesh_v=mesh_v, joint_traditional_v=None, joint_verts_v=joint_verts_v, joint_faces_v=joint_faces_v, device=device
                    )

                particle_pos = wp.to_torch(self.mpm_state.particle_x).clone()

                cloth_verts = self.sim2wld(particle_pos[self.n_elements:])
                geo_loss += F.mse_loss(cloth_verts, self.train_frame_verts[i+1, self.reordered_cloth_v_idx])
                
                if self.visualize and dD == 0 and dE == 0 and dH == 0:
                    if self.step % self.video_iters == 0 or self.step == self.iterations - 1:
                        human_verts = self.train_frame_verts[i+1, self.reordered_human_v_idx]
                        verts = torch.zeros_like(self.first_frame_verts)
                        verts[self.reordered_cloth_v_idx] = cloth_verts
                        verts[self.reordered_human_v_idx] = human_verts
                        img = render_mesh(verts.detach().cpu().numpy(), self.gaussians.faces.cpu().numpy(), self.scene.test_dataset.camera_list[-1])
                        sim_video.append(img)
            
            if self.visualize and dD == 0 and dE == 0 and dH == 0:
                if self.step % self.video_iters == 0 or self.step == self.iterations - 1:
                    sim_video = np.stack(sim_video, 0)
                    if self.best_video is None:
                        self.best_video = sim_video
            
            geo_loss /= (self.scene.train_frame_num - 1)
            
            geo_losses.append(geo_loss.item())

        gradients = {}
        gradients['D'] = (geo_losses[1] - geo_losses[0]) / 0.05
        gradients['E'] = (geo_losses[2] - geo_losses[0]) / 0.05
        gradients['H'] = (geo_losses[3] - geo_losses[0]) / 0.005

        self.optimizer.zero_grad()
        self.torch_param['D'].grad = torch.tensor(gradients['D']).float()
        self.torch_param['E'].grad = torch.tensor(gradients['E']).float()
        self.torch_param['H'].grad = torch.tensor(gradients['H']).float()

        self.optimizer.step()
        self.scheduler.step()

        self.torch_param['D'].clamp_(min=float(self.param_ranges['D'][0]),
                                     max=float(self.param_ranges['D'][-1]))
        self.torch_param['E'].clamp_(min=float(self.param_ranges['E'][0]),
                                     max=float(self.param_ranges['E'][-1]))
        self.torch_param['H'].clamp_(min=float(self.param_ranges['H'][0]),
                                     max=float(self.param_ranges['H'][-1]))
                
        log_loss_dict["loss"] = geo_losses[0]
        print(
            "D:",
            self.torch_param['D'].item(),
            "E:",
            self.torch_param['E'].item() * 100,
            "H:",
            self.torch_param['H'].item(),
            "loss:",
            geo_losses[0],
        )

        if accelerator.is_main_process:
            with torch.no_grad():
                wandb_dict = {
                    "D": self.torch_param['D'].item(),
                    "E": self.torch_param['E'].item() * 100,
                    "H": self.torch_param['H'].item(),
                }
                self.last_params = {
                    "D": wandb_dict["D"],
                    "E": wandb_dict["E"],
                    "H": wandb_dict["H"],
                    "loss": log_loss_dict["loss"],
                    "step": self.step,
                    **evaluated,
                }

                if log_loss_dict["loss"] < self.best_params["loss"]:
                    self.best_params = {k: v for k, v in self.last_params.items()}
                    if self.visualize:
                        self.best_video = sim_video

                wandb_dict.update(log_loss_dict)
                
                if self.visualize:
                    if self.step % self.video_iters == 0 or self.step == self.iterations - 1:
                        video = np.concatenate([sim_video, self.best_video, self.gt_video], 2).transpose(0, 3, 1, 2)
                        wandb_dict["simulation - best - gt"] = wandb.Video(video, fps=25, format='gif')

                if self.use_wandb and self.step % self.wandb_iters == 0:
                    wandb.log(wandb_dict, step=self.step)

        self.accelerator.wait_for_everyone()

    def train(self) -> None:
        if self.step >= self.iterations:
            print(f"Material already completed {self.step} iterations: {self.output_path}", flush=True)
            return
        for index in tqdm(range(self.step, self.iterations), desc="Training progress"):
            self.train_one_step()
            if self.accelerator.is_main_process:
                self.save()
            self.accelerator.wait_for_everyone()
            self.step += 1

    def save(self) -> None:
        initial_state = {
            "initial_velocity_policy": "training_forward_difference_25hz",
            "initial_velocity_frame_ids": np.asarray(self.scene.train_frame_index[:2]),
            "initial_cloth_velocity_world_m_s": self.initial_cloth_velocity.cpu().numpy(),
            "initial_tension_model": "original_rest_height_H",
            "fitting_frame_ids": np.asarray(self.scene.train_frame_index),
            "dataset_dir": os.path.abspath(self.scene.dataset_dir),
        }
        for name, params in (("best", self.best_params), ("last", self.last_params)):
            path = Path(self.output_path) / f"{name}_param_{self.step:05d}.npz"
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez(stream, **{**params, **initial_state})
            temporary.replace(path)
        save_training_state(Path(self.output_path), self.step + 1, self.torch_param,
                            self.optimizer, self.scheduler, self.best_params, self.last_params,
                            self.resume_context, self.optimizer_reset_step)
        export_progress(Path(self.output_path))
        render_material(Path(self.output_path) / "history.csv", Path(self.output_path) / "fit_curves.png")
        return
    
    @torch.no_grad()
    def eval(
        self,
        skip_sim: bool = False,
        skip_render: bool = False,
        skip_video: bool = False,
    ) -> None:
        if self.args.prescribed_surface_path and not skip_render:
            manifest = load_manifest(Path(self.scene.dataset_dir))
            assert manifest.get("body_shape_experiment", {}).get("prediction_body") != "fitted_smplx", (
                "Fitted appearance uses the original SMPL-H body topology and cannot render this "
                "SMPL-X body experiment. Use run.py --render-appearance gt_lighting for exact "
                "fitted-body videos."
            )
        if skip_sim and self.args.prescribed_surface_path:
            score_evaluation(self, load_manifest(Path(self.scene.dataset_dir)))
        if not skip_sim and self.args.prescribed_surface_path:
            all_verts = simulate_future(self)
        elif not skip_sim:
            device = "cuda:{}".format(self.accelerator.process_index)

            density, youngs_modulus, poisson, gamma, kappa = self.get_material_params(device)

            init_verts = self.wld2sim(self.test_frame_verts[0, self.reordered_cloth_v_idx].clone())
            init_elts = init_verts[self.new_cloth_faces].mean(1)
            init_traditionals = torch.zeros(0, 3).float().cuda()
            init_xyzs = torch.cat([init_elts, init_traditionals, init_verts], 0)
            init_verts_velo = self.test_frame_verts_velo[0, self.reordered_cloth_v_idx].clone() * self.scale
            init_elts_velo = init_verts_velo[self.new_cloth_faces].mean(1)
            init_traditionals_velo = torch.zeros(0, 3).float().cuda()
            init_velocity = torch.cat([init_elts_velo, init_traditionals_velo, init_verts_velo], 0)
            particle_d, _, _, _ = self.compute_dir_vol(init_verts, self.new_cloth_faces, thickness=1e-5)
            particle_C = torch.zeros(init_xyzs.shape[0], 3, 3).to(device=device)
            particle_R_inv = self.compute_rest_dir_inv_from_vf(torch.stack([self.vertices_init_position[:, 0], self.vertices_init_position[:, 1] * self.torch_param['H'], self.vertices_init_position[:, 2]], 1), self.new_cloth_faces)

            delta_time = 1.0 / 25
            substep_size = delta_time / self.args.substep
            num_substeps = int(delta_time / substep_size)

            self.mpm_state.reset_density(
                density.clone(), None, device, update_mass=True
            )
            self.mpm_solver.set_E_nu_from_torch(
                self.mpm_model, youngs_modulus.clone(), poisson.clone(), gamma.clone(), kappa.clone(), device
            )
            self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device)

            self.mpm_state.continue_from_torch(
                init_xyzs,
                init_velocity,
                particle_d,
                particle_C,
                tensor_R_inv=particle_R_inv,
                device=device,
                requires_grad=False,
            )
            
            vt_f, ft = [], []
            with open(self.scene.uv_path, "r") as f:
                for line in f:
                    if line[:2] == "vt":
                        vt_f.append(line)
                    elif line[:2] == "f ":
                        parts = line.strip().split()
                        ft.append([int(parts[1].split("/")[1]), int(parts[2].split("/")[1]), int(parts[3].split("/")[1])])
            vt_f += [f"f {v[0]}/{vt[0]} {v[1]}/{vt[1]} {v[2]}/{vt[2]}\n" for v, vt in zip(self.gaussians.faces.cpu().numpy()+1, ft)]
            
            all_verts = []
            mesh_dir = os.path.join(self.output_path, "uvmesh")
            os.makedirs(mesh_dir, exist_ok=True)
            with open(os.path.join(mesh_dir, f"{0:03d}.obj"), "w") as f:
                f.writelines([f"v {v[0]} {v[1]} {v[2]}\n" for v in self.test_frame_verts[0].detach().cpu().numpy()])
                f.writelines(vt_f)
            all_verts.append(self.test_frame_verts[0])

            for i in tqdm(range(self.scene.test_frame_num - 1), desc="Simulation progress"):
                mesh_x = self.wld2sim(self.test_frame_collider[i].clone())
                mesh_v = self.test_frame_collider_velo[i].clone() * self.scale
                if "lower" in self.split_idx_path or "upper" in self.split_idx_path:
                    human_x = self.wld2sim(self.test_frame_verts[i, self.reordered_human_v_idx].clone())
                    mesh_x = torch.cat([mesh_x, human_x], 0)
                    human_v = self.test_frame_verts_velo[i, self.reordered_human_v_idx].clone() * self.scale
                    mesh_v = torch.cat([mesh_v, human_v], 0)
                joint_verts_v = self.test_frame_verts_velo[i, self.joint_v_idx].clone() * self.scale
                joint_faces_v = joint_verts_v[self.new_cloth_faces[:self.num_joint_f]].mean(1).clone()

                for substep_local in range(num_substeps):
                    mesh_x_curr = mesh_x + substep_size * substep_local * mesh_v
                    self.mpm_solver.p2g2p(
                        self.mpm_model, self.mpm_state, substep_size, mesh_x=mesh_x_curr, mesh_v=mesh_v, joint_traditional_v=None, joint_verts_v=joint_verts_v, joint_faces_v=joint_faces_v, device=device
                    )

                pos = wp.to_torch(self.mpm_state.particle_x).clone()
                cloth_verts = self.sim2wld(pos[self.n_elements+self.n_traditional:])
                human_verts = self.test_frame_verts[i+1, self.reordered_human_v_idx]
                
                verts = torch.zeros_like(self.first_frame_verts)
                verts[self.reordered_cloth_v_idx] = cloth_verts
                verts[self.reordered_human_v_idx] = human_verts
                
                with open(os.path.join(mesh_dir, f"{i+1:03d}.obj"), "w") as f:
                    f.writelines([f"v {v[0]} {v[1]} {v[2]}\n" for v in verts.detach().cpu().numpy()])
                    f.writelines(vt_f)
                all_verts.append(verts)
        
        if not skip_render:
            command_bake = ["blender", "-b", "-P", "blender/bake.py", "--", "--output_path", self.output_path]
            subprocess.run(command_bake, check=True)

            if skip_sim:
                all_verts = []
                for frame in tqdm(range(self.scene.test_frame_num), desc="Reading simulated mesh vertices..."):
                    verts, _ = read_obj(os.path.join(self.output_path, "uvmesh", f"{frame:03d}.obj"))
                    all_verts.append(torch.from_numpy(verts).float().cuda())

            ao_maps = []
            for filename in tqdm(sorted(glob(os.path.join(self.output_path, "aomap/*.png"))), desc="Reading test frame AO Map..."):
                ao_map = np.array(Image.open(filename).convert("L")).astype(np.float32) / 255.
                ao_maps.append(ao_map)
            ao_maps = torch.from_numpy(np.array(ao_maps)).unsqueeze(1).contiguous().float().cuda()

            for data in self.test_dataloader:
                cam = data["cam"][0]
                camera_idx = self.scene.test_camera_index[data["camera_idx"][0]]

                gt_rgbs = data["rgb"][0]
                gt_msks = data["msk"][0]

                gt_imgs = gt_rgbs * gt_msks
                if self.scene.white_bkgd:
                    gt_imgs += (1.0 - gt_msks)
                
                savedir = os.path.join(self.output_path, "fitted_appearance", cam.camera_id)
                preddir = os.path.join(savedir, "pred")
                gtdir = os.path.join(savedir, "gt")
                os.makedirs(preddir, exist_ok=True)
                os.makedirs(gtdir, exist_ok=True)

                for i, verts in enumerate(tqdm(all_verts, desc=f"Camera {cam.camera_id} Rendering progress")):
                    self.gaussians.set_mesh_by_verts(verts)
                    shadow_map = self.gaussians.shadow_net(ao_maps[i])["shadow_map"]
                    shadow = F.grid_sample(shadow_map, self.gaussians.uv_coord, mode='bilinear', align_corners=False).squeeze()[..., None][self.gaussians.binding]

                    colors_precomp = shadow * convert_SH(self.gaussians.get_features, cam, self.gaussians, self.gaussians.get_xyz)

                    render_pkg = render(cam, self.gaussians, self.pipe, self.bg, override_color=colors_precomp)
                    rendering = render_pkg["render"] * torch.exp(self.gaussians.cam_m[camera_idx])[:, None, None] + self.gaussians.cam_c[camera_idx][:, None, None]
                    rendering = rendering * render_pkg["mask"]
                    if self.scene.white_bkgd:
                        rendering += (1.0 - render_pkg["mask"])

                    img_pred = (torch.clamp(rendering.permute(1, 2, 0), 0., 1.).detach().cpu().numpy() * 255.).astype(np.uint8)
                    img_pred_path = os.path.join(preddir, f"{self.scene.test_frame_index[i]:04d}.png")
                    Image.fromarray(img_pred).save(img_pred_path)

                    img_gt = (torch.clamp(gt_imgs[i].permute(1, 2, 0), 0., 1.).detach().cpu().numpy() * 255.).astype(np.uint8)
                    img_gt_path = os.path.join(gtdir, f"{self.scene.test_frame_index[i]:04d}.png")
                    Image.fromarray(img_gt).save(img_gt_path)

                if not skip_video:
                    encode_comparison(Path(savedir), self.scene.test_frame_index)
        
        return

def parse_args():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--run_eval", action="store_true", default=False)
    parser.add_argument("--smplx_num_betas", type=int, default=300)
    parser.add_argument("--prescribed_surface_path", type=str, default="")
    parser.add_argument("--raw_collider_path", type=str, default="")
    parser.add_argument("--body_shape", action="store_true")
    parser.add_argument("--body_checkpoint", type=str, default="")
    parser.add_argument("--body_iterations", type=int, default=100)
    parser.add_argument("--beta_lr", type=float, default=0.01)
    parser.add_argument("--beta_epsilon", type=float, default=0.01)
    parser.add_argument("--body_batch_size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_parameters", action="store_true")
    parser.add_argument("--skip_sim", action="store_true", default=False)
    parser.add_argument("--skip_render", action="store_true", default=False)
    parser.add_argument("--skip_video", action="store_true", default=False)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    args = parser.parse_args(sys.argv[1:])

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    model_args = lp.extract(args)
    model_args.smplx_num_betas = args.smplx_num_betas
    model_args.prescribed_surface_path = args.prescribed_surface_path
    model_args.raw_collider_path = args.raw_collider_path
    model_args.body_shape = args.body_shape
    model_args.body_checkpoint = args.body_checkpoint
    model_args.body_iterations = args.body_iterations
    model_args.beta_lr = args.beta_lr
    model_args.beta_epsilon = args.beta_epsilon
    model_args.body_batch_size = args.body_batch_size
    assert not args.body_shape or (not args.run_eval and not args.resume and not args.body_checkpoint)
    assert not args.body_checkpoint or args.run_eval
    model_args.resume = args.resume
    model_args.resume_parameters = args.resume_parameters
    return model_args, op.extract(args), pp.extract(args), args.run_eval, args.skip_sim, args.skip_render, args.skip_video

if __name__ == "__main__":
    args, opt, pipe, run_eval, skip_sim, skip_render, skip_video = parse_args()
    trainer = Trainer(args, opt, pipe, run_eval)

    if args.body_shape:
        from body_shape import fit_body_shape
        fit_body_shape(trainer, Path(args.dataset_dir), Path(args.init_params_path),
                       iterations=args.body_iterations, learning_rate=args.beta_lr,
                       finite_difference=args.beta_epsilon, batch_size=args.body_batch_size,
                       resume=(Path(trainer.output_path) / "body_shape_state.pt").is_file())
    elif run_eval:
        evaluation = Path(trainer.output_path)
        material = Path(args.init_params_path)
        with np.load(material, allow_pickle=False) as selected:
            material_step = int(selected["step"])
        render_material(material.parent / "history.csv", evaluation / "material_fit_curves.png", material_step)
        if args.body_checkpoint:
            body = Path(args.body_checkpoint)
            with np.load(body, allow_pickle=False) as selected:
                body_step = int(selected["step"])
                beta_index = int(selected["beta_index"])
            render_body(body.parent / "beta_history.csv", evaluation / "body_fit_curves.png",
                        beta_index, body_step)
        trainer.eval(skip_sim, skip_render, skip_video)
    else:
        trainer.train()
