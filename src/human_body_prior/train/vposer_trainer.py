# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2020.12.12

# from pytorch_lightning import Trainer

import glob
import os
import os.path as osp
from datetime import datetime as dt

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from torch.utils.data import DataLoader

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.data.dataloader import VPoserDS
from human_body_prior.data.prepare_data import dataset_exists, prepare_vposer_datasets
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.angle_continuous_repres import geodesic_loss_R
from human_body_prior.tools.configurations import dump_config, load_config
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import (
    get_support_data_dir,
    log2file,
    make_deterministic,
    makepath,
)
from human_body_prior.tools.rotation_tools import aa2matrot
from human_body_prior.visualizations.training_visualization import (
    vposer_trainer_renderer,
)

DEBUG = {}


class VPoserTrainer(LightningModule):
    """

    It includes all data loading and train / val logic., and it is used for both training and testing models.
    """

    def __init__(self, _config):
        super(VPoserTrainer, self).__init__()

        vp_ps = load_config(**_config)

        make_deterministic(vp_ps.general.rnd_seed)

        self.expr_id = vp_ps.general.expr_id
        self.dataset_id = vp_ps.general.dataset_id

        self.work_dir = vp_ps.logging.work_dir = makepath(
            vp_ps.general.work_basedir, self.expr_id
        )
        self.dataset_dir = vp_ps.logging.dataset_dir = osp.join(
            vp_ps.general.dataset_basedir, vp_ps.general.dataset_id
        )

        self._log_prefix = "[{}]".format(self.expr_id)
        self.text_logger = log2file(prefix=self._log_prefix)

        self.seq_len = vp_ps.data_parms.num_timeseq_frames

        self.vp_model = VPoser(vp_ps)

        with torch.no_grad():
            self.bm_train = BodyModel(vp_ps.body_model.bm_fname)

        if vp_ps.logging.render_during_training:
            self.renderer = vposer_trainer_renderer(
                self.bm_train, vp_ps.logging.num_bodies_to_display
            )
        else:
            self.renderer = None

        self.example_input_array = {
            "pose_body": torch.ones(vp_ps.train_parms.batch_size, 63),
        }
        self.vp_ps = vp_ps

    def forward(self, pose_body):
        return self.vp_model(pose_body)

    def _get_data(self, split_name):
        assert split_name in ("train", "vald", "test")

        split_name = split_name.replace("vald", "vald")

        assert dataset_exists(self.dataset_dir), FileNotFoundError(
            "Dataset does not exist dataset_dir = {}".format(self.dataset_dir)
        )
        dataset = VPoserDS(
            osp.join(self.dataset_dir, split_name), data_fields=["pose_body"]
        )

        assert len(dataset) != 0, ValueError("Dataset has nothing in it!")

        return DataLoader(
            dataset,
            batch_size=self.vp_ps.train_parms.batch_size,
            shuffle=True if split_name == "train" else False,
            num_workers=self.vp_ps.data_parms.num_workers,
            pin_memory=True,
        )

    @rank_zero_only
    def on_train_start(self):
        if self.global_rank != 0:
            return
        self.train_starttime = dt.now().replace(microsecond=0)

        ######## make a backup of vposer
        git_repo_dir = os.path.abspath(__file__).split("/")
        git_repo_dir = "/".join(
            git_repo_dir[: git_repo_dir.index("human_body_prior") + 1]
        )
        starttime = dt.strftime(self.train_starttime, "%Y_%m_%d_%H_%M_%S")
        archive_path = makepath(
            self.work_dir, "code", "vposer_{}.tar.gz".format(starttime), isfile=True
        )
        cmd = "cd %s && git ls-files -z | xargs -0 tar -czf %s" % (
            git_repo_dir,
            archive_path,
        )
        os.system(cmd)
        ########
        self.text_logger("Created a git archive backup at {}".format(archive_path))
        dump_config(self.vp_ps, osp.join(self.work_dir, "{}.yaml".format(self.expr_id)))

    def train_dataloader(self):
        return self._get_data("train")

    def val_dataloader(self):
        return self._get_data("vald")

    def configure_optimizers(self):
        def params_count(params):
            return sum(p.numel() for p in params if p.requires_grad)

        gen_params = [
            a[1] for a in self.vp_model.named_parameters() if a[1].requires_grad
        ]
        gen_optimizer_class = getattr(
            optim_module, self.vp_ps.train_parms.gen_optimizer.type
        )
        gen_optimizer = gen_optimizer_class(
            gen_params, **self.vp_ps.train_parms.gen_optimizer.args
        )

        self.text_logger(
            "Total Trainable Parameters Count in vp_model is %2.2f M."
            % (params_count(gen_params) * 1e-6)
        )

        lr_sched_class = getattr(
            lr_sched_module, self.vp_ps.train_parms.lr_scheduler.type
        )

        gen_lr_scheduler = lr_sched_class(
            gen_optimizer, **self.vp_ps.train_parms.lr_scheduler.args
        )

        schedulers = [
            {
                "scheduler": gen_lr_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        ]
        return [gen_optimizer], schedulers

    def _compute_loss(self, dorig, drec):
        l1_loss = torch.nn.L1Loss(reduction="mean")
        geodesic_loss = geodesic_loss_R(reduction="mean")

        bs, latentD = drec["poZ_body_mean"].shape
        device = drec["poZ_body_mean"].device

        loss_kl_wt = self.vp_ps.train_parms.loss_weights.loss_kl_wt
        loss_rec_wt = self.vp_ps.train_parms.loss_weights.loss_rec_wt
        loss_matrot_wt = self.vp_ps.train_parms.loss_weights.loss_matrot_wt
        loss_jtr_wt = self.vp_ps.train_parms.loss_weights.loss_jtr_wt

        # q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])
        q_z = drec["q_z"]
        # dorig['fullpose'] = torch.cat([dorig['root_orient'], dorig['pose_body']], dim=-1)

        # ======================================================
        # Reconstruction loss - L1 on the output mesh
        #   note: paper suggests using L2 loss, but it appears they implement with L1 instead
        #         for training. they output L2 reconstruction loss in the validation metrics though
        # ======================================================
        with torch.no_grad():
            bm_orig = self.bm_train(pose_body=dorig["pose_body"])
        bm_rec = self.bm_train(pose_body=drec["pose_body"].contiguous().view(bs, -1))
        v2v = l1_loss(bm_rec.v, bm_orig.v)

        # ======================================================
        # KL Divergence loss - kl divergence with N(0,1)
        # ======================================================
        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros((bs, latentD), device=device, requires_grad=False),
            scale=torch.ones((bs, latentD), device=device, requires_grad=False),
        )
        kl_loss = torch.mean(
            torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1])
        )

        # ======================================================
        # the weighted loss dict, they use for training
        # ======================================================

        weighted_loss_dict = {
            "loss_kl": loss_kl_wt * kl_loss,
            "loss_mesh_rec": loss_rec_wt * v2v,
        }

        # ======================================================
        # they only add these matrix checks up to some epochs, likely for stability
        # ======================================================
        if (
            self.current_epoch
            < self.vp_ps.train_parms.keep_extra_loss_terms_until_epoch
        ):
            # ======================================================
            # Rotation matrix loss - Geodesic loss between the rotations.
            #   note: paper suggests using L2 loss, but it appears they implement with geodesic loss instead
            # ======================================================
            weighted_loss_dict["matrot"] = loss_matrot_wt * geodesic_loss(
                drec["pose_body_matrot"].view(-1, 3, 3),
                aa2matrot(dorig["pose_body"].view(-1, 3)),
            )

            # ======================================================
            # Loss on the joint positions - L1 reconstruction loss on the joints
            # ======================================================
            weighted_loss_dict["jtr"] = loss_jtr_wt * l1_loss(bm_rec.Jtr, bm_orig.Jtr)
            pass

        weighted_loss_dict["loss_total"] = torch.stack(
            list(weighted_loss_dict.values())
        ).sum()

        with torch.no_grad():
            unweighted_loss_dict = {
                "v2v": torch.sqrt(torch.pow(bm_rec.v - bm_orig.v, 2).sum(-1)).mean()
            }
            unweighted_loss_dict["loss_total"] = (
                torch.cat(
                    list(
                        {
                            k: v.view(-1) for k, v in unweighted_loss_dict.items()
                        }.values()
                    ),
                    dim=-1,
                )
                .sum()
                .view(1)
            )

        return {
            "weighted_loss": weighted_loss_dict,
            "unweighted_loss": unweighted_loss_dict,
        }

    # @torch.no_grad()
    # def _check_nans(self, batch):
    #     if torch.isnan(batch["pose_body"]).any():
    #         raise ValueError("NaN detected in input batch['pose_body']")
    #     if torch.isinf(batch["pose_body"]).any():
    #         raise ValueError("Inf detected in input batch['pose_body']")
    #     for name, param in self.vp_model.named_parameters():
    #         if torch.isnan(param).any():
    #             raise ValueError(f"NaN detected in model parameter {name}")
    #         if torch.isinf(param).any():
    #             raise ValueError(f"Inf detected in model parameter {name}")
    #         if param.grad is not None:
    #             if torch.isnan(param.grad).any():
    #                 raise ValueError(
    #                     f"NaN detected in gradient of model parameter {name}"
    #                 )
    #             if torch.isinf(param.grad).any():
    #                 raise ValueError(
    #                     f"Inf detected in gradient of model parameter {name}"
    #                 )

    # @torch.no_grad()
    # def _find_max_abs_grad(self):
    #     max_abs_grad = 0.0
    #     max_param_name = ""
    #     for name, param in self.vp_model.named_parameters():
    #         if param.grad is not None:
    #             param_max = param.grad.abs().max().item()
    #             print(
    #                 f"Param: {name:<30} Max abs grad: {param_max:<15.6f} Max abs value: {param.data.abs().max().item():<15.6f}"
    #             )
    #             if param_max > max_abs_grad:
    #                 max_abs_grad = param_max
    #                 max_param_name = name
    #     return max_abs_grad, max_param_name

    # @torch.no_grad()
    # def _zero_grad(self):
    #     for param in self.vp_model.parameters():
    #         if param.grad is not None:
    #             param.grad.zero_()

    # @torch.no_grad()
    # def _prettyprint_loss(self, loss):
    #     def _convert_subloss(subloss_dict):
    #         return "  ".join([f"{k}={v.item():.4f}" for k, v in subloss_dict.items()])

    #     weighted_str = _convert_subloss(loss["weighted_loss"])
    #     unweighted_str = _convert_subloss(loss["unweighted_loss"])
    #     return f"weighted: {{{weighted_str}}}, unweighted: {{{unweighted_str}}}"

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # for debugging
        # self._check_nans(batch)

        drec = self(batch["pose_body"].view(-1, 63))

        loss = self._compute_loss(batch, drec)

        train_loss = loss["weighted_loss"]["loss_total"]

        # for debugging
        # if train_loss.isnan() or train_loss.isinf():
        #     raise ValueError("NaN or Inf detected in train_loss")
        # print("[LOSS]", self._prettyprint_loss(loss))
        # train_loss.backward(retain_graph=True)
        # max_grad, max_param = self._find_max_abs_grad()
        # print(f"Max abs grad: {max_grad:.6f} (param: {max_param})")
        # self._check_nans(batch)
        # self._zero_grad()

        tensorboard_logs = {"train_loss": train_loss}
        progress_bar = {k: c2c(v) for k, v in loss["weighted_loss"].items()}
        return {
            "loss": train_loss,
            "progress_bar": progress_bar,
            "log": tensorboard_logs,
        }

    def validation_step(self, batch, batch_idx):
        drec = self(batch["pose_body"].view(-1, 63))

        loss = self._compute_loss(batch, drec)
        val_loss = loss["unweighted_loss"]["loss_total"]

        if (
            self.renderer is not None
            and self.global_rank == 0
            and batch_idx % 500 == 0
            and np.random.rand() > 0.5
        ):
            out_fname = makepath(
                self.work_dir,
                "renders/vald_rec_E{:03d}_It{:04d}_val_loss_{:.2f}.png".format(
                    self.current_epoch, batch_idx, val_loss.item()
                ),
                isfile=True,
            )
            self.renderer([batch, drec], out_fname=out_fname)
            dgen = self.vp_model.sample_poses(self.vp_ps.logging.num_bodies_to_display)
            out_fname = makepath(
                self.work_dir,
                "renders/vald_gen_E{:03d}_I{:04d}.png".format(
                    self.current_epoch, batch_idx
                ),
                isfile=True,
            )
            self.renderer([dgen], out_fname=out_fname)

        # Log metrics directly in validation_step for PL 2.x compatibility
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        self.log(
            "v2v", val_loss, prog_bar=True, logger=True
        )  # Assuming 'v2v' is the key metric

        return {"val_loss": c2c(val_loss)}

    # Remove validation_epoch_end entirely, as it's deprecated in PL 2.x.
    # Metrics are now logged per step and aggregated automatically.

    @rank_zero_only
    def on_train_end(self):
        self.train_endtime = dt.now().replace(microsecond=0)
        endtime = dt.strftime(self.train_endtime, "%Y_%m_%d_%H_%M_%S")
        elapsedtime = self.train_endtime - self.train_starttime
        self.vp_ps.logging.best_model_fname = (
            self.trainer.checkpoint_callback.best_model_path
        )

        self.text_logger(
            "Epoch {} - Finished training at {} after {}".format(
                self.current_epoch, endtime, elapsedtime
            )
        )
        self.text_logger(
            "best_model_fname: {}".format(self.vp_ps.logging.best_model_fname)
        )

        dump_config(
            self.vp_ps,
            osp.join(self.work_dir, "{}_{}.yaml".format(self.expr_id, self.dataset_id)),
        )
        self.hparams = self.vp_ps.toDict()

    @rank_zero_only
    def prepare_data(self):
        """' Similar to standard AMASS dataset preparation pipeline:
        Donwload npz file, corresponding to body data from https://amass.is.tue.mpg.de/ and place them under amass_dir
        """
        self.text_logger = log2file(
            makepath(self.work_dir, "{}.log".format(self.expr_id), isfile=True),
            prefix=self._log_prefix,
        )

        prepare_vposer_datasets(
            self.dataset_dir,
            self.vp_ps.data_parms.amass_splits,
            self.vp_ps.data_parms.amass_dir,
            logger=self.text_logger,
        )


def create_expr_message(ps):
    expr_msg = "[{}] batch_size = {}.".format(
        ps.general.expr_id, ps.train_parms.batch_size
    )

    return expr_msg


def train_vposer_once(_config):
    resume_training_if_possible = True

    model = VPoserTrainer(_config)
    model.vp_ps.logging.expr_msg = create_expr_message(model.vp_ps)
    # model.text_logger(model.vp_ps.logging.expr_msg.replace(". ", '.\n'))
    dump_config(model.vp_ps, osp.join(model.work_dir, "{}.yaml".format(model.expr_id)))

    logger = TensorBoardLogger(model.work_dir, name="tensorboard")
    lr_monitor = LearningRateMonitor()

    snapshots_dir = osp.join(model.work_dir, "snapshots")
    checkpoint_callback = ModelCheckpoint(
        dirpath=makepath(snapshots_dir, isfile=True),
        filename="%s_{epoch:02d}_{val_loss:.2f}" % model.expr_id,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(**model.vp_ps.train_parms.early_stopping)

    resume_from_checkpoint = None
    if resume_training_if_possible:
        available_ckpts = sorted(
            glob.glob(osp.join(snapshots_dir, "*.ckpt")), key=os.path.getmtime
        )
        if len(available_ckpts) > 0:
            resume_from_checkpoint = available_ckpts[-1]
            model.text_logger(
                "Resuming the training from {}".format(resume_from_checkpoint)
            )

    trainer = pl.Trainer(
        # initial repo had deprecated arguments
        # - deprecated distributed_backend
        # - deprecated weights_summary
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[lr_monitor, early_stop_callback, checkpoint_callback],
        max_epochs=model.vp_ps.train_parms.num_epochs,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # gradient_clip_val=1.0,
    )

    # we pass ckpt_path to fit() instead
    trainer.fit(model, ckpt_path=resume_from_checkpoint)
