# =====================================================================================
# ===================================== IMPORTS =======================================
# =====================================================================================

import argparse
import os
from os.path import abspath, dirname, exists, join
from typing import List, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, field_validator, model_validator
from torch import nn
import time
import code
import pandas as pd

import pytorch3d.structures as pd3d_structures
from pytorch3d.transforms import quaternion_apply

print("Original PYOPENGL_PLATFORM:", os.environ.get("PYOPENGL_PLATFORM"))
print("Forcing PYOPENGL_PLATFORM to 'glx'")
os.environ["PYOPENGL_PLATFORM"] = "glx"

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import flatten_list, log2file

can_display = True

try:
    from body_visualizer.mesh.psbody_mesh_cube import points_to_cubes
    from body_visualizer.mesh.psbody_mesh_sphere import points_to_spheres
    from body_visualizer.tools.mesh_tools import rotateXYZ
    from body_visualizer.tools.vis_tools import colors
    from psbody.mesh import Mesh, MeshViewers
    from psbody.mesh.lines import Lines

except Exception as e:
    print(e)
    print("psbody.mesh based visualization could not be started. skipping ...")
    can_display = False

# =====================================================================================
# ================================== GENERAL HELPERS===================================
# =====================================================================================


def vprint(threshold, verbosity, *args, **kwargs):
    if verbosity >= threshold:
        print(*args, **kwargs)


def index_tensor_dict(td: dict, index: Union[int, List[int], torch.Tensor]):
    """Index a dictionary of tensors."""
    return {k: v[index] for k, v in td.items()}


# =====================================================================================
# ===================================== SETUP =========================================
# =====================================================================================

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {TORCH_DEVICE}")

# =====================================================================================
# ================================== BODY MODEL =======================================
# =====================================================================================


def get_body_model(
    body_model_path,
    num_betas=10,
):
    body_model = BodyModel(body_model_path, num_betas=num_betas)
    return body_model


def compute_vertex_normal_batched(v, f):
    return (
        pd3d_structures.Meshes(verts=v, faces=f.expand(len(v), -1, -1))
        .verts_normals_packed()
        .view(-1, v.shape[1], 3)
    )


# =====================================================================================
# ================================== VISUALIZER =======================================
# =====================================================================================


def _mesh_to_vertex_normal_lines(
    vertices, faces=None, normals=None, length=0.02, color=None
):
    if color is None:
        color = colors["blue"]

    tbv = torch.tensor(vertices).to(TORCH_DEVICE)

    if normals is None:
        if faces is None:
            raise ValueError("Either faces or normals must be provided")
        tbf = torch.tensor(faces).to(TORCH_DEVICE)
        body_vn = compute_vertex_normal_batched(tbv, tbf)
    else:
        body_vn = torch.tensor(normals).to(TORCH_DEVICE)

    lines_v = torch.cat([tbv, tbv + length * body_vn], dim=1)
    lines_e = torch.tensor([[i, i + tbv.shape[1]] for i in range(tbv.shape[1])]).to(
        torch.int32
    )
    lines = Lines(v=c2c(lines_v[0]), e=c2c(lines_e))
    lines.vc = (lines.v * 0.0 + 1) * color
    return lines


class Visualizer:
    def __init__(self, rows=1, cols=1, keepalive=True):
        if not can_display:
            print("Visualization not available")
            return

        mvs = MeshViewers((rows, cols), keepalive=keepalive)
        self.mvs = flatten_list(mvs)
        self.mvs[0].set_background_color(colors["white"])

        self.render_packs = [{} for _ in range(len(self.mvs))]
        self.persisted_render_packs = [{} for _ in range(len(self.mvs))]

        self.set_titlebar()

    # =============================== SANITIZATION ====================================

    def _check_initialized(self):
        if not can_display:
            raise ValueError("Visualization not available")
        if self.mvs is None:
            raise ValueError("Visualizer not initialized properly")

    def _check_view_id(self, view_id):
        if view_id < 0 or view_id >= len(self.mvs):
            raise ValueError(
                f"view_id {view_id} is out of range. Should be between 0 and {len(self.mvs) - 1}"
            )

    class RenderPointList(BaseModel):
        model_config = {"arbitrary_types_allowed": True}

        position: torch.Tensor
        radius: torch.Tensor
        color: torch.Tensor

        @field_validator("position")
        @classmethod
        def validate_position(cls, v):
            if len(v.shape) == 1 and v.shape[0] == 3:
                return v
            elif len(v.shape) == 2 and v.shape[1] == 3:
                return v
            raise ValueError("Position must be of shape (3,) or (N, 3)")

        @field_validator("radius")
        @classmethod
        def validate_radius(cls, v):
            if len(v.shape) == 0:
                return v
            elif len(v.shape) == 1:
                return v
            raise ValueError("Radius must be a scalar or of shape (N,)")

        @field_validator("color")
        @classmethod
        def validate_color(cls, v):
            if len(v.shape) == 1 and v.shape[0] == 3:
                return v
            elif len(v.shape) == 2 and v.shape[1] == 3:
                return v
            raise ValueError("Color must be of shape (3,) or (N, 3)")

        @model_validator(mode="after")
        def validate_lengths(cls, values):
            pos, rad, col = (
                values.get("position"),
                values.get("radius"),
                values.get("color"),
            )
            n_points = pos.shape[0] if len(pos.shape) == 2 else 1

            if len(rad.shape) == 1 and rad.shape[0] != n_points:
                raise ValueError("Radius length must match number of positions")
            if len(col.shape) == 2 and col.shape[0] != n_points:
                raise ValueError("Color length must match number of positions")
            return values

    def _convert_to_render_points(self, points_render_list):
        try:
            return self.RenderPointList(**points_render_list)
        except Exception as e:
            raise ValueError(f"Invalid RenderPointList: {e}")

    def _adapt_body_model_instance(self, body_model_instance):
        # ======== extract vertices and faces ========

        if isinstance(body_model_instance, dict):
            if "v" in body_model_instance and "f" in body_model_instance:
                vertices = body_model_instance["v"]
                faces = body_model_instance["f"]
            else:
                raise ValueError(
                    "body_model_instance dict must contain 'v' and 'f' keys"
                )
        elif hasattr(body_model_instance, "v") and hasattr(body_model_instance, "f"):
            vertices = body_model_instance.v
            faces = body_model_instance.f
        else:
            raise ValueError(
                "body_model_instance must be a dict with 'v' and 'f' or have v and f attributes"
            )

        if isinstance(vertices, torch.Tensor):
            vertices = c2c(vertices)
        elif isinstance(vertices, np.ndarray):
            vertices = torch.tensor(vertices)
        else:
            raise ValueError("vertices must be a torch.Tensor or np.ndarray")

        if isinstance(faces, torch.Tensor):
            faces = c2c(faces).astype(np.int32)
        elif isinstance(faces, np.ndarray):
            faces = torch.tensor(faces).astype(np.int32)
        else:
            raise ValueError("faces must be a torch.Tensor or np.ndarray")

        # if we are given a 1 batched body model, remove the batch dimension
        # otherwise raise an error
        if len(vertices.shape) == 3 and vertices.shape[0] == 1:
            vertices = vertices[0]
        elif len(vertices.shape) == 2:
            pass
        else:
            raise ValueError(
                "vertices must be of shape (N, 3) or (1, N, 3). Got: ", vertices.shape
            )
        if vertices.shape[1] != 3:
            raise ValueError("vertices must have 3 coordinates per vertex")

        return {
            "v": vertices,
            "f": faces,
        }

    # =============================== RENDERING =======================================

    def _render(self, i):
        mv = self.mvs[i]
        persisted_render_pack = self.persisted_render_packs[i]
        render_pack = self.render_packs[i]

        dynamic_lines = (
            []
            + persisted_render_pack.get("dynamic_lines", [])
            + render_pack.get("dynamic_lines", [])
        )
        dynamic_meshes = (
            []
            + persisted_render_pack.get("dynamic_meshes", [])
            + render_pack.get("dynamic_meshes", [])
        )
        static_meshes = (
            []
            + persisted_render_pack.get("static_meshes", [])
            + render_pack.get("static_meshes", [])
        )

        # ==== render body ====
        if "body" in render_pack and render_pack["body"] is not None:
            body = render_pack["body"]
            body_verts = c2c(body["v"])
            body_faces = c2c(body["f"])
            # body_mesh = Mesh(v=body_verts, f=body_faces, vc=colors["grey"])
            body_mesh = Mesh(v=body_verts, f=[], vc=colors["grey"])
            static_meshes.append(body_mesh)

            if (
                "show_vertex_normals" in render_pack
                and render_pack["show_vertex_normals"]
            ):
                lines = _mesh_to_vertex_normal_lines(
                    [body["v"]],
                    faces=body["f"],
                    length=0.01,
                )
                dynamic_lines.append(lines)

        # ==== render spheres ===
        if "spheres" in render_pack and render_pack["spheres"] is not None:
            spheres = render_pack["spheres"]
            sphere_mesh = points_to_spheres(
                c2c(spheres.position),
                radius=c2c(spheres.radius),
                point_color=c2c(spheres.color),
            )
            dynamic_meshes.append(sphere_mesh)

        # ==== render cubes ====
        if "cubes" in render_pack and render_pack["cubes"] is not None:
            cubes = render_pack["cubes"]
            cube_mesh = points_to_cubes(
                c2c(cubes.position),
                radius=c2c(cubes.radius),
                point_color=c2c(cubes.color),
            )
            dynamic_meshes.append(cube_mesh)

        mv.set_dynamic_lines(dynamic_lines)
        mv.set_dynamic_meshes(dynamic_meshes)
        mv.set_static_meshes(static_meshes)

    # =============================== USER INTERFACE ==================================

    def display_mesh(
        self,
        view_id=0,
        body=None,
        additional_spheres=None,
        additional_cubes=None,
        show_vertex_normals=False,
        re_render=True,
    ):
        # check args
        self._check_initialized()
        self._check_view_id(view_id)

        # setup render pack
        additional_spheres = (
            self._convert_to_render_points(additional_spheres)
            if additional_spheres
            else None
        )
        additional_cubes = (
            self._convert_to_render_points(additional_cubes)
            if additional_cubes
            else None
        )
        body = self._adapt_body_model_instance(body) if body is not None else None
        render_pack = {
            "body": body,
            "spheres": additional_spheres,
            "cubes": additional_cubes,
            "show_vertex_normals": show_vertex_normals,
        }
        self.render_packs[view_id] = render_pack

        # render
        if re_render:
            self._render(view_id)

    def also_render(
        self,
        view_id=0,
        dynamic_lines=None,
        dynamic_meshes=None,
        static_meshes=None,
        persist=False,
        re_render=True,
    ):
        self._check_initialized()
        self._check_view_id(view_id)

        rp = (
            self.render_packs[view_id]
            if not persist
            else self.persisted_render_packs[view_id]
        )

        rp.setdefault("dynamic_lines", []).extend(dynamic_lines or [])
        rp.setdefault("dynamic_meshes", []).extend(dynamic_meshes or [])
        rp.setdefault("static_meshes", []).extend(static_meshes or [])

        # re-render
        if re_render:
            self._render(view_id)

    def clear_view(
        self,
        view_id=0,
        persist=False,
        re_render=True,
    ):
        self._check_initialized()
        self._check_view_id(view_id)

        self.render_packs[view_id] = {}

        if persist:
            self.persisted_render_packs[view_id] = {}

        # re-render
        if re_render:
            self._render(view_id)

    def set_titlebar(self, titlebar="Visualizer"):
        self._check_initialized()
        self.mvs[0].set_titlebar(titlebar)


# =====================================================================================
# =============================== DATA PROCESSING =====================================
# =====================================================================================


def extract_orients_from_df(
    df: pd.DataFrame,
    suffixes: List[str],
) -> Tuple[List[str], np.ndarray]:
    # find the candidate markers
    candidate_markers = set()
    for col in df.columns:
        for suffix in suffixes:
            if col.endswith(suffix):
                marker_name = col[: -len(suffix)]
                candidate_markers.add(marker_name)

    # filter markers that have all suffixes
    valid_markers = []
    for marker in candidate_markers:
        if all((marker + suffix) in df.columns for suffix in suffixes):
            valid_markers.append(marker)

    # create a dict of tensors of shape (N, 4) for quaternions
    marker_orients = {}
    for marker in valid_markers:
        orient_cols = [marker + suffix for suffix in suffixes]
        marker_orients[marker] = df[orient_cols].to_numpy(dtype=np.float32)
        marker_orients[marker] = torch.from_numpy(marker_orients[marker]).to(
            TORCH_DEVICE
        )

        # get the z-vector of the orientations
        # z_vec = torch.tensor([0, 0, 1], dtype=torch.float32).to(TORCH_DEVICE)
        # marker_orients[marker] = quaternion_apply(marker_orients[marker], z_vec)

    return marker_orients


# =====================================================================================
# =============================== VPOSER OPTIMIZER ====================================
# =====================================================================================

DEFAULT_VARS_TO_FIT = [
    "pose_body_latent",
    "trans",
    "root_orient",
]


def fit_bodies(
    num_bodies,
    vp_model,
    params_to_fit,
    loss_function,
    optimizer_args=None,
    initial_body_params=None,
    callback=None,
    verbosity=1,
):
    """
    Fit VPoser to markers by optimizing specified variables.
    loss_function: callable that takes (body_parameters) and computes the loss

    returns the optimized body parameters
    """

    # ========================== INITIAL CONDITIONS =========================

    body_params = {}
    if initial_body_params is not None:
        body_params = {
            k: v.clone().to(TORCH_DEVICE) for k, v in initial_body_params.items()
        }

    if "pose_body" not in body_params:
        body_params["pose_body"] = torch.zeros(
            (num_bodies, 63),
            dtype=torch.float32,
            requires_grad=False,
        ).to(TORCH_DEVICE)
    if "trans" not in body_params:
        body_params["trans"] = torch.zeros(
            (num_bodies, 3),
            dtype=torch.float32,
            requires_grad=False,
        ).to(TORCH_DEVICE)
    if "betas" not in body_params:
        body_params["betas"] = torch.zeros(
            (num_bodies, 10),
            dtype=torch.float32,
            requires_grad=False,
        ).to(TORCH_DEVICE)
    if "root_orient" not in body_params:
        body_params["root_orient"] = torch.zeros(
            (num_bodies, 3),
            dtype=torch.float32,
            requires_grad=False,
        ).to(TORCH_DEVICE)

    body_params["pose_body_latent"] = vp_model.encode(body_params["pose_body"]).mean

    # ========================== SETUP FREE VARS =========================

    free_vars = {
        name: nn.Parameter(params.detach(), requires_grad=True)
        for name, params in body_params.items()
        if name in params_to_fit
    }

    vprint(1, verbosity, f"Free vars: {list(free_vars.keys())}")

    if len(free_vars) == 0:
        raise ValueError("No free variables to optimize. Please check vars_to_fit.")

    # ========================== SETUP OPTIMIZER =========================

    if optimizer_args is None:
        optimizer_args = {
            "type": "lbfgs",
        }
    if "type" not in optimizer_args:
        raise ValueError(
            "optimizer_args must contain a 'type' key specifying the optimizer type"
        )
    optimizer_type = optimizer_args.pop("type")

    if optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS(
            list(free_vars.values()),
            lr=optimizer_args.get("lr", 1),
            max_iter=optimizer_args.get("max_iter", 300),
            tolerance_change=optimizer_args.get("tolerance_change", 1e-5),
            max_eval=optimizer_args.get("max_eval", None),
            history_size=optimizer_args.get("history_size", 100),
            line_search_fn=optimizer_args.get("line_search_fn", "strong_wolfe"),
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # ========================== FIT FUNCTION =========================

    current_loss = None

    iteration_index = 0

    def fit():
        nonlocal iteration_index
        iteration_index += 1

        # zero grads
        optimizer.zero_grad()

        # update body params with free vars
        body_params.update(free_vars)

        # update body pose from latent
        body_params["pose_body"] = (
            vp_model.decode(body_params["pose_body_latent"])["pose_body"]
            .contiguous()
            .view(-1, 63)
        )

        # compute loss
        loss = loss_function(body_params=body_params, idx=iteration_index)

        nonlocal current_loss
        current_loss = loss.item()

        # backprop
        loss.backward()

        # callback
        if callback is not None:
            callback(body_params=body_params, idx=iteration_index, loss=loss)

        # done, return loss for optimizer
        return loss

    # ========================== OPTIMIZATION =========================

    optimizer.step(lambda: fit())

    # ========================== RETURN =========================

    return body_params


def create_loss_function(
    body_model,
    marker_orients: dict,
    markers: dict,
):
    def loss_function(body_params=None, idx=None):
        if body_params is None:
            raise ValueError("body_params cannot be None")

        body = body_model(**body_params)

        # ===================== MARKER ORIENT LOSS =====================
        vertex_normals = compute_vertex_normal_batched(body.v, body.f)

        marker_names = list(markers.keys())
        markers_vert_ids = [markers[name] for name in marker_names]
        target_normals = []
        for name in marker_names:
            target_normals.append(marker_orients[name].unsqueeze(1))
        target_normals = torch.cat(target_normals, dim=1)

        pred_normals = vertex_normals[:, markers_vert_ids, :]

        # compute cosine distance
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        orient_loss = 1 - cos(pred_normals, target_normals)
        orient_loss = orient_loss.mean()

        return orient_loss

    return loss_function


# =====================================================================================
# ===================================== MAIN ==========================================
# =====================================================================================

DEFAULT_TARGET_CONFIG = {
    # "orient_suffixes": ["_rw", "_rx", "_ry", "_rz"],
    "orient_suffixes": ["_nx", "_ny", "_nz"],
    "coord_suffixes": ["_x", "_y", "_z"],
}


def main(
    body_model_path=None,
    vposer_expr_dir=None,
    target_path=None,
    markers_path=None,
    target_config=None,
):
    # ========================= VERIFY INPUTS =========================

    if body_model_path is None or not exists(body_model_path):
        print("Please provide a valid body_model_path")
        return

    if vposer_expr_dir is None or not exists(vposer_expr_dir):
        print("Please provide a valid vposer_expr_dir")
        return

    if target_path is None or not exists(target_path):
        print("Please provide a valid target_path")
        return

    if markers_path is None or not exists(markers_path):
        print("Please provide a valid markers_path")
        return

    if target_config is None:
        target_config = DEFAULT_TARGET_CONFIG

    # ========================= SETUP =========================

    # load smplx
    body_model = get_body_model(body_model_path)
    body_model = body_model.to(TORCH_DEVICE)

    # load vposer
    vp_model, _ = load_model(
        vposer_expr_dir,
        model_code=VPoser,
        remove_words_in_model_weights="vp_model.",
        disable_grad=True,
    )
    vp_model = vp_model.to(TORCH_DEVICE)

    # load markers
    markers_data = np.load(markers_path, allow_pickle=True)
    markers = {
        name: vertex_id
        for name, vertex_id in zip(
            markers_data["marker_names"], markers_data["marker_indices"]
        )
    }

    # load target dataframe
    target_df = pd.read_csv(target_path)

    # trim
    target_df = target_df.iloc[250:500]

    marker_orients = extract_orients_from_df(
        target_df,
        suffixes=target_config["orient_suffixes"],
    )

    # filter markers to those present in both marker_orients and markers
    common_markers = set(marker_orients.keys()).intersection(set(markers.keys()))
    marker_orients = {k: v for k, v in marker_orients.items() if k in common_markers}
    markers = {k: v for k, v in markers.items() if k in common_markers}

    # ====================== OPTIMIZE SETUP ==========================

    num_bodies = list(marker_orients.values())[0].shape[0]
    print(f"Number of bodies to fit: {num_bodies}")

    loss_function = create_loss_function(
        body_model=body_model,
        marker_orients=marker_orients,
        markers=markers,
    )

    def callback(body_params=None, idx=None, loss=None):
        print(f"Callback at iteration {idx}, loss: {loss.item():.4f}")

    optimized_body_params = fit_bodies(
        num_bodies,
        vp_model,
        DEFAULT_VARS_TO_FIT,
        loss_function,
        optimizer_args={
            "type": "lbfgs",
            "max_iter": 0
        },
        callback=callback,
    )

    # ========================= TESTING CODE =========================

    vis = Visualizer(rows=1, cols=1, keepalive=False)

    # for i in range(num_bodies):
    for i in [0]:
        vis.clear_view(view_id=0, re_render=False)
        body_i = index_tensor_dict(optimized_body_params, [i])
        result_body = body_model(**body_i)
        vis.display_mesh(
            view_id=0, body=result_body, show_vertex_normals=True, re_render=False
        )
        vis.set_titlebar(f"Fitted Body {i + 1}/{num_bodies}")

        markers_names = list(markers.keys())
        markers_vert_ids = [markers[name] for name in markers_names]
        marker_vertices = result_body.v[:, markers_vert_ids, :]
        target_normals = []
        for name in markers_names:
            target_normals.append(marker_orients[name][i].unsqueeze(0))
        target_normals = torch.cat(target_normals, dim=0).unsqueeze(0)
        target_lines = _mesh_to_vertex_normal_lines(
            [c2c(marker_vertices[0])],
            normals=c2c(target_normals[0]),
            length=0.1,
            color=colors["orange"],
        )
        marker_spheres = points_to_spheres(
            c2c(marker_vertices[0]), radius=0.01, point_color=colors["red"]
        )
        vis.also_render(
            view_id=0,
            dynamic_lines=[target_lines],
            dynamic_meshes=[marker_spheres],
            persist=False,
        )

        time.sleep(0.01)

    # pos_Z = [
    #     torch.zeros((1, 32), dtype=torch.float32).to(TORCH_DEVICE) for i in range(1)
    # ]
    # for i in range(2700):
    #     for d, z in enumerate(pos_Z):
    #         # z[0][d] += 0.1
    #         pose_body = vp_model.decode(z)["pose_body"].contiguous().view(-1, 63)
    #         test_base = body_model(pose_body=pose_body)
    #         vis.display_mesh(view_id=d, body=test_base, show_vertex_normals=False, re_render=False)

    #         markers_names = list(markers.keys())
    #         markers_vert_ids = [markers[name] for name in markers_names]
    #         marker_vertices = test_base.v[:, markers_vert_ids, :]
    #         target_normals = []
    #         for name in markers_names:
    #             target_normals.append(marker_orients[name][i].unsqueeze(0))
    #         target_normals = torch.cat(target_normals, dim=0).unsqueeze(0)
    #         target_lines = _mesh_to_vertex_normal_lines(
    #             [c2c(marker_vertices[0])],
    #             normals=c2c(target_normals[0]),
    #             length=0.1,
    #             color=colors["orange"],
    #         )
    #         vis.also_render(view_id=d, dynamic_lines=[target_lines], persist=False)

    #     vis.set_titlebar(f"Iteration {i}")

    # # sample body
    # poses = [torch.zeros((1, 63), dtype=torch.float32) for i in range(63)]

    # for i in range(100):
    #     for d, pose in enumerate(poses):
    #         pose[0][d] += 0.01
    #         test_base = body_model(pose_body=pose)
    #         vis.display_mesh(view_id=d, body=test_base)
    #     print(i)

    # =================== GO INTO INTERACTIVE MODE ====================

    print("Done. Entering interactive mode. Live variables:")
    for name, val in locals().items():
        if name.startswith("__") and name.endswith("__"):
            continue
        print(f"--> {name}: {type(val)}")
    code.interact(local=locals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main script.")

    parser.add_argument(
        "--body_model_path",
        type=str,
        default=None,
        help="Path to the body model directory containing model.npz",
    )

    parser.add_argument(
        "--vposer_expr_dir",
        type=str,
        default=None,
        help="Path to the VPoser experiment directory containing snapshots and config",
    )

    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
        help="Path to the target CSV file",
    )

    parser.add_argument(
        "--markers_path",
        type=str,
        default=None,
        help="Path to the markers npz file",
    )

    args = parser.parse_args()

    main(
        body_model_path=args.body_model_path,
        vposer_expr_dir=args.vposer_expr_dir,
        target_path=args.target_path,
        markers_path=args.markers_path,
    )
