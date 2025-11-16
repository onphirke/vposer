##### `body_model/`

This directory implements the **SMPL-X Body Model** described by the equations $M(\beta, \theta, \psi)$ and $T_p(\beta, \theta, \psi)$.

* **`body_model.py`:** This is the main implementation of the SMPL-X model.
    * **Connection to Approach:** This class loads the model parameters (like the template mesh $\bar{T}$, shape blend shapes $\mathcal{S}$, pose blend shapes $\mathcal{P}$, and expression blend shapes $\mathcal{E}$) and defines the forward pass that computes the final mesh $M(\beta, \theta, \psi)$. It takes **shape** ($\beta$), **pose** ($\theta$), and **expression** ($\psi$) parameters as input.
* **`lbs.py`:** Implements the **Linear Blend Skinning** function.
    * **Connection to Approach:** This file contains the code for the $W(...)$ function, which deforms the mesh vertices based on the posed skeleton $J(\beta)$ and the learned blend weights $\mathcal{W}$.
* **`rigid_object_model.py`:** Defines rigid object models.
    * **Connection to Approach:** Implements the `RigidObjectModel` class, which defines a rigid object mesh (loaded via `psbody.mesh.Mesh`) with learnable global translation (`trans`) and root orientation (`root_orient`) parameters. The rigid object can be posed via batch Rodrigues rotation and then transformed into the world frame, effectively extending the core SMPL-X model to handle held objects or environmental interactions.
* **`parts_segm/`:** Utilities for body part segmentation.
    * **Connection to Approach:** Contains mapping utilities for associating body parts with mesh vertices (see `bodypart2vertexid.py` in `tools/`). These mappings are essential for computing the joint regressor $J(\beta)$, which maps vertices from the posed body mesh to joint positions.

---

##### `models/`

This directory implements the **Body Pose Prior (VPoser)**, which is the Variational Autoencoder (VAE) used to learn a latent space for valid body poses ($\theta$).

* **`vposer_model.py`:** This is the core VAE model implementation.
    * **Connection to Approach:** This file defines the **Encoder** $q(Z|R)$ (which maps input poses $R$ to the latent space $Z$) and the **Decoder** $p(R|Z)$ (which reconstructs poses $\hat{R}$ from a latent code $Z$). The model computes the mean and log-variance of the latent distribution $q(Z|R) = \mathcal{N}(\mu_Z, \Sigma_Z)$, enabling reparameterization sampling during training.
* **`model_components.py`:** Reusable neural network blocks (e.g., linear layers, normalizers) used to build the VAE in `vposer_model.py`.
* **`ik_engine.py`:** An Inverse Kinematics (IK) solver.
    * **Connection to Approach:** This is a key *application* of the VPoser. It uses the learned pose prior (the VAE decoder) to find a realistic 3D pose $\theta$ that matches a target (e.g., 2D keypoints), effectively solving the "non-trivial" 2D-to-3D mapping problem mentioned in the approach.

---

##### `train/`

This directory contains the training infrastructure for the VPoser VAE.

* **`vposer_trainer.py`:** The main trainer class.
    * **Connection to Approach:** This script implements the **total loss function $L_{total}$** by combining its five components. ***NOTE THIS SEEMS TO BE DIFFERENT THAN THE APPROACH DESCRIBED IN THE PAPER***:
        * $\mathcal{L}_{KL}$ (KL Divergence): Computed in `_compute_loss()` using `torch.distributions.normal.Normal` and `torch.distributions.kl.kl_divergence()` against a standard normal distribution $\mathcal{N}(0,I)$, then summed and mean-reduced.
        * $\mathcal{L}_{rec}$ (Reconstruction): Implemented as L1 loss (`torch.nn.L1Loss`) between the reconstructed mesh vertices `bm_rec.v` and original mesh vertices `bm_orig.v`, computed via the body model (not directly on rotations $R$, but on the resulting mesh).
        * $\mathcal{L}_{orth}$ (Orthogonality): Implemented via the `geodesic_loss_R` class from `angle_continuous_repres.py`, which computes the geodesic distance between rotation matrices by computing the trace of the relative rotation and converting to the angle via $\cos(\theta) = \frac{\text{tr}(R)-1}{2}$.
        * $\mathcal{L}_{det1}$ (Determinant): Not explicitly implemented as a separate loss term. Instead, orthogonality is enforced implicitly through the geodesic loss and by ensuring the decoder outputs properly constrained rotation representations.
        * $\mathcal{L}_{reg}$ (Regularization): Handled by PyTorch's optimizer (ADAM) with weight decay parameter, applied to all learnable parameters $\phi$ of the VAE.
    * It also implements the **ADAM solver** and uses the loss weights ($c_1$ through $c_5$) to train the VAE parameters $\phi$.
* **`V02_05/` and `Vme/`:** Subdirectories containing configuration files for pre-trained models.

---

##### `tools/`

Utility functions that support both the body model and the VAE.

* **`rotation_tools.py`:** Contains functions for rotation representation conversions.
    * **Connection to Approach:** This is a critical file. The VAE operates on 23 joints, each represented by a rotation matrix $R$ (totaling 207 parameters). This module provides functions to convert between representations (e.g., axis-angle $\theta$ to matrix $R$) and to compute the rotation-specific losses **$\mathcal{L}_{orth}$** and **$\mathcal{L}_{det1}$**.
* `angle_continuous_repres.py`: Helper for handling continuous angle representations, related to `rotation_tools.py`.
* `model_loader.py`: Loads pre-trained VPoser (`vposer_model.py`) and SMPL-X (`body_model.py`) models.
* `configurations.py`: Manages configuration files (e.g., loading model paths, hyperparameters like loss weights).
* `bodypart2vertexid.py`: Mappings for body parts to mesh vertices, likely used by `body_model/parts_segm/`.
* `tgm_conversion.py` / `omni_tools.py`: General utilities.

---

##### `data/`, `evaluations/`, `visualizations/`

* **`data/`:** Contains scripts for handling and preprocessing the training data.
    * **Connection to Approach:** This is where the **AMASS**, **CMU**, and **Human 3.6M** datasets are loaded and processed into the SMPL poses ($R$) used to train the VAE.
* **`evaluations/`:** Scripts to evaluate the trained VPoser, likely by measuring the **Reconstruction Term** ($\mathcal{L}_{rec}$) on a test set.
* **`visualizations/`:** Utilities to render the final 3D mesh output $M(\beta, \theta, \psi)$ for visual inspection.

---

##### Application and Execution: `./src/`, `./MAIN/`, `./tutorials/`

* **`./src/main.py`:** The main entry point for running VPoser training or evaluation.
* **`./src/ikmain.py`:** The main entry point for running the **IK solver** (`ik_engine.py`).
* **`./src/main.yaml`:** Configuration file specifying hyperparameters (e.g., learning rate, latent dimension, loss weights $c_1-c_5$).

##### `./MAIN/`
Contains practical application scripts that use the `human_body_prior` library for body pose fitting from IMU sensor streams and visualization. These were coded by me and don't really have a paper reference, and instead was to try to infer body pose from the IMU sensors.

* **`main.py`:** The primary application script with two main workflows:
    * **`main_optimize()`:** Fits SMPL-X body models to marker orientation data (quaternions) from IMU Sensors. Uses loss functions that combine:
        * Marker orientation matching (via cosine similarity or geodesic distance)
        * Body pose latent regularization to keep poses within the learned prior
        * Smoothness regularization to enforce temporal coherence (the function is not very strong though)
        * Optimizes using LBFGS or ADAM to solve $\arg\min_{\beta,\theta,\psi} L(\text{markers}, M(\beta,\theta,\psi))$
    * **`main_visualize()`:** Visualizes fitted body parameters with interactive playback and displays target marker orientations as 3D axes
    * Uses `Visualizer` class for 3D mesh rendering with psbody.mesh and body_visualizer
    * Implements multiple loss components: `BodyOrientLoss_Cosine`, `BodyOrientLoss_Geodesic`, `BodyPoseLatentRegularizationLoss`, `BodyPoseLatentSmoothnessLoss`

* **`marker_editor.py`:** Interactive tool for placing and editing body markers on the SMPL mesh:
    * Allows 3D visualization of the SMPL body using psbody.mesh
    * Keyboard-based marker placement on mesh vertices
    * Interactive commands: `add()`, `add_vertex()`, `rename()`, `delete()`, `save()`, `list()`
    * Saves marker definitions to an NPZ file mapping marker names to vertex indices
    
* **`npz_inspect.py`:** Utility tool for inspecting and analyzing NumPy `.npz` data files:
    * Displays array contents, shapes, data types, and statistics
    * Provides interactive mode for exploring file structure
    * Useful for debugging and understanding saved data formats

* **`optimized_body_params.pt`:** Saved PyTorch tensor containing optimized body parameters ($\beta$, $\theta$, $\psi$) from a fitting run
* **`markers.npz`:** Marker definitions file containing marker names, vertex indices, and orientation reference vertices
* **`_gen/`:** Directory for generating and storing intermediate results (e.g., `optimized_body_params_100_1500.pt`)
* **`_optimize.sh`, `_visualize.sh`:** Shell scripts for running optimization and visualization workflows

##### `./tutorials/`
Jupyter notebooks and scripts demonstrating VPoser usage:
* **`vposer.ipynb`:** Shows how to load the VAE, sample from the latent space $Z$, and use the decoder to generate novel, realistic human poses $\hat{R}$ (which can then be fed as $\theta$ into the SMPL-X model).
* **`ik_example_*.py`:** Scripts demonstrating inverse kinematics fitting to motion capture data
* **`ik_joints_parkinson.py`:** IK example applied to Parkinson's patient motion data

---

##### Tests and Data: `./tests/`, `./DATAPROC/`, `_data/`, `_good_runs/`

* **`./tests/`:** Unit tests.
    * `test_vposer.py`: Tests the VAE implementation (`vposer_model.py`).
    * `test_rotations.py`: Tests the crucial functions in `rotation_tools.py` (which are needed for the loss functions).
* **`./DATAPROC/`:** Data processing scripts, e.g., converting mocap data (like Euler angles) into the rotation matrix format $R$ required by the VAE. These are currently incomplete.
* **`_data/`:** Default directory for datasets.
    * `amass/`: Contains the **AMASS dataset** subsets used for training.
    * `_runs/`: Default output directory for training logs and checkpoints.
* **`_good_runs/`:** Contains the final, pre-trained model checkpoints (the learned parameters $\phi$ of the VAE) that are loaded by `model_loader.py`.