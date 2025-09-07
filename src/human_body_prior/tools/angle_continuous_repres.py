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
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np


# Numpy implementation of vector normalization
def normalize_vector(vector):
    """
    Normalize a vector to unit length.

    Original code:
    def norm(v):
        return v/np.linalg.norm(v)
    """
    return vector / np.linalg.norm(vector)


# Gram-Schmidt orthogonalization process for 3D space
def gram_schmidt_3d(matrix):
    """
    Creates an orthonormal basis from the first two columns of a matrix using Gram-Schmidt process.
    Returns a 3x3 rotation matrix.

    Original code:
    def gs(M):
        a1 = M[:,0]
        a2 = M[:,1]
        b1 = norm(a1)
        b2 = norm((a2-np.dot(b1,a2)*b1))
        b3 = np.cross(b1,b2)
        return np.vstack([b1,b2,b3]).T
    """
    # Get the two input vectors
    vector1 = matrix[:, 0]
    vector2 = matrix[:, 1]

    # Normalize the first vector to get the first basis vector
    basis1 = normalize_vector(vector1)

    # Project vector2 onto basis1 and subtract to get orthogonal component
    # Then normalize to get the second basis vector
    projection = np.dot(basis1, vector2) * basis1
    basis2 = normalize_vector(vector2 - projection)

    # Cross product gives the third orthogonal vector
    basis3 = np.cross(basis1, basis2)

    # Stack the basis vectors to form a rotation matrix
    return np.vstack([basis1, basis2, basis3]).T


# Batched Gram-Schmidt process using PyTorch
def batched_gram_schmidt(input_vectors):
    """
    Performs Gram-Schmidt orthogonalization on a batch of 3D vector pairs.

    Args:
        input_vectors: Tensor of shape (batch_size, 3, 2) where each sample contains two 3D vectors

    Returns:
        Rotation matrices of shape (batch_size, 3, 3)

    Original code:
    def bgs(d6s):
        bsz = d6s.shape[0]
        b1 = F.normalize(d6s[:,:,0], p=2, dim=1)
        a2 = d6s[:,:,1]
        c = torch.bmm(b1.view(bsz,1,-1),a2.view(bsz,-1,1)).view(bsz,1)*b1
        b2 = F.normalize(a2-c,p=2,dim=1)
        b3=torch.cross(b1,b2,dim=1)
        return torch.stack([b1,b2,b3],dim=1).permute(0,2,1)
    """
    batch_size = input_vectors.shape[0]

    # Normalize the first vector to create first basis vector
    basis1 = F.normalize(input_vectors[:, :, 0], p=2, dim=1)

    # Get the second input vector
    vector2 = input_vectors[:, :, 1]

    # Calculate projection of vector2 onto basis1
    # torch.bmm performs batch matrix multiplication
    projection_magnitude = torch.bmm(
        basis1.view(batch_size, 1, -1), vector2.view(batch_size, -1, 1)
    ).view(batch_size, 1)

    projection_vector = projection_magnitude * basis1

    # Subtract projection and normalize to get second basis vector
    basis2 = F.normalize(vector2 - projection_vector, p=2, dim=1)

    # Cross product gives third basis vector, completing the orthonormal basis
    basis3 = torch.cross(basis1, basis2, dim=1)

    # Stack and rearrange to get rotation matrices of shape (batch_size, 3, 3)
    return torch.stack([basis1, basis2, basis3], dim=1).permute(0, 2, 1)


class geodesic_loss_R(nn.Module):
    """
    Computes the geodesic distance between rotation matrices.
    The geodesic distance is the angle of rotation in the rotation space.
    """

    EPSILON = 1e-6

    def __init__(self, reduction="batchmean"):
        """
        Initialize the geodesic loss for rotation matrices.

        Args:
            reduction: How to reduce the loss ('mean', 'batchmean', or None)
        """
        super(geodesic_loss_R, self).__init__()
        self.reduction = reduction
        self.eps = 1e-6

    def batch_geodesic_distance(self, rotation1, rotation2):
        """
        Compute the geodesic distance between batches of rotation matrices.

        Original code:
        def bgdR(self,m1,m2):
            batch = m1.shape[0]
            m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
            cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
            cos = torch.min(cos, m1.new(np.ones(batch)))
            cos = torch.max(cos, m1.new(np.ones(batch)) * -1)
            return torch.acos(cos)
        """
        batch_size = rotation1.shape[0]

        # Compute the relative rotation: R_rel = R1 @ R2.T
        relative_rotation = torch.bmm(rotation1, rotation2.transpose(1, 2))

        # The trace of the relative rotation minus 1, divided by 2, gives the cosine of the angle
        # tr(R) = 1 + 2*cos(θ) for a rotation matrix R with angle θ
        trace = (
            relative_rotation[:, 0, 0]
            + relative_rotation[:, 1, 1]
            + relative_rotation[:, 2, 2]
        )
        cos_angle = (trace - 1) / 2

        # Clamp to valid range [-1, 1] to avoid numerical issues
        cos_angle = torch.clamp(
            cos_angle,
            -1 + geodesic_loss_R.EPSILON,
            1 - geodesic_loss_R.EPSILON,
        )
        # cos_angle = torch.min(cos_angle, rotation1.new(np.ones(batch_size)))
        # cos_angle = torch.max(cos_angle, rotation1.new(np.ones(batch_size)) * -1)

        # Convert to angle (in radians)
        return torch.acos(cos_angle)

    def forward(self, predicted_rotation, true_rotation):
        """
        Compute the geodesic loss between predicted and true rotation matrices.

        Args:
            predicted_rotation: Predicted rotation matrices
            true_rotation: Ground truth rotation matrices

        Returns:
            The geodesic distance loss according to the specified reduction method
        """
        theta = self.batch_geodesic_distance(predicted_rotation, true_rotation)

        # check for nans
        with torch.no_grad():
            if torch.any(torch.isnan(theta)) or torch.any(torch.isinf(theta)):
                raise ValueError("NaN detected in geodesic loss computation!")

        if self.reduction == "mean":
            return torch.mean(theta)
        elif self.reduction == "batchmean":
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))
        else:
            return theta
