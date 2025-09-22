

On WSL2g, have to use `PYOPENGL_PLATFORM=glx` for correctness



## Body Model
[Body Model File](src/human_body_prior/body_model/body_model.py)

#### Key Notes
- The body model data we use is SMPL-X, as it was easier to work with. It is a superset of SMPL, so we can incorporate that as well, but I haven't done that yet.
- VPoser only needs SMPL body data
- SMPL has 21 joints, so 21 * 3 (joints) + 3 (root joint) = 66 parameters
- VPoser is trained to output the pose of the body

## Inverse Kinematics

#### Key Notes
- Pretty sensitive to optimization setup. Will have to experiment with the PPMI data


image labels retrieved from https://www.researchgate.net/publication/391284907_A_Method_for_Redirecting_Humanoid_Robots_Based_on_Segmented_Geometric_Inverse_Kinematics