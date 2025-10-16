uv run main.py visualize \
    --body_model_path ../support_data/dowloads/models/smplx/neutral/model.npz \
    --optimized_body_path ./optimized_body_params.pt \
    --target_path ../_data/quaternions2_from_euler.csv \
    --target_indices 0 1500 \
    --markers_path ./markers.npz

    # --target_path ../_data/quaternions_from_oriinc.csv \