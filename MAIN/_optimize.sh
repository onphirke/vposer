uv run main.py optimize \
    --body_model_path ../support_data/dowloads/models/smplx/neutral/model.npz \
    --vposer_expr_dir ../_good_runs/V02_05 \
    --target_path ../_data/quaternions_from_euler.csv \
    --markers_path ./markers.npz \
    --optimized_body_path ./optimized_body_params.pt

    # --target_path ../_data/quaternions_from_oriinc.csv \