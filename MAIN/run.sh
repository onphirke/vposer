
uv run main.py \
    --body_model_path ../support_data/dowloads/models/smplx/neutral/model.npz \
    --vposer_expr_dir ../_good_runs/V02_05 \
    --target_path ../_data/normals_euler.csv \
    --markers_path ./markers.npz
    # --target_path ../_data/test_orients.csv \
    # --target_path ../_data/normals.csv \