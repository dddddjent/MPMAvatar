#!/usr/bin/env bash
# Template command (workspace root, allocated A40, mpmavatar environment):
# bash MPMAvatar/scripts/native_4ddress_pipeline.sh 170 all
# Arguments: SUBJECT = 170|185|190|191; STAGE = prepare|tracking|appearance|physics|evaluate|all
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: native_4ddress_pipeline.sh SUBJECT STAGE' >&2; exit 2; }
SUBJECT=$1
STAGE=$2
case "$STAGE" in prepare|tracking|appearance|physics|evaluate|all) ;; *) echo "Unknown stage: $STAGE" >&2; exit 2 ;; esac
WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PATH="$WORKSPACE/slurm/native_bin:$PATH"
REPO="$WORKSPACE/MPMAvatar"
DATASET="$WORKSPACE/../datasets/4DDress"
MPM_PYTHON=${MPM_PYTHON:-/work/nvme/bivb/junlinl6/conda/envs/mpmavatar/bin/python}
RENDER_PYTHON=${RENDER_PYTHON:-/work/nvme/bivb/junlinl6/conda/envs/synthetic_avatar/bin/python}
case "$SUBJECT" in
    170) TAKE=1; TEST_TAKE=5; START=21; TEST_START=21; MATERIAL_START=21; LOWER_START=60;
         TRAIN_SOURCE="$DATASET/00170_Inner_1/Inner/Take1"; TEST_SOURCE="$DATASET/00170_Inner_1/Inner/Take5"; LABELS=(3 4) ;;
    185) TAKE=1; TEST_TAKE=7; START=11; TEST_START=11; MATERIAL_START=19; LOWER_START=45;
         TRAIN_SOURCE="$DATASET/00185_Inner_1/Inner/Take1"; TEST_SOURCE="$DATASET/00185_Inner_2/Inner/Take7"; LABELS=(3 4) ;;
    190) TAKE=2; TEST_TAKE=5; START=11; TEST_START=11; MATERIAL_START=19;
         TRAIN_SOURCE="$DATASET/00190_Inner/Inner/Take2"; TEST_SOURCE="$DATASET/00190_Inner/Inner/Take5"; LABELS=(3) ;;
    191) TAKE=2; TEST_TAKE=8; START=21; TEST_START=11; MATERIAL_START=21;
         TRAIN_SOURCE="$DATASET/00191_Inner/Inner/Take2"; TEST_SOURCE="$DATASET/00191_Inner/Inner/Take8"; LABELS=(3) ;;
    *) echo "Unknown subject: $SUBJECT" >&2; exit 2 ;;
esac
NAME="s${SUBJECT}_t${TAKE}"
PREPARED="$WORKSPACE/data/MPMAvatar/4DDress/examples/$NAME"
DATADIR="$PREPARED/data"
ASSETS="$DATADIR/$NAME"
TRACKING="$PREPARED/output/tracking/${NAME}_${START}_100"
MODEL="$PREPARED/model/$NAME"
PHYS="$PREPARED/output/phys"
assert_file() { [[ -f "$1" ]] || { echo "Required file missing: $1" >&2; exit 1; }; }
assert_file "$MPM_PYTHON"
assert_file "$RENDER_PYTHON"

prepare() {
    cd "$WORKSPACE"
    [[ ! -e "$PREPARED" ]] || { echo "Prepared destination already exists: $PREPARED" >&2; exit 1; }
    "$MPM_PYTHON" "$REPO/preprocess/prepare_4ddress.py" \
        --train-sequence "$TRAIN_SOURCE" --test-sequence "$TEST_SOURCE" \
        --output "$PREPARED" --smplx "$WORKSPACE/../datasets/smplx" \
        --vposer "$WORKSPACE/../datasets/vposer_v1_0/snapshots/TR00_E096.pt" \
        --train-start "$START" --train-count 100 --test-start "$TEST_START" --test-count 100 \
        --labels "${LABELS[@]}" --template-source "$REPO/data/$NAME" --stage all
}

tracking() {
    cd "$WORKSPACE"
    assert_file "$PREPARED/preparation.json"
    TRACKING_ARGS=()
    if [[ -e "$TRACKING" ]]; then
        assert_file "$TRACKING/tracking_state.pt"
        TRACKING_ARGS=(--resume)
    fi
    "$MPM_PYTHON" "$REPO/preprocess/track_4ddress.py" --prepared "$PREPARED" \
        --render-python "$RENDER_PYTHON" --stage all --wandb-entity '' "${TRACKING_ARGS[@]}"
}

appearance() {
    cd "$REPO"
    assert_file "$PREPARED/tracking_report.json"
    assert_file "$ASSETS/optimized_weights.npy"
    APPEARANCE_ARGS=()
    if [[ -e "$MODEL" ]]; then
        assert_file "$MODEL/training_state.pt"
        APPEARANCE_ARGS=(--start_checkpoint "$MODEL/training_state.pt")
    fi
    "$MPM_PYTHON" "$REPO/train_appearance.py" -m "$MODEL" --iterations 30000 --checkpoint_sampler \
        --dataset_dir "$DATADIR" --uv_path "$ASSETS/mesh_processed.obj" \
        --subject "$SUBJECT" --train_take "$TAKE" --test_take "$TAKE" \
        --train_frame_start_num "$START" 100 --test_frame_start_num "$START" 100 \
        --trained_model_path "$TRACKING" --test_camera_index 0 --dataset_type 4ddress \
        "${APPEARANCE_ARGS[@]}"
    assert_file "$MODEL/point_cloud/timestep_030000/point_cloud.ply"
}

material_command() {
    local save_name=$1
    local split_file=$2
    local train_start=$3
    shift 3
    "$MPM_PYTHON" "$REPO/train_material_params.py" \
        --save_name "$save_name" --trained_model_path "$TRACKING" --model_path "$MODEL" \
        --dataset_dir "$DATADIR" --output_dir "$PHYS" --smplx_gender female \
        --subject "$SUBJECT" --train_take "$TAKE" --test_take "$TEST_TAKE" \
        --verts_start_idx "$START" --split_idx_path "$ASSETS/$split_file" \
        --dataset_type 4ddress --uv_path "$ASSETS/mesh_processed.obj" --test_camera_index 0 \
        --train_frame_start_num "$train_start" "$@"
}

fit_material() {
    local save_name=$1
    local split_file=$2
    local train_start=$3
    local state="$PHYS/$save_name/seed0/training_state.pt"
    local resume_args=()
    if [[ -f "$PHYS/$save_name/seed0/best_param_00199.npz" ]]; then
        echo "Material fit already complete: $save_name"
        return
    fi
    if [[ -e "$PHYS/$save_name" ]]; then
        assert_file "$state"
        resume_args=(--resume)
    fi
    material_command "$save_name" "$split_file" "$train_start" 12 \
        --test_frame_start_num "$train_start" 2 --iterations 200 --init_params_path '' \
        --checkpoint_material --wandb_name "invphys_$save_name" --visualize "${resume_args[@]}"
    assert_file "$PHYS/$save_name/seed0/best_param_00199.npz"
}

physics() {
    cd "$REPO"
    assert_file "$MODEL/point_cloud/timestep_030000/point_cloud.ply"
    if [[ "$SUBJECT" == 170 || "$SUBJECT" == 185 ]]; then
        fit_material "${NAME}_upper" split_idx_upper.npz "$MATERIAL_START"
        fit_material "${NAME}_lower" split_idx_lower.npz "$LOWER_START"
    else
        fit_material "$NAME" split_idx.npz "$MATERIAL_START"
    fi
}

simulate_part() {
    local save_name=$1
    local split_file=$2
    shift 2
    assert_file "$PHYS/$save_name/seed0/best_param_00199.npz"
    material_command "$save_name" "$split_file" "$MATERIAL_START" 2 \
        --test_frame_start_num "$TEST_START" 100 \
        --init_params_path "$PHYS/$save_name/seed0/best_param_00199.npz" --run_eval --checkpoint_eval "$@"
}

evaluate() {
    cd "$REPO"
    if [[ -f "$PHYS/$NAME/seed0/metric.npz" ]]; then
        echo "Evaluation already complete: $PHYS/$NAME/seed0/metric.npz"
        return
    fi
    assert_file "$MODEL/point_cloud/timestep_030000/point_cloud.ply"
    if [[ "$SUBJECT" == 170 || "$SUBJECT" == 185 ]]; then
        simulate_part "${NAME}_upper" split_idx_upper.npz --skip_render
        simulate_part "${NAME}_lower" split_idx_lower.npz --skip_render
        cd "$PREPARED"
        "$MPM_PYTHON" "$REPO/merge_meshes.py" --seq "$NAME" --output_dir output/phys --data_dir "$DATADIR"
        cd "$REPO"
        material_command "$NAME" split_idx_upper.npz "$MATERIAL_START" 2 \
            --test_frame_start_num "$TEST_START" 100 \
            --init_params_path "$PHYS/${NAME}_upper/seed0/best_param_00199.npz" --run_eval --checkpoint_eval --skip_sim
    else
        simulate_part "$NAME" split_idx.npz
    fi
    "$MPM_PYTHON" "$REPO/eval.py" --output_path "$PHYS/$NAME/seed0" \
        --mesh_path "$ASSETS/mesh_processed.obj" \
        --data_path "$DATADIR/4D-DRESS/00${SUBJECT}_Inner/Inner/Take${TEST_TAKE}" \
        --start_idx "$TEST_START" --num_timesteps 100 --dataset 4ddress
    assert_file "$PHYS/$NAME/seed0/metric.npz"
}

printf 'Native 4D-DRESS subject %s, stage %s, destination %s\n' "$SUBJECT" "$STAGE" "$PREPARED"
if [[ "$STAGE" == physics || "$STAGE" == evaluate ]]; then
    # Duplicate A40/A100 chains share outputs; one job per subject holds this Lustre flock.
    exec 9>"$PREPARED/.pipeline.lock"
    flock -n 9 || { echo "Subject $SUBJECT is locked by another job; exiting"; exit 0; }
fi
case "$STAGE" in
    all) prepare; tracking; appearance; physics; evaluate ;;
    *) "$STAGE" ;;
esac
