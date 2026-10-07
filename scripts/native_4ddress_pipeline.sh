#!/usr/bin/env bash
# Template command (workspace root, allocated A40, mpmavatar environment):
# bash MPMAvatar/scripts/native_4ddress_pipeline.sh 170 all
# Arguments: SUBJECT = 170|185|190|191 (shipped examples) or 134|148|169|187 (new cases,
# MPMAvatar/data/4ddress_cases/s<SUBJECT>.json); STAGE = prepare|tracking|appearance|physics|evaluate|all, or physics_upper|physics_lower
# (one garment of a two-garment case, so both fits can run on two GPUs at once)
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: native_4ddress_pipeline.sh SUBJECT STAGE' >&2; exit 2; }
SUBJECT=$1
STAGE=$2
case "$STAGE" in prepare|tracking|appearance|physics|physics_upper|physics_lower|evaluate|all) ;; *) echo "Unknown stage: $STAGE" >&2; exit 2 ;; esac
WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PATH="$WORKSPACE/slurm/native_bin:$PATH"
REPO="$WORKSPACE/MPMAvatar"
DATASET="$WORKSPACE/../datasets/4DDress"
MPM_PYTHON=${MPM_PYTHON:-/work/nvme/bivb/junlinl6/conda/envs/mpmavatar/bin/python}
RENDER_PYTHON=${RENDER_PYTHON:-/work/nvme/bivb/junlinl6/conda/envs/synthetic_avatar/bin/python}
assert_file() { [[ -f "$1" ]] || { echo "Required file missing: $1" >&2; exit 1; }; }
# Shipped examples: author templates/splits and 12-frame material windows.
TEMPLATE_ARGS=()
MATERIAL_FRAMES=12
case "$SUBJECT" in
    170) TAKE=1; TEST_TAKE=5; START=21; TEST_START=21; MATERIAL_START=21; LOWER_START=60;
         TRAIN_SOURCE="$DATASET/00170_Inner_1/Inner/Take1"; TEST_SOURCE="$DATASET/00170_Inner_1/Inner/Take5"; LABELS=(3 4) ;;
    185) TAKE=1; TEST_TAKE=7; START=11; TEST_START=11; MATERIAL_START=19; LOWER_START=45;
         TRAIN_SOURCE="$DATASET/00185_Inner_1/Inner/Take1"; TEST_SOURCE="$DATASET/00185_Inner_2/Inner/Take7"; LABELS=(3 4) ;;
    190) TAKE=2; TEST_TAKE=5; START=11; TEST_START=11; MATERIAL_START=19;
         TRAIN_SOURCE="$DATASET/00190_Inner/Inner/Take2"; TEST_SOURCE="$DATASET/00190_Inner/Inner/Take5"; LABELS=(3) ;;
    191) TAKE=2; TEST_TAKE=8; START=21; TEST_START=11; MATERIAL_START=21;
         TRAIN_SOURCE="$DATASET/00191_Inner/Inner/Take2"; TEST_SOURCE="$DATASET/00191_Inner/Inner/Take8"; LABELS=(3) ;;
    134|148|169|187)
        # New cases: template built from the first training scan; per-garment 10-frame
        # material windows and the lower split are chosen after tracking (select_4ddress_material.py).
        CASE="$REPO/data/4ddress_cases/s${SUBJECT}.json"
        assert_file "$CASE"
        TAKE=$(jq -r .train_take "$CASE"); TEST_TAKE=$(jq -r .test_take "$CASE")
        START=$(jq -r .train_start "$CASE"); TEST_START=$(jq -r .test_start "$CASE")
        TRAIN_SOURCE=$(jq -r .train_source "$CASE"); TEST_SOURCE=$(jq -r .test_source "$CASE")
        mapfile -t LABELS < <(jq -r '.labels[]' "$CASE")
        MATERIAL_FRAMES=10 ;;
    *) echo "Unknown subject: $SUBJECT" >&2; exit 2 ;;
esac
case "$SUBJECT" in 170|185|190|191) TEMPLATE_ARGS=(--template-source "$REPO/data/s${SUBJECT}_t${TAKE}") ;; esac
NAME="s${SUBJECT}_t${TAKE}"
PREPARED="$WORKSPACE/data/MPMAvatar/4DDress/examples/$NAME"
DATADIR="$PREPARED/data"
ASSETS="$DATADIR/$NAME"
TRACKING="$PREPARED/output/tracking/${NAME}_${START}_100"
MODEL="$PREPARED/model/$NAME"
PHYS="$PREPARED/output/phys"
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
        --labels "${LABELS[@]}" "${TEMPLATE_ARGS[@]}" --stage all
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
    local gender
    gender=$(jq -r .train.gender "$PREPARED/preparation.json")
    "$MPM_PYTHON" "$REPO/train_material_params.py" \
        --save_name "$save_name" --trained_model_path "$TRACKING" --model_path "$MODEL" \
        --dataset_dir "$DATADIR" --output_dir "$PHYS" --smplx_gender "$gender" \
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
    if [[ -f "$PHYS/$save_name/seed0/best_param_final.npz" ]]; then
        echo "Material fit already complete: $save_name"
        return
    fi
    if [[ -e "$PHYS/$save_name" ]]; then
        assert_file "$state"
        resume_args=(--resume)
    fi
    material_command "$save_name" "$split_file" "$train_start" "$MATERIAL_FRAMES" \
        --test_frame_start_num "$train_start" 2 --iterations 200 --init_params_path '' \
        --early_stop_patience 40 --early_stop_tol 0.01 --checkpoint_material --wandb_name "invphys_$save_name" --visualize "${resume_args[@]}"
    assert_file "$PHYS/$save_name/seed0/best_param_final.npz"
}

material_setup() {
    # New cases only: windows (and lower fix_v/splits) from the tracked training frames.
    [[ -z "${CASE:-}" ]] && return
    cd "$WORKSPACE"
    # physics_upper and physics_lower may start together; one writes the setup and splits.
    exec 8>"$PREPARED/.material_setup.lock"
    flock 8
    if [[ ! -f "$PREPARED/material_setup.json" ]]; then
        if [[ ! -f "$TRACKING/tracking_geo_metric.json" ]]; then
            (cd "$REPO" && "$MPM_PYTHON" eval_tracking.py --prepared "$PREPARED" --seed 0)
        fi
        # Windows may only use well-tracked frames: limits are the worst frame of the
        # original 170/185/190/191 material windows (garment CD 0.222, label stretch 35.9).
        "$MPM_PYTHON" "$REPO/preprocess/select_4ddress_material.py" --prepared "$PREPARED" \
            --frames "$MATERIAL_FRAMES" --joint-fraction 0.18 --max-garment-cd 0.222 --max-stretch 35.9
    fi
    MATERIAL_START=$(jq -r '.windows["3"].start' "$PREPARED/material_setup.json")
    if [[ ${#LABELS[@]} == 2 ]]; then
        LOWER_START=$(jq -r '.windows["4"].start' "$PREPARED/material_setup.json")
        if [[ ! -f "$ASSETS/split_idx_lower.npz" ]]; then
            "$MPM_PYTHON" "$REPO/preprocess/split_garments.py" --mesh_path "$ASSETS/mesh_processed.obj" \
                --cloth_npz "$ASSETS/cloth_vertices.npz" --labels 3 --iteration 0 \
                --filename "$ASSETS/split_idx_upper.npz"
            "$MPM_PYTHON" "$REPO/preprocess/split_garments.py" --mesh_path "$ASSETS/mesh_processed.obj" \
                --cloth_npz "$ASSETS/cloth_vertices.npz" --labels 4 --fix_v "$ASSETS/fix_v.npy" \
                --iteration "$(jq -r .lower_split.iteration "$PREPARED/material_setup.json")" \
                --filename "$ASSETS/split_idx_lower.npz"
        fi
    fi
    flock -u 8
}

physics() {
    cd "$REPO"
    assert_file "$MODEL/point_cloud/timestep_030000/point_cloud.ply"
    material_setup
    cd "$REPO"
    if [[ ${#LABELS[@]} == 2 ]]; then
        fit_material "${NAME}_upper" split_idx_upper.npz "$MATERIAL_START"
        fit_material "${NAME}_lower" split_idx_lower.npz "$LOWER_START"
    else
        fit_material "$NAME" split_idx.npz "$MATERIAL_START"
    fi
}

physics_garment() {
    local label=$1
    [[ ${#LABELS[@]} == 2 ]] || { echo "physics_$label needs a two-garment case" >&2; exit 2; }
    cd "$REPO"
    assert_file "$MODEL/point_cloud/timestep_030000/point_cloud.ply"
    material_setup
    cd "$REPO"
    if [[ $label == upper ]]; then
        fit_material "${NAME}_upper" split_idx_upper.npz "$MATERIAL_START"
    else
        fit_material "${NAME}_lower" split_idx_lower.npz "$LOWER_START"
    fi
}

physics_upper() { physics_garment upper; }
physics_lower() { physics_garment lower; }

simulate_part() {
    local save_name=$1
    local split_file=$2
    shift 2
    assert_file "$PHYS/$save_name/seed0/best_param_final.npz"
    material_command "$save_name" "$split_file" "$MATERIAL_START" 2 \
        --test_frame_start_num "$TEST_START" 100 \
        --init_params_path "$PHYS/$save_name/seed0/best_param_final.npz" --run_eval --checkpoint_eval "$@"
}

evaluate() {
    cd "$REPO"
    if [[ -f "$PHYS/$NAME/seed0/metric.npz" ]]; then
        echo "Evaluation already complete: $PHYS/$NAME/seed0/metric.npz"
        return
    fi
    assert_file "$MODEL/point_cloud/timestep_030000/point_cloud.ply"
    material_setup
    cd "$REPO"
    if [[ ${#LABELS[@]} == 2 ]]; then
        simulate_part "${NAME}_upper" split_idx_upper.npz --skip_render
        simulate_part "${NAME}_lower" split_idx_lower.npz --skip_render
        cd "$PREPARED"
        "$MPM_PYTHON" "$REPO/merge_meshes.py" --seq "$NAME" --output_dir output/phys --data_dir "$DATADIR"
        cd "$REPO"
        material_command "$NAME" split_idx_upper.npz "$MATERIAL_START" 2 \
            --test_frame_start_num "$TEST_START" 100 \
            --init_params_path "$PHYS/${NAME}_upper/seed0/best_param_final.npz" --run_eval --checkpoint_eval --skip_sim
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
if [[ "$STAGE" == physics* || "$STAGE" == evaluate ]]; then
    # Duplicate A40/A100 chains share outputs; one job per subject (per garment for physics_<garment>) holds this Lustre flock.
    lock_name=.pipeline.lock
    [[ "$STAGE" == physics_* ]] && lock_name=".pipeline_${STAGE#physics_}.lock"
    exec 9>"$PREPARED/$lock_name"
    flock -n 9 || { echo "Subject $SUBJECT is locked by another job; exiting"; exit 0; }
fi
case "$STAGE" in
    all) prepare; tracking; appearance; physics; evaluate ;;
    *) "$STAGE" ;;
esac
