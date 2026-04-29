#!/usr/bin/env bash
set -e

# Configure these paths before running, either by editing the placeholders below
# or by exporting the variables in your shell.
export PROJECT_ROOT="${PROJECT_ROOT:-/path/to/SADGS}"
export MIPNERF360_DATASET="${MIPNERF360_DATASET:-/path/to/datasets/360_v2}"
export TANDT_DB_DATASET="${TANDT_DB_DATASET:-/path/to/datasets/tandt_db/db}"
export TANDT_DATASET="${TANDT_DATASET:-/path/to/datasets/tandt_db/tandt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "ERROR: PROJECT_ROOT does not exist: $PROJECT_ROOT"
    echo "Edit run_train.sh or export PROJECT_ROOT before running."
    exit 1
fi

cd "$PROJECT_ROOT"

# Installation reference:
# conda create -n sadgs python=3.12 -y
# conda activate sadgs
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install plyfile tqdm websockets
# pip install submodules/diff-gaussian-rasterization_structgs
# pip install submodules/simple-knn
# pip install submodules/fused-ssim

run_dataset_pipeline() {
    local DATASET_PATH="$1"
    local SWITCH_ID="$2"

    if [ "$#" -lt 3 ]; then
        echo "Error: Usage: run_dataset_pipeline <dataset_path> <switch_id> <scene1> [scene2...]"
        return 1
    fi

    if [ ! -d "$DATASET_PATH" ]; then
        echo "ERROR: Dataset path does not exist: $DATASET_PATH"
        return 1
    fi

    shift 2
    local SCENE_LIST=("$@")

    export MIPDATASET="$DATASET_PATH"

    for JOB in "${SCENE_LIST[@]}"; do
        echo "=============================="
        echo "Processing scene: $JOB"
        echo "Dataset: $MIPDATASET"
        echo "Mode (Switch): $SWITCH_ID"
        echo "GPU-ID: $CUDA_VISIBLE_DEVICES"
        echo "=============================="

        case "$SWITCH_ID" in
            1)
                echo "Running Configuration 1 (Tanks and Temples)..."
                OAR_JOB_ID="$JOB" python train.py -s "$MIPDATASET/$JOB" -i images --eval --ks_scale_power 1.2 --highfeature_lr 0.01 --split_ratio_threshold 0.4 --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.4 --optimizer_type hybrid --sample_bbox_faces --warmup_densification --mult 0.7 --iterations 7000 --position_lr_max_steps 7000 --densification_interval 500 --save_iterations 7000 --opacity_reset_interval 3000 --densify_from_iter 500 --densify_until_iter 4000 --prune_iterations 4000
                python render.py -m "output/$JOB" --iteration 7000 --skip_train --mult 0.7
                ;;

            2)
                echo "Running Configuration 2 (Mip-NeRF 360 indoor)..."
                OAR_JOB_ID="$JOB" python train.py -s "$MIPDATASET/$JOB" -i images --eval --scale_rotation_scheduler --adam_eps_order 10 --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.5 --batch_size 2 --optimizer_type hybrid --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --save_iterations 3000 --prune_iterations 1900
                python render.py -m "output/$JOB" --skip_train --iteration 3000
                ;;

            3)
                echo "Running Configuration 3 (Mip-NeRF 360 outdoor)..."
                OAR_JOB_ID="$JOB" python train.py -s "$MIPDATASET/$JOB" -i images --eval --optimizer_type hybrid --split_ratio_threshold 0.4 --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --save_iterations 3000 --prune_iterations 1900
                python render.py -m "output/$JOB" --skip_train --iteration 3000
                ;;

            4)
                echo "Running Configuration 4 (Deep Blending)..."
                OAR_JOB_ID="$JOB" python train.py -s "$MIPDATASET/$JOB" -i images --eval --ks_scale_power 1.2 --opacity_reset_decay 0.01 --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.2 --batch_size 2 --optimizer_type hybrid --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --save_iterations 3000 --prune_iterations 2100
                python render.py -m "output/$JOB" --skip_train --iteration 3000
                ;;

            *)
                echo "ERROR: Invalid switch ID '$SWITCH_ID'. Please use 1, 2, 3, or 4."
                return 1
                ;;
        esac

        python metrics.py -m "output/$JOB"
    done
}

scenes_360_indoor=(
    bonsai
    counter
    kitchen
    room
)
run_dataset_pipeline "$MIPNERF360_DATASET" 2 "${scenes_360_indoor[@]}"

echo "Running Configuration garden..."
if [ ! -d "$MIPNERF360_DATASET" ]; then
    echo "ERROR: MIPNERF360_DATASET does not exist: $MIPNERF360_DATASET"
    exit 1
fi
OAR_JOB_ID=garden python train.py -s "$MIPNERF360_DATASET/garden" -i images --eval --adam_eps_order 10 --optimizer_type hybrid --split_ratio_threshold 0.4 --sample_bbox_faces --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100 --save_iterations 3000 --prune_iterations 1900
python render.py -m output/garden --skip_train --iteration 3000
python metrics.py -m output/garden

scenes_360_outdoor=(
    bicycle
    stump
    flowers
    treehill
)
run_dataset_pipeline "$MIPNERF360_DATASET" 3 "${scenes_360_outdoor[@]}"

scenes_tandt_db=(
    drjohnson
    playroom
)
run_dataset_pipeline "$TANDT_DB_DATASET" 4 "${scenes_tandt_db[@]}"

scenes_tandt=(
    train
    truck
)
run_dataset_pipeline "$TANDT_DATASET" 1 "${scenes_tandt[@]}"
