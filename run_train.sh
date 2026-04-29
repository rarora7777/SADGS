

export HF_HOME=/HPS/VisibilityLearning/work/cache/huggingface
export TORCH_HOME=/HPS/VisibilityLearning/work/cache/torch



mamba activate structgs
echo $CONDA_DEFAULT_ENV
export PROJECT=/HPS/VisibilityLearning/work/Programs/FreqGS
cd $PROJECT




run_dataset_pipeline() {
    # 1. Argument Parsing
    local DATASET_PATH=$1   # First argument: Path
    local SWITCH_ID=$2      # Second argument: The Switch/Mode (1, 2, 3...)
    
    # Check if arguments are sufficient
    if [ -z "$SWITCH_ID" ]; then
        echo "Error: Usage: run_dataset_pipeline <path> <switch_id> <scene1> [scene2...]"
        return 1
    fi

    shift 2                 # Remove Path and Switch, leaving only the scenes
    local SCENE_LIST=("$@") # All remaining arguments are the scenes

    # Export the dataset path so Python can see it
    export MIPDATASET=$DATASET_PATH
	export CUDA_VISIBLE_DEVICES=0

    # Iterate over the specific scenes passed to this function
    for JOB in "${SCENE_LIST[@]}"; do
        echo "=============================="
        echo "Processing scene: $JOB"
        echo "Dataset: $MIPDATASET"
        echo "Mode (Switch): $SWITCH_ID"
		echo "GPU-ID:$CUDA_VISIBLE_DEVICES"
        echo "=============================="

        # Select logic based on the Switch ID
        case $SWITCH_ID in
            
            1)
                
                echo "Running Configuration 1 (tandt_db/db)..."
                OAR_JOB_ID=$JOB python train_curve_test.py -s $MIPDATASET/$JOB  -i images --eval --ks_scale_power 1.2  --highfeature_lr 0.01  --split_ratio_threshold 0.4 --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.4  --optimizer_type hybrid    --sample_bbox_faces --warmup_densification  --mult 0.7 --iterations 7000 --position_lr_max_steps 7000 --densification_interval 500 --save_iterations  7000 --opacity_reset_interval 3000 --densify_from_iter 500 --densify_until_iter 4000 --prune_iterations 4000 #
				# python render.py -m output_curve_sequence/$JOB  --iteration 7000 --skip_train  --mult 0.7 
                ;;

            2)
               
                echo "Running Configuration 2 (mipnerf Indoor)..."
                OAR_JOB_ID=$JOB python train_curve_test.py -s $MIPDATASET/$JOB  -i images --eval --scale_rotation_scheduler --adam_eps_order 10  --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.5 --batch_size 2 --optimizer_type hybrid   --sample_bbox_faces --warmup_densification --mult 0.25  --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100  --save_iterations  3000 --prune_iterations 1900  # --sample_bbox_faces 
				# python render.py -m output_curve_sequence/$JOB --skip_train --iteration 3000 #--render_extra
                ;;

            3)
                
                echo "Running Configuration 3 (mipnerf Outdoor)..."
                OAR_JOB_ID=$JOB python train_curve_test.py -s $MIPDATASET/$JOB  -i images --eval  --optimizer_type hybrid  --split_ratio_threshold 0.4   --sample_bbox_faces --warmup_densification --mult 0.25  --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100  --save_iterations  3000 --prune_iterations 1900  # --sample_bbox_faces 
				# python render.py -m output_curve_sequence/$JOB --skip_train --iteration 3000 #--render_extra
                ;;
			
			
			4)
               
                echo "Running Configuration 4 (DB)..."
                OAR_JOB_ID=$JOB python train_curve_test.py -s $MIPDATASET/$JOB  -i images --eval --ks_scale_power 1.2 --opacity_reset_decay 0.01 --freq_transmittance_threshold 0.8 --freq_opacity_threshold 0.2 --batch_size 2 --optimizer_type hybrid   --sample_bbox_faces --warmup_densification --mult 0.25  --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100  --save_iterations  3000 --prune_iterations 2100  # --sample_bbox_faces 
				# python render.py -m output_curve_sequence/$JOB --skip_train --iteration 3000 #--render_extra
                ;;
				
            *)
                echo "ERROR: Invalid switch ID '$SWITCH_ID'. Please use 1-7."
                ;;
        esac

        # Run metrics for all successful runs
        # python metrics.py -m output_curve_sequence/$JOB
    done
}

# scenes_360=(
    # bonsai
    # counter
    # kitchen
    # room
# )
# run_dataset_pipeline "/HPS/VisibilityLearning/work/Programs/NeuS/data/360_v2" 2 "${scenes_360[@]}"


# echo "Running Configuration garden..."
# export DATASET=/HPS/VisibilityLearning/work/Programs/NeuS/data/360_v2
# OAR_JOB_ID=garden python train_curve_test.py -s $DATASET/garden  -i images --eval --adam_eps_order 10  --optimizer_type hybrid  --split_ratio_threshold 0.4   --sample_bbox_faces --warmup_densification --mult 0.25  --iterations 3000 --position_lr_max_steps 3000 --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900 --densify_from_iter 100  --save_iterations  3000 --prune_iterations 1900  # --sample_bbox_faces 
# python render.py -m output_curve_sequence/garden --skip_train --iteration 3000 #--render_extra
# python metrics.py -m output_curve_sequence/garden
				
# scenes_360=(
    # bicycle
    # stump
    # flowers
    # treehill
# )
# run_dataset_pipeline "/HPS/VisibilityLearning/work/Programs/NeuS/data/360_v2" 3 "${scenes_360[@]}"

# scenes_tandt_db=(
    # drjohnson
    # playroom
# )

# run_dataset_pipeline "/HPS/VisibilityLearning/work/Programs/NeuS/data/tandt_db/db" 4 "${scenes_tandt_db[@]}"


scenes_tandt=(
    train
    truck
)

run_dataset_pipeline "/HPS/VisibilityLearning/work/Programs/NeuS/data/tandt_db/tandt" 1 "${scenes_tandt[@]}"

