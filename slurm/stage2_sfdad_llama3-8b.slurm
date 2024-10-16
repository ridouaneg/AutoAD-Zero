#!/bin/bash
#SBATCH --job-name=extract_ads_stage2
#SBATCH --array=0-19
#SBATCH -A kcn@v100
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/kcn/ucm72yx/slurm/extract_ads_stage2_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/kcn/ucm72yx/slurm/extract_ads_stage2_%A_%a.err

module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.2.0
source $SCRATCH/virtual_envs/autoad_zero/bin/activate
cd $WORK/code/AutoAD-Zero/

python stage2/main.py \
    --dataset sfdad \
    --model_path $DSDIR/HuggingFace_Models/meta-llama/Meta-Llama-3-8B-Instruct \
    --pred_path results/sfdad_full_videollama2-7b/sfdad_ads/stage1_-none-0_it$SLURM_ARRAY_TASK_ID.csv
