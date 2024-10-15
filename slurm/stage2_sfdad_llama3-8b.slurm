#!/bin/bash
#SBATCH --job-name=extract_ads
#SBATCH --array=0-19
#SBATCH -A kcn@v100
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/kcn/ucm72yx/slurm/extract_ads_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/kcn/ucm72yx/slurm/extract_ads_%A_%a.err

VIDEOS_PER_JOB=500

#module load arch/a100
module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.2.0
source $SCRATCH/virtual_envs/autoad_zero/bin/activate
cd $WORK/code/AutoAD-Zero/

python stage2/main.py \
    --dataset sfdad \
    --model_path $DSDIR/HuggingFace_Models/meta-llama/Meta-Llama-3-8B-Instruct \
    --pred_dir results/sfdad_full_videollama2-7b \
    --iteration $SLURM_ARRAY_TASK_ID
