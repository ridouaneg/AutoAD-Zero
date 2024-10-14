#!/bin/bash
#SBATCH --job-name=extract_ads
#SBATCH --array=0-39
#SBATCH -A kcn@v100
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/kcn/ucm72yx/slurm/extract_ads_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/kcn/ucm72yx/slurm/extract_ads_%A_%a.err

SAMPLES_PER_JOB=8192

module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.2.0
source $SCRATCH/virtual_envs/autoad_zero/bin/activate
cd $WORK/code/AutoAD-Zero/

python stage1/main.py \
    --dataset sfdad \
    --video_dir $WORK/data/SFD/videos \
    --anno_path resources/annotations/sfdad_anno.csv \
    --charbank_path resources/charbanks/sfdad_charbank_empty.json \
    --model_path $WORK/weights/VideoLLaMA2-7B/ \
    --output_dir results/sfdad_videollama2-7b \
    --label_type none \
    --iteration $SLURM_ARRAY_TASK_ID \
    --samples_per_job $SAMPLES_PER_JOB
