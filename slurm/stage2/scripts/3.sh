#!/bin/bash
#SBATCH --job-name=extract_ads
#SBATCH -A kcn@v100
#SBATCH -C v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/kcn/ucm72yx/slurm/%j.out
#SBATCH --error=/lustre/fswork/projects/rech/kcn/ucm72yx/slurm/%j.err

module load ffmpeg/6.1.1
module load pytorch-gpu/py3/2.2.0
source /lustre/fsn1/projects/rech/kcn/ucm72yx/virtual_envs/autoad_zero/bin/activate
cd /lustre/fswork/projects/rech/kcn/ucm72yx/code/AutoAD-Zero/
python stage2/main.py --dataset sfdad --pred_path results/sfdad_no_bboxes/sfdad_ads/stage1_-none-0_it3.csv --access_token tmp
