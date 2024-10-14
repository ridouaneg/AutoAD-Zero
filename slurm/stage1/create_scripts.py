# Python script to generate 20 SLURM bash scripts

base_script = """#!/bin/bash
#SBATCH --job-name=extract_ads_{i}
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
python stage1/main.py --dataset sfdad --video_dir $WORK/data/SFD/videos --anno_path resources/annotations/sfdad_anno.csv --charbank_path resources/charbanks/sfdad_charbank_empty.json --model_path $WORK/weights/VideoLLaMA2-7B/ --output_dir results/sfdad_no_bboxes --label_type none --it {i}
"""

import os

output_dir = "scripts"
os.makedirs(output_dir, exist_ok=True)

for i in range(1, 21):
    script_content = base_script.format(i=i)
    filename = os.path.join(output_dir, f"{i}.sh")
    with open(filename, 'w') as f:
        f.write(script_content)

    print(f"Created {filename}")
