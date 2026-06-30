#!/bin/bash
#SBATCH --job-name=comp_emb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=0-4
#SBATCH --account=torch_pr_529_general
#SBATCH --array=0-12
#SBATCH -o sweep_agent_%a.log

source ~/.bashrc
export TORCH_HOME="~/.cache/torch_home"
export DATA_PATH="/scratch/at4219/computed_embeddings"
export SWEEP_SAVE_DIR="/tmp/sweep_results"  # don't care much for these

# Fill in sweep id here:
sweepid=""

singularity exec --nv --fakeroot \
    --overlay /scratch/at4219/disentanglement_project.ext3:ro \
    $CUDA_IMAGE \
    /bin/bash -c "source /ext3/audio-disentanglement/.venv/bin/activate; wandb agent $sweepid"
