### "step_size": [0.1, 0.5, 1.0], "stability_coef": [0.001, 0.01, 0.1, 1.0, 10.0],

## test mlm for ss_perc_sheet
#step_size = 0.1
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.1 \
    +guidance_kwargs.stability_coef=0.001 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.1 \
    +guidance_kwargs.stability_coef=0.01 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.1 \
    +guidance_kwargs.stability_coef=0.1 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.1 \
    +guidance_kwargs.stability_coef=1.0 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.1 \
    +guidance_kwargs.stability_coef=10.0 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet

#step_size = 0.5
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.5 \
    +guidance_kwargs.stability_coef=0.001 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.5 \
    +guidance_kwargs.stability_coef=0.01 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.5 \
    +guidance_kwargs.stability_coef=0.1 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.5 \
    +guidance_kwargs.stability_coef=1.0 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.5 \
    +guidance_kwargs.stability_coef=10.0 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet

#step_size = 1.0
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=1.0 \
    +guidance_kwargs.stability_coef=0.001 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=1.0 \
    +guidance_kwargs.stability_coef=0.01 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=1.0 \
    +guidance_kwargs.stability_coef=0.1 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=1.0 \
    +guidance_kwargs.stability_coef=1.0 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet
PYTHONPATH="." python scripts/sample.py \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_vanilla/mlm/models/best_by_valid/epoch-49-step-56900.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=1.0 \
    +guidance_kwargs.stability_coef=10.0 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet

## test mlm for ss_perc_sheet (DPS)
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.1 \
    +guidance_kwargs.stability_coef=0.001 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.1 \
    +guidance_kwargs.stability_coef=0.01 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.1 \
    +guidance_kwargs.stability_coef=0.1 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.1 \
    +guidance_kwargs.stability_coef=1.0 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.1 \
    +guidance_kwargs.stability_coef=10.0 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps

PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.5 \
    +guidance_kwargs.stability_coef=0.001 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.5 \
    +guidance_kwargs.stability_coef=0.01 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.5 \
    +guidance_kwargs.stability_coef=0.1 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.5 \
    +guidance_kwargs.stability_coef=1.0 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=0.5 \
    +guidance_kwargs.stability_coef=10.0 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps

PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=1.0 \
    +guidance_kwargs.stability_coef=0.001 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=1.0 \
    +guidance_kwargs.stability_coef=0.01 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=1.0 \
    +guidance_kwargs.stability_coef=0.1 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=1.0 \
    +guidance_kwargs.stability_coef=1.0 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/home/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps/mlm/models/best_by_valid/epoch-49-step-7150.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    +guidance_kwargs.step_size=1.0 \
    +guidance_kwargs.stability_coef=10.0 \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_ss_perc_sheet_dps
