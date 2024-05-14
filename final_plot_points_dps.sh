for step_size in 1.0;
do
    for stability_coef in 0.001, 0.01, 0.1, 1.0, 10.0;
    do 

    step_size_str=$(printf "%.5f" $step_size)
    stability_coef_str=$(printf "%.5f" $stability_coef)

    PYTHONPATH="." python scripts/sample.py \
    model._target_=seq_models.model.mlm_diffusion.MLMDiffusion_DPS \
    dps_enable=True \
    model=mlm \
    model.network.target_channels=1 \
    ckpt_path='/data/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_dps_5_ffs/mlm/models/best_by_valid/epoch\=89-step\=127980.ckpt' \
    'target_cols=['ss_perc_sheet']' \
    "+guidance_kwargs.step_size=$step_size_str" \
    "+guidance_kwargs.stability_coef=$stability_coef_str" \
    +guidance_kwargs.num_steps=5  \
    +seeds_fn=/home/cemri/NOS/poas_seeds.csv \
    +results_dir=/home/cemri/NOS/scratch/sampling_results/guided_protein_mlm_perc_sheet; 
    done; 
done
#     ckpt_path='/data/cemri/NOS/scratch/logs/guided_protein_mlm_aromaticity_DPS_5_ffs/mlm/models/best_by_valid/epoch\=54-step\=625570.ckpt' \
#     ckpt_path='/data/cemri/NOS/scratch/logs/guided_infill_protein_mlm_ss_perc_sheet_dps_5_ffs/mlm/models/best_by_valid/epoch\=40-step\=466334.ckpt'

#     ckpt_path='/data/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_helix_dps_5_ffs/mlm/models/best_by_valid/epoch\=77-step\=887172.ckpt' \
#     ckpt_path='/data/cemri/NOS/scratch/logs/guided_protein_mlm_ss_perc_sheet_and_helix_dps_5_ffs/mlm/models/best_by_valid/epoch\=70-step\=807554.ckpt' \