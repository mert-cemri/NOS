import os
import sys
import pprint
from pathlib import Path
import hydra
import torch
import wandb
import warnings
from omegaconf import OmegaConf

import seq_models.metrics as metrics
from scripts.utils import flatten_config

from seq_models.trainer import sample_model

#print("imports done")

@hydra.main(config_path="../configs", config_name="sample")
def main(config):
    #print("we are here")
    Path(config.exp_dir).mkdir(parents=True, exist_ok=True)
    #print("test 2")
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep='/')
    log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
    #wandb.init(
    #     project="guided_protein_seq",
    #     config=log_config,
    #)

    pprint.pprint(dict(config))
    print(1)
    model = hydra.utils.instantiate(config.model, _recursive_ = False)
    print(2)    
    if config.ckpt_path is not None:
        state_dict = torch.load(config.ckpt_path)['state_dict']
        model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    results = sample_model(
        model,
        num_samples=config.num_samples,
        infill_seed_file=config.infill_seeds_fn,
        vocab_file=config.vocab_file,
        gt_data_file=os.path.join(config.data_dir, config.val_fn),
        guidance_kwargs=config.guidance_kwargs,
        dps_enable = config.dps_enable
    )
    #print(results)
    pprint.pprint(results[0]['guided_unconditional']['biopython']['samp'])
    import pickle 
    with open(f'/home/cemri/NOS/plot_results/dps{config.dps_enable}_{config.guidance_kwargs.step_size}_{config.guidance_kwargs.stability_coef}_{config.guidance_kwargs.num_steps}.pkl', 'wb') as f:  # open a text file
        pickle.dump(results[0]['guided_unconditional']['biopython']['samp'], f) # serialize the list

    #wandb.log(results)

if __name__ == "__main__":    
    main()
    sys.exit()
