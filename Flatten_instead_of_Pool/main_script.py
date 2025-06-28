##FINAL REPRODUCIBLE
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import esm
import torch
from data_module_script import dmod
from sparse_auto_script import LitLit
import argparse
import multiprocessing
from datetime import datetime
import os
import random

import numpy as np

def set_all_seeds(seed):
    """Set all seeds to make run reproducible."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)

class CheckpointLoggingCallback(pl.Callback):
    def __init__(self, wandb_logger):
        super().__init__()
        self.wandb_logger = wandb_logger

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.checkpoint_callback.best_model_path:
            self.wandb_logger.experiment.config.update({
                "best_checkpoint_path": trainer.checkpoint_callback.best_model_path
            })

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train SAE")
    parser.add_argument("--batch_size", type=int, default=512, )
    parser.add_argument("--learning_rate", type=float, default=0.001, )
    parser.add_argument("--k", type=int, default=10, )
    parser.add_argument("--hidden_dim", type=int, default=0, )
    parser.add_argument("--epochs", type=int, default=10, )
    parser.add_argument("--cuda_device", type=str, default='0', help="CUDA device to use (e.g., '0', '1', '2', '3')")
    parser.add_argument("--encoder_decoder_init", type=int, default=1, choices=[0, 1], help="Initialize encoder as transpose of decoder (1) or use random initialization (0)")
    parser.add_argument("--uniref_file", type=str, required=True, )
    parser.add_argument("--esm_layer", type=int, default=-1, help="(-1 for last layer)")
    parser.add_argument("--max_seq_len", type=int, default=1024, )
    parser.add_argument("--esm_model", type=str, default="esm2_t6_8M_UR50D", 
                        choices=["esm2_t33_650M_UR50D", "esm2_t6_8M_UR50D", 
                                 "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D"],
                        )
    parser.add_argument("--seed_only", type=int, default=0, choices=[0, 1], 
                        help="Use only seed sequences (1) or allow all sequences (0)")
    parser.add_argument("--max_samples", type=int, default=50000000, 
                       )
    parser.add_argument("--inactive_threshold", type=int, default=50,
                       help="Threshold batches")
    parser.add_argument("--val_check_interval", type=int, default=300, 
                        )
    parser.add_argument("--limit_val_batches", type=int, default=40,
                        )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_steps", type=int, default=None, 
                       help="Maximum number of training steps (batches) to run. None for full epochs.")
    parser.add_argument("--aux_alpha", type=float, default=0.03125, 
                        help="Scaling the auxiliary loss (default: 0.03125)")
    parser.add_argument("--wandb_api_key", type=str, required=True, help="Wandb API key")

    return parser.parse_args()
    


def main():
    # ###If Wandb produces timeout issues, you may need to uncomment this out
    # os.environ["WANDB__SERVICE_WAIT"] = "300" 

    # Add this at the start of main()
    if torch.cuda.is_available():
        multiprocessing.set_start_method('spawn', force=True)
    args = parse_arguments()

    set_all_seeds(args.seed)
    
    # Setup device
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    
    wandb.login(key=args.wandb_api_key)
    
    # Initialize wandb logger with a custom run name
    random_number = random.randint(1000, 9999)  # Generate a 4-digit random number
    run_name = f"AA_Flattened{args.esm_layer}_rand{random_number}"
    wandb_logger = WandbLogger(project="ESM-sparse-autoencoder", entity="onkarproject", name=run_name)
    
    model_dict = {
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D
    }
    esm_model, alphabet = model_dict[args.esm_model]()
    esm_model = esm_model.to(device)
    esm_model.eval()

    # Initialize data module
    data_module = dmod(
        args.uniref_file, esm_model, alphabet, device, args.esm_layer,
        args.max_seq_len, args.batch_size, args.seed_only, args.max_samples
    )

    # Initialize model
    model = LitLit(
        input_dim=esm_model.embed_dim,
        hidden_dim=args.hidden_dim,
        k=args.k,
        encoder_decoder_init=args.encoder_decoder_init,
        learning_rate=args.learning_rate,
        inactive_threshold=args.inactive_threshold,
        aux_alpha=args.aux_alpha
    )

    # Get start time for the run
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='Topk_weights',
        filename=f'Regular_esmLayer{args.esm_layer}_MeanPooled_sae_{start_time}_esm{args.esm_model.split("_")[1]}_k{args.k}_hd{args.hidden_dim}_lr{args.learning_rate}_ep{args.epochs}',
        mode='min',
        save_top_k=1
    )
    
    checkpoint_logging_callback = CheckpointLoggingCallback(wandb_logger)

    # Initialize trainer with both callbacks
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, checkpoint_logging_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[int(args.cuda_device)] if torch.cuda.is_available() else None,
        log_every_n_steps=1,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches
    )

    # Train model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()