##FINAL REPRODUCIBLE
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from swissprot_filtered_uniref_dataset import UniRefDataset
import torch
import numpy as np
import random

def collate_fn(batch):
    # Unzip the batch into current embeddings, next embeddings, metadata, and sequences
    current_embeddings, next_embeddings, metadata, sequences = zip(*batch)
    
    # Stack embeddings - they're already mean-pooled so just stack them
    stacked_current = torch.stack(current_embeddings)
    stacked_next = torch.stack(next_embeddings)
    
    # Return stacked embeddings, metadata, and sequences
    return stacked_current, stacked_next, list(metadata), list(sequences)

class dmod(pl.LightningDataModule):
    def __init__(self, uniref_file, esm_model, alphabet, device, esm_layer, max_seq_len, 
                 batch_size=512, seed_only=False,
                 max_samples=50000000, num_workers=0, return_difference=False):
        super().__init__()
        self.uniref_file = uniref_file
        self.esm_model = esm_model
        self.alphabet = alphabet
        self.device = device
        self.esm_layer = esm_layer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.seed_only = seed_only
        self.max_samples = max_samples
        self.num_workers = num_workers
        self.return_difference = return_difference
        self.seed = 42  # Hard-coded seed value
        
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Initialize these to None
        self.train_dataset = None
        self.test_dataset = None
        
        # Call setup explicitly during initialization
        self.setup()

    def setup(self, stage=None):
        # Add error checking
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            return  # Setup has already been run
            
        # Create dataset
        dataset = UniRefDataset(
            self.uniref_file, self.esm_model, self.alphabet, self.device,
            self.esm_layer, self.max_seq_len, self.seed_only,
            self.max_samples, self.return_difference
        )
        
        # Check if dataset is empty
        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty! Please check the uniref_file path: {self.uniref_file}")
        
        # Log dataset size
        print(f"Total dataset size: {len(dataset)}")
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        # Verify split sizes
        if train_size == 0 or test_size == 0:
            raise ValueError(f"Invalid split sizes! train_size: {train_size}, test_size: {test_size}")
            
        # Use generator for reproducible splits
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.test_dataset = random_split(
            dataset, 
            [train_size, test_size],
            generator=generator
        )
        
        # Log split sizes
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        # Add generator for reproducible shuffling
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            generator=generator
        )
        print(f"\nTrain DataLoader:")
        print(f"Dataset size: {len(self.train_dataset)}")
        print(f"Batch size: {self.batch_size}")
        return loader

    def val_dataloader(self):
        # Add generator for reproducible shuffling
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            generator=generator        
        )
        print(f"\nVal DataLoader:")
        print(f"Dataset size: {len(self.test_dataset)}")
        print(f"Batch size: {self.batch_size}")
        return loader

    def test_dataloader(self):
        # Add generator for reproducible shuffling
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            generator=generator
        )