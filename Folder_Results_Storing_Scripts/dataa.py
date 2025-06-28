import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from new_uniref_dataset import UniRefDataset
import torch

def collate_fn(batch):
    current_embeddings, next_embeddings, metadata, sequences, go_ids, go_terms, additional_metadata = zip(*batch)
    return (torch.stack(current_embeddings), torch.stack(next_embeddings), 
            list(metadata), list(sequences), list(go_ids), list(go_terms), 
            list(additional_metadata))

class dmod(pl.LightningDataModule):
    def __init__(self, uniref_file, esm_model, alphabet, device, esm_layer, max_seq_len, 
                 batch_size=512, seed_only=False,
                 max_samples=50000000, num_workers=0, random_seed=42, human_filter=1,
                 return_difference=False):
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
        self.random_seed = random_seed
        self.human_filter = human_filter
        self.return_difference = return_difference
        
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
            self.max_samples, random_seed=self.random_seed,
            human_filter=self.human_filter, return_difference=self.return_difference
        )
        
        # Check if dataset is empty
        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty! Please check the uniref_file path: {self.uniref_file}")
        
        # Log dataset size
        print(f"Total dataset size: {len(dataset)}")

        # Assign all data to train_dataset
        self.train_dataset = dataset
        self.test_dataset = None  # Set test_dataset to None

        # Log sizes
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: 0")

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        print(f"\nTrain DataLoader:")
        print(f"Dataset size: {len(self.train_dataset)}")
        print(f"Batch size: {self.batch_size}")
        return loader

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None        

