##FINAL REPRODUCIBLE
import torch
from torch.utils.data import Dataset
import gzip
from Bio import SeqIO
from datetime import datetime
import sys
import random
import pandas as pd

class UniRefDataset(Dataset):
    def __init__(self, uniref_file, esm_model, alphabet, device, esm_layer, max_seq_len, seed_only=False, max_samples=50000000):

        # Set random seeds for reproducibility
        random_seed = 42  # Hardcoded random seed
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.esm_model = esm_model
        self.alphabet = alphabet
        self.device = device
        self.max_seq_len = max_seq_len
        self.esm_layer = esm_layer
        self.sequences = []
        self.metadata = []
        self.seed_only = seed_only
        self.max_samples = max_samples
        self.batch_converter = alphabet.get_batch_converter()

        print(f"Loading sequences from {uniref_file}...")
        print(f"Max sequence length: {max_seq_len}")
        print(f"Seed only: {seed_only}")
        print(f"Max samples to load: {max_samples:,}")
        sys.stdout.flush()

        total_sequences = 0
        loaded_sequences = 0
        identical_sequences = 0
        too_long_sequences = 0
        non_seed_sequences = 0
        last_print_time = datetime.now()

        # Hardcoded SwissProt file path
        # swissprot_file = 'uniprotkb_reviewed_true_2024_10_23.tsv' ###The older swissprot file 
        # swissprot_file = 'uniprotkb_AND_reviewed_true_2024_11_12.tsv' ##The newer swissprot 
        swissprot_file = 'swissprot.tsv'
        print(f"Loading SwissProt sequences from {swissprot_file}...")
        swissprot_data = pd.read_csv(swissprot_file, sep='\t')
        self.swissprot_sequences = set(swissprot_data['Sequence'].values)
        print(f"Loaded {len(self.swissprot_sequences):,} SwissProt sequences")
        sys.stdout.flush()

        def is_seed_sequence(header):
            parts = header.split()
            if len(parts) < 2:
                return False
            
            id_part = parts[0].split('_', 1)
            if len(id_part) != 2:
                return False
            
            uniprot_id = id_part[1]
            
            for part in parts:
                if part.startswith("RepID="):
                    rep_id = part.split("=")[1]
                    return uniprot_id == rep_id
            
            return False

        try:
            with gzip.open(uniref_file, "rt") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    total_sequences += 1
                    seq_str = str(record.seq)
                    
                    # Track sequences that are too long
                    if len(seq_str) > max_seq_len:
                        too_long_sequences += 1
                        continue

                    # Track non-seed sequences when seed_only is True
                    if seed_only and not is_seed_sequence(record.description):
                        non_seed_sequences += 1
                        continue
                    
                    if seq_str in self.swissprot_sequences:
                        identical_sequences += 1
                    else:
                        self.sequences.append(seq_str)
                        self.metadata.append(record.description)
                        loaded_sequences += 1

                    # Updated progress printing with complete breakdown
                    current_time = datetime.now()
                    if (current_time - last_print_time).total_seconds() >= 10 or loaded_sequences % 100000 == 0:
                        computed_sum = loaded_sequences + identical_sequences + too_long_sequences + non_seed_sequences
                        breakdown = (
                            f"\rProcessed: {total_sequences:,} "
                            f"[Loaded: {loaded_sequences:,} + "
                            f"Identical: {identical_sequences:,} + "
                            f"Too long: {too_long_sequences:,}"
                        )
                        if seed_only:
                            breakdown += f" + Non-seed: {non_seed_sequences:,}"
                        breakdown += f" = {computed_sum:,}]"
                        
                        # Add verification
                        if computed_sum != total_sequences:
                            breakdown += f" WARNING: Sum mismatch! Diff: {total_sequences - computed_sum:,}"
                        
                        print(breakdown, end="")
                        sys.stdout.flush()
                        last_print_time = current_time
                    
                    if loaded_sequences >= self.max_samples:
                        print(f"\nReached the maximum number of sequences ({loaded_sequences:,}). Stopping sequence loading.")
                        break

        except Exception as e:
            print(f"Error loading UniRef dataset: {str(e)}")
            raise

        print(f"\n\nDataset loading complete.")
        print(f"Total sequences processed: {total_sequences:,}")
        print(f"Sequences too long: {too_long_sequences:,}")
        if seed_only:
            print(f"Non-seed sequences skipped: {non_seed_sequences:,}")
        print(f"Unique sequences loaded: {loaded_sequences:,}")
        print(f"Identical sequences found: {identical_sequences:,}")
        print(f"Identical sequence percentage: {(identical_sequences/total_sequences)*100:.2f}%")
        sys.stdout.flush()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        metadata = self.metadata[idx]
        
        # Prepare data in the format expected by batch_converter
        data = [(metadata, seq)]
        
        # Convert using batch_converter
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
        # Move tokens to device and compute embeddings
        with torch.no_grad():
            results = self.esm_model(
                batch_tokens.to(self.device), 
                repr_layers=[self.esm_layer], 
                return_contacts=False
            )
        
        # Print full embedding shape (including bos and eos tokens)
        # full_embedding = results["representations"][self.esm_layer][0]  # Shape: [L+2, D]
        # print(f"Full embedding shape (with bos/eos): {full_embedding.shape}")
        
        # Extract embeddings but no bos and eos
        embedding = results["representations"][self.esm_layer][0, 1:-1]  # Shape: [L, D]
        
        # # Print sequence length and embedding shape
        # print(f"Original sequence length: {len(seq)}")
        # print(f"Embedding shape before mean pooling: {embedding.shape}")
        
        # Mean pool across sequence length
        embedding = torch.mean(embedding, dim=0)  # Shape: [D]
        
        return embedding, metadata, seq

