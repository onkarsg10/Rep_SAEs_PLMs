# Standard library imports
import torch  # PyTorch deep learning framework
import argparse  # For parsing command line arguments
from tqdm import tqdm  # For progress bars
import esm  # Facebook's ESM protein language model
import numpy as np  # For numerical operations
from collections import defaultdict  # For flexible dictionary initialization
import h5py  # For handling HDF5 file format
from datetime import datetime  # For timestamping output files
import os  # For file and directory operations
from pathlib import Path  # For path manipulations
import random  # For random number generation
import pandas as pd  # Add this import for CSV handling

# Custom module imports
from sparse_auto_script import LitLit 
from dataa import dmod  # Custom data module

def parse_args():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Extract and store sequence information and representations")
    
    # Required arguments
    parser.add_argument("--ckpt_file", type=str, required=True, 
                        )
    parser.add_argument("--uniref_file", type=str, required=True,
                        )
    
    # Optional arguments with defaults
    parser.add_argument("--batch_size", type=int, default=32,
                        )
    parser.add_argument("--cuda_device", type=str, default='0',
                        help="CUDA device to use")
    parser.add_argument("--esm_model", type=str, default="esm2_t6_8M_UR50D",
                        choices=["esm2_t33_650M_UR50D", "esm2_t6_8M_UR50D",
                                "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D"])
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        )
    parser.add_argument("--esm_layer", type=int, default=-1,
                        help="(-1 for last layer)")
    parser.add_argument("--max_samples", type=int, default=5000000,
                        )
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--separate", type=int, default=0, choices=[0, 1],
                        help="Whether to separate output into multiple files (0: single file, 1: separate files)")
    parser.add_argument("--human_filter", type=int, default=1, choices=[0, 1],
                        help="Whether to filter for human sequences only (1: human only, 0: all organisms)")
    parser.add_argument("--output_dir", type=str, default="Extracted_Data",
                        )
    return parser.parse_args()

def format_go_terms(terms):
    """Helper function to ensure consistent formatting of GO terms."""
    if isinstance(terms, (list, np.ndarray)):
        return [term.decode('utf-8').strip() if isinstance(term, bytes) else str(term).strip() for term in terms]
    return []

def main():
    """Main function to process and store protein sequence data."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    
    # Add comprehensive deterministic settings
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for deterministic operations
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.set_float32_matmul_precision('highest')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Modify output filename generation to handle separate files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(args.ckpt_file).stem
    if args.separate:
        output_filename_embeddings = os.path.join(args.output_dir, f"POOLING_embeddings_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}.npy")
        output_filename_activations = os.path.join(args.output_dir, f"POOLING_activations_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}.npy")
        output_filename_indices = os.path.join(args.output_dir, f"POOLING_indices_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}.npy")
        output_filename_metadata = os.path.join(args.output_dir, f"POOLING_metadata_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}.csv")
    else:
        output_filename = os.path.join(args.output_dir, f"POOLING_data_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}.h5")
    
    # Set up device (CPU/GPU) for computation
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    

    model_dict = {
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D
    }
    esm_model, alphabet = model_dict[args.esm_model]()
    esm_model = esm_model.to(device)  # Move model to device
    esm_model.eval()  # Set model to evaluation mode

    # Load model from checkpoint
    model = LitLit.load_from_checkpoint(args.ckpt_file)
    model = model.to(device)  # Move model to device
    model.eval()  # Set model to evaluation mode

    # Initialize data module with all parameters
    data_module = dmod(
        args.uniref_file, esm_model, alphabet, device, args.esm_layer,
        args.max_seq_len, batch_size=args.batch_size, seed_only=False, max_samples=args.max_samples, num_workers=0,
        random_seed=args.random_seed, human_filter=args.human_filter, return_difference=True
    )

    # Initialize files dictionary only for non-separate case
    files = {}
    if not args.separate:
        output_filename = os.path.join(args.output_dir, f"POOLING_data_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}.h5")
        files['all'] = h5py.File(output_filename, 'w')
        sequences_group = files['all'].create_group('sequences')
        metadata_group = files['all'].create_group('metadata')
        metadata_group.attrs['timestamp'] = timestamp
        metadata_group.attrs['checkpoint'] = checkpoint_name
        metadata_group.attrs['esm_model'] = args.esm_model
        metadata_group.attrs['max_samples'] = args.max_samples

    try:
        if args.separate:
            # Initialize lists to store data
            all_embeddings = []
            all_activations = []
            all_indices = []
            metadata_records = []
            
            # Process sequences batch by batch
            print("Processing sequences...")
            sample_idx = 0

            with torch.no_grad():
                for batch in tqdm(data_module.train_dataloader(), desc="Processing batches"):
                    embeddings, next_embeddings, metadata, sequences, go_ids, go_terms, additional_metadata = batch
                    
                    embeddings = embeddings.to(device)

                    ln, mean, std = model.model.feature_normalizer.forward(embeddings)
                    shifted_input = ln - model.model.weights[3]
                    
                    mid_before_topk = torch.matmul(shifted_input, model.model.weights[0]) + model.model.weights[2]
                    
                    # Get top-k and their indices
                    sparse_values, sparse_indices = model.model.helper_for_extraction(mid_before_topk)

                    # Convert to numpy and store
                    embeddings = embeddings.cpu().numpy()
                    sparse_values = sparse_values.cpu().numpy()
                    sparse_indices = sparse_indices.cpu().numpy()
                    
                    all_embeddings.append(embeddings)
                    all_activations.append(sparse_values)
                    all_indices.append(sparse_indices)
                    
                    # Process metadata for each sequence
                    for idx in range(len(sequences)):
                        add_meta = additional_metadata[idx]
                        metadata_dict = {
                            'sample_idx': sample_idx,
                            'sequence_id': metadata[idx],
                            'sequence': sequences[idx],
                            'sequence_length': len(sequences[idx]),
                            'go_ids': ','.join(go_ids[idx]),
                            'go_terms': ','.join(go_terms[idx]),
                            'length': str(add_meta['length']).strip(),
                            'go_biological': ','.join(format_go_terms(add_meta['go_biological'])),
                            'go_cellular': ','.join(format_go_terms(add_meta['go_cellular'])),
                            'go_molecular': ','.join(format_go_terms(add_meta['go_molecular'])),
                            'entry_name': str(add_meta['entry_name']).strip(),
                            'protein_names': str(add_meta['protein_names']).strip(),
                            'gene_names': str(add_meta['gene_names']).strip(),
                            'organism': str(add_meta['organism']).strip(),
                            'protein_families': str(add_meta['protein_families']).strip() if add_meta['protein_families'] is not None else "",
                        }
                        metadata_records.append(metadata_dict)
                        sample_idx += 1
            
            # Convert lists to numpy arrays and save
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            final_activations = np.concatenate(all_activations, axis=0)
            final_indices = np.concatenate(all_indices, axis=0)
            
            np.save(output_filename_embeddings, final_embeddings)
            np.save(output_filename_activations, final_activations)
            np.save(output_filename_indices, final_indices)
            
            # Save metadata as CSV
            pd.DataFrame(metadata_records).to_csv(output_filename_metadata, index=False)
            
            print(f"\nProcessed and stored {sample_idx} sequences:")
            print(f"Embeddings saved to: {output_filename_embeddings}")
            print(f"Activations saved to: {output_filename_activations}")
            print(f"Indices saved to: {output_filename_indices}")
            print(f"Metadata saved to: {output_filename_metadata}")
        else:
            # Original single-file storage logic
            # ... (keep existing HDF5 file storage code here)
            pass

    finally:
        # Close files only if using HDF5
        if not args.separate:
            for f in files.values():
                f.close()

# Entry point of the script
if __name__ == "__main__":
    main()