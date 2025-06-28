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
                       )
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

def create_and_mean_pool_sparse_embeddings(sparse_values, sparse_indices, num_neurons=20000):
    """
    Create full sparse representations and perform mean pooling over sequence length.
    
    Args:
        sparse_values: numpy array of shape (seq_len, top_k) containing activation values
        sparse_indices: numpy array of shape (seq_len, top_k) containing indices
        num_neurons: total number of neurons in sparse layer (default: 20000)
    
    Returns:
        mean_pooled_sparse: numpy array of shape (num_neurons,) containing mean pooled values
    """
    seq_len = sparse_values.shape[0]
    
    # Initialize full sparse representation matrix
    full_sparse_matrix = np.zeros((seq_len, num_neurons))
    
    # Fill in non-zero activations at their corresponding indices for each position
    for pos in range(seq_len):
        full_sparse_matrix[pos, sparse_indices[pos]] = sparse_values[pos]
    
    # Perform mean pooling over sequence length
    mean_pooled_sparse = np.mean(full_sparse_matrix, axis=0)  # shape: (num_neurons,)
    
    return mean_pooled_sparse

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
        output_filename_embeddings = os.path.join(args.output_dir, f"MEANPOOL_embeddings_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}.npy")
        output_filename_activations = os.path.join(args.output_dir, f"MEANPOOL_activations_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}.npy")
        output_filename_indices = os.path.join(args.output_dir, f"MEANPOOL_indices_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}.npy")
        output_filename_metadata = os.path.join(args.output_dir, f"MEANPOOL_metadata_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}.csv")
    else:
        output_filename = os.path.join(args.output_dir, f"MEANPOOL_data_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}.h5")
    
    # Set up device (CPU/GPU) for computation
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    

    # Dictionary mapping model names to their corresponding functions
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
        output_filename = os.path.join(args.output_dir, f"MEANPOOL_data_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}.h5")
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
            all_mean_pooled_embeddings = []
            all_mean_pooled_sparse = []
            
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
                    
                    # Get actual sequence lengths (non-padded)
                    seq_lengths = [len(seq) for seq in sequences]
                    
                    # Process each sequence in the batch individually
                    for i in range(len(sequences)):
                        print("\n===============================================")
                        print(f"Sequence {i} in batch:")
                        print("===============================================")
                        print(f"Original sequence length: {seq_lengths[i]}")
                        print(f"Full embeddings shape: {embeddings[i].shape}")
                        print(f"Full sparse_values shape: {sparse_values[i].shape}")
                        print(f"Full sparse_indices shape: {sparse_indices[i].shape}")
                        
                        # Extract non-padded part for current sequence
                        seq_len = seq_lengths[i]
                        curr_embeddings = embeddings[i, :seq_len].cpu().numpy()
                        curr_sparse_values = sparse_values[i, :seq_len].cpu().numpy()
                        curr_sparse_indices = sparse_indices[i, :seq_len].cpu().numpy()
                        
                        # Create and pool sparse embeddings
                        mean_pooled_sparse = create_and_mean_pool_sparse_embeddings(
                            curr_sparse_values, 
                            curr_sparse_indices
                        )
                        
                        print(f"Mean pooled sparse embeddings shape: {mean_pooled_sparse.shape}")
                        
                        # Perform mean pooling over sequence length
                        # curr_embeddings shape: (seq_len, embedding_dim)
                        mean_pooled_embeddings = np.mean(curr_embeddings, axis=0)  # shape: (embedding_dim,)
                        
                        print("\nAfter trimming and pooling:")
                        print("-----------------------------------------------")
                        print(f"Trimmed embeddings shape: {curr_embeddings.shape}")
                        print(f"Mean pooled embeddings shape: {mean_pooled_embeddings.shape}")
                        print(f"Trimmed sparse_values shape: {curr_sparse_values.shape}")
                        print(f"Trimmed sparse_indices shape: {curr_sparse_indices.shape}")
                        print("===============================================")
                        
                        # Store mean pooled vectors for later concatenation
                        all_mean_pooled_embeddings.append(mean_pooled_embeddings)
                        all_mean_pooled_sparse.append(mean_pooled_sparse)
                        
                        # Process metadata
                        add_meta = additional_metadata[i]
                        metadata_dict = {
                            'sample_idx': sample_idx,
                            'sequence_id': metadata[i],
                            'sequence': sequences[i],
                            'sequence_length': seq_len,
                            'go_ids': go_ids[i],
                            'go_terms': go_terms[i],
                            
                            'entry': add_meta['Entry'],
                            'reviewed': add_meta['Reviewed'],
                            'entry_name': add_meta['Entry Name'],
                            'protein_names': add_meta['Protein names'],
                            'gene_names': add_meta['Gene Names'],
                            'organism': add_meta['Organism'],
                            'length': add_meta['Length'],
                            'go_biological': add_meta['Gene Ontology (biological process)'],
                            'go_cellular': add_meta['Gene Ontology (cellular component)'],
                            'go_molecular': add_meta['Gene Ontology (molecular function)'],
                            'protein_families': add_meta['Protein families']
                        }
                        metadata_records.append(metadata_dict)
                        sample_idx += 1
            
            # After processing all sequences, save concatenated vectors
            concatenated_filename_base = f"MEANPOOL_concatenated_time{timestamp}_ckpt{checkpoint_name}_maxsamples{args.max_samples}"
            
            # Convert lists to numpy arrays and save
            all_mean_pooled_embeddings_array = np.stack(all_mean_pooled_embeddings, axis=0)
            all_mean_pooled_sparse_array = np.stack(all_mean_pooled_sparse, axis=0)
            
            np.save(os.path.join(args.output_dir, f"{concatenated_filename_base}_embeddings.npy"), 
                   all_mean_pooled_embeddings_array)
            np.save(os.path.join(args.output_dir, f"{concatenated_filename_base}_sparse.npy"), 
                   all_mean_pooled_sparse_array)
            
            print(f"\nSaved concatenated mean pooled vectors:")
            print(f"Embeddings shape: {all_mean_pooled_embeddings_array.shape}")
            print(f"Sparse shape: {all_mean_pooled_sparse_array.shape}")
            
            # Save metadata as CSV with paths to individual numpy files
            pd.DataFrame(metadata_records).to_csv(output_filename_metadata, index=False)
            
            print(f"\nProcessed and stored {sample_idx} sequences:")
            print(f"Data directory: {args.output_dir}")
            print(f"Metadata file: {output_filename_metadata}")
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