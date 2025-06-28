###The neuron random sampling logic is by PRESHUFFLE 


# Core data processing and scientific computing libraries
import numpy as np  # For efficient numerical operations and array handling
import pandas as pd  # For structured data manipulation and CSV file handling
import random  # For random sampling operations
from anthropic import Anthropic  # For interfacing with Claude AI API
from scipy.stats import pearsonr  # For calculating correlation coefficients
import json  # For reading/writing JSON data
from pathlib import Path  # For cross-platform file path handling
import torch  # For PyTorch operations (though not heavily used in this script)
from tqdm import tqdm  # For progress bar visualization
import argparse  # For command-line argument parsing
import sys  # For system-specific parameters and functions
import os  # For operating system dependent functionality
import datetime  # For timestamping results files
 



def find_pooling_files(folder_path):

    try:
        folder = Path(folder_path)
        
        # Verify folder exists
        if not folder.is_dir():
            print(f"Error: Folder '{folder_path}' does not exist")
            return None
            
        # Find files matching each prefix
        embeddings_files = list(folder.glob("MEANPOOL_concatenated*embeddings.npy"))
        sparse_files = list(folder.glob("MEANPOOL_concatenated*sparse.npy"))
        metadata_files = list(folder.glob("MEANPOOL_metadata*"))
        
        # Check if exactly one file was found for each prefix
        for file_type, files in [
            ("embeddings", embeddings_files),
            ("sparse matrix", sparse_files),
            ("metadata", metadata_files)
        ]:
            if len(files) == 0:
                print(f"Error: No {file_type} file found in {folder_path}")
                return None
            if len(files) > 1:
                print(f"Error: Multiple {file_type} files found in {folder_path}:")
                for f in files:
                    print(f"  - {f.name}")
                return None
        
        # Get the paths as strings
        files = (
            str(embeddings_files[0]),
            str(sparse_files[0]),
            str(metadata_files[0])
        )
        
        # Print found files
        print("\nFound MEANPOOL files:")
        print(f"Embeddings: {embeddings_files[0].name}")
        print(f"Sparse matrix: {sparse_files[0].name}")
        print(f"Metadata: {metadata_files[0].name}\n")
        
        return files
        
    except Exception as e:
        print(f"Error while searching for MEANPOOL files: {str(e)}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze neuron interpretations using Claude API')
    
    # Add random seed argument
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    
    # Replace individual file paths with single folder path
    parser.add_argument('--data_folder', type=str, required=True,
                      help='Path to folder containing POOLING files')
    
    # Add specific neuron argument
    parser.add_argument('--specific_neuron', type=int, default=None,
                      help='Analyze a specific neuron instead of random sampling (default: None)')
    
    # Number of neurons to analyze
    parser.add_argument('--num_neurons', type=int, default=200,
                      help='Number of neurons to randomly sample (default: 200)')
    
    # Runtime mode configuration
    parser.add_argument('--offline_mode', action='store_true',
                      help='Run in offline mode (manual copy-paste to Claude)')
    parser.add_argument('--claude_api_key', type=str,
                      help='Claude API key (not required in offline mode)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='Offline_Prompts',
                      help='Directory to store offline prompts and results (default: Offline_Prompts)')
    
    # Analysis mode configuration
    parser.add_argument('--embedding_mode', type=int, default=0,
                      help='0 for sparse activations, 1 for embedding analysis (default: 0)')
    
    parser.add_argument('--high_block_samples', type=int, default=6,
                      help='Number of sequences to sample from highest activation block (0.8-1.0) (default: 6)')
    parser.add_argument('--mid_block_samples', type=int, default=3,
                      help='Number of sequences to sample from each middle block (0.2-0.4, 0.4-0.6, 0.6-0.8) (default: 3)')
    parser.add_argument('--low_block_samples', type=int, default=2,
                      help='Number of sequences to sample from lowest active block (0.0-0.2) (default: 2)')
    parser.add_argument('--inactive_samples', type=int, default=15,
                      help='Number of inactive sequences to sample (default: 15)')
    
    parser.add_argument('--include_sequence', type=int, default=0, choices=[0, 1],
                      help='Whether to include the actual sequence in prompts (0: exclude, 1: include) (default: 1)')
    
    # Add train/test split ratio argument
    parser.add_argument('--train_ratio', type=float, default=0.75,
                      help='Ratio of data to use for training (default: 0.75)')
    
    # Add embedding_relu argument
    parser.add_argument('--embedding_relu', type=int, default=0, choices=[0, 1],
                      help='For embedding mode only: Apply ReLU before normalization-- for inactive samples (0: no, 1: yes) (default: 0)')
    
    # Parse and return arguments
    return parser.parse_args()



class NeuronInterpreter:
    def __init__(self, embeddings_file, sparse_matrix_file, metadata_file, 
                 claude_api_key=None, offline_mode=False, output_dir='Offline_Prompts', embedding_mode=0, include_sequence=1, train_ratio=0.75, embedding_relu=0,
                 specific_neuron=None, num_neurons=200,
                 high_block_samples=6, mid_block_samples=3,
                 low_block_samples=2, inactive_samples=15,
                 random_seed=42):

        # Set random seeds at initialization
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Store all parameters as instance variables
        self.offline_mode = offline_mode
        self.output_dir = output_dir
        self.embedding_mode = embedding_mode
        self.include_sequence = include_sequence
        self.train_ratio = train_ratio
        self.embedding_relu = embedding_relu
        self.specific_neuron = specific_neuron
        self.num_neurons = num_neurons
        self.high_block_samples = high_block_samples
        self.mid_block_samples = mid_block_samples
        self.low_block_samples = low_block_samples
        self.inactive_samples = inactive_samples
        
        # Create timestamp for results file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_str = 'embedding' if embedding_mode else 'sparse'
        run_type = 'offline' if offline_mode else 'online'
        
        # Format: mode_numNeurons_runType_timestamp
        neurons_str = f"n{num_neurons}"  # Simplified format: n200 instead of num_neurons_200
        base_name = f'{mode_str}_{neurons_str}_{run_type}_{timestamp}'
        
        # Save results file in the input data folder instead of output_dir
        data_folder = os.path.dirname(embeddings_file)
        self.results_file = os.path.join(data_folder, f'{base_name}_results.txt')
        
        # Create Prompts directory inside the data folder with the same naming pattern
        self.prompts_dir = os.path.join(data_folder, f'{base_name}_prompts')
        os.makedirs(self.prompts_dir, exist_ok=True)
        
        # Create output directory if it doesn't exist (for offline prompts)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all arguments from argparse
        args = sys.argv[1:]
        
        # Initialize results file with header and arguments
        with open(self.results_file, 'w') as f:
            f.write(f"Neuron Analysis Results\n")
            f.write(f"Run started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {mode_str}\n")
            f.write(f"Run type: {run_type}\n\n")
            
            # Write all arguments
            f.write("Command Line Arguments:\n")
            f.write(f"{'='*50}\n")
            
            # Required file paths
            f.write("\nData Files:\n")
            f.write(f"embeddings_file: {embeddings_file}\n")
            f.write(f"sparse_matrix_file: {sparse_matrix_file}\n")
            f.write(f"metadata_file: {metadata_file}\n")
            
            # Runtime configuration
            f.write("\nRuntime Configuration:\n")
            f.write(f"offline_mode: {offline_mode}\n")
            f.write(f"output_dir: {output_dir}\n")
            if not offline_mode:
                f.write(f"claude_api_key: {'[PROVIDED]' if claude_api_key else '[NOT PROVIDED]'}\n")
            
            # Analysis configuration
            f.write("\nAnalysis Configuration:\n")
            f.write(f"embedding_mode: {embedding_mode}\n")
            f.write(f"embedding_relu: {embedding_relu}\n")
            f.write(f"include_sequence: {include_sequence}\n")
            f.write(f"train_ratio: {train_ratio}\n")
            
            # Neuron selection
            f.write("\nNeuron Selection:\n")
            f.write(f"specific_neuron: {specific_neuron}\n")
            f.write(f"num_neurons: {num_neurons}\n")
            
            # Sampling configuration
            f.write("\nSampling Configuration:\n")
            f.write(f"high_block_samples: {high_block_samples}\n")
            f.write(f"mid_block_samples: {mid_block_samples}\n")
            f.write(f"low_block_samples: {low_block_samples}\n")
            f.write(f"inactive_samples: {inactive_samples}\n")
            
            # Raw command
            f.write(f"\nRaw command:\n")
            f.write(f"python {sys.argv[0]} {' '.join(args)}\n")
            f.write(f"{'='*50}\n\n")
        
        # If not in offline mode, initialize Claude API client
        if not offline_mode:
            # Verify API key is provided
            if claude_api_key is None:
                raise ValueError("Claude API key is required when not in offline mode")
            # Initialize Anthropic client with provided API key
            self.anthropic = Anthropic(api_key=claude_api_key)
            
        # Load all necessary data files
        self.load_data(embeddings_file, sparse_matrix_file, metadata_file)

    def load_data(self, embeddings_file, sparse_matrix_file, metadata_file):




        def zero_out_bottom_percentile(activations, percentile=0):

            # Calculate threshold for each neuron (column)
            thresholds = np.percentile(activations, percentile, axis=0)
            # Create mask for values below threshold
            mask = activations < thresholds[None, :]
            # Set values below threshold to zero
            activations[mask] = 0
            return activations

        # Load embeddings and metadata
        self.embeddings = np.load(embeddings_file)
        self.metadata = pd.read_csv(metadata_file)

        
        # Branch for handling embedding analysis mode
        if hasattr(self, 'embedding_mode') and self.embedding_mode:
            # Print debug information about the embedding processing mode
            print(f"DEBUG: Loading in embedding mode")
            # Display shape of embedding matrix (sequences × dimensions)
            print(f"DEBUG: Embeddings shape: {self.embeddings.shape}")
            self.num_sequences = self.embeddings.shape[0]
            self.num_neurons = self.embeddings.shape[1]
            print(f"DEBUG: Set num_sequences={self.num_sequences}, num_neurons={self.num_neurons}")
            
            # Create a copy of embeddings to avoid modifying original data
            embeddings_processed = self.embeddings.copy()
            # Optionally apply ReLU (set negative values to 0) if enabled
            if hasattr(self, 'embedding_relu') and self.embedding_relu:
                print("DEBUG: Applying ReLU to embeddings before normalization")
                # Replace all negative values with 0 using boolean indexing
                embeddings_processed[embeddings_processed < 0] = 0
                
                print("DEBUG: Zeroing out bottom percentile of activations for each neuron")
                embeddings_processed = zero_out_bottom_percentile(embeddings_processed)
            



            
            # Begin min-max normalization process for embeddings
            # Calculate minimum value for each neuron across all sequences
            embeddings_min = embeddings_processed.min(axis=0, keepdims=True)
            # Calculate maximum value for each neuron across all sequences
            embeddings_max = embeddings_processed.max(axis=0, keepdims=True)
            # Add small epsilon to prevent division by zero during normalization
            eps = 1e-10
            # Calculate denominator for normalization, ensuring it's never zero
            scaling_denominator = np.maximum(embeddings_max - embeddings_min, eps)
            # Apply min-max normalization: (x - min) / (max - min)
            self.full_activation_matrix = (embeddings_processed - embeddings_min) / scaling_denominator
            
        else:  # Sparse mode
            print(f"DEBUG: Loading in sparse mode")
            # Load pre-constructed sparse matrix directly
            self.full_activation_matrix = np.load(sparse_matrix_file)
            print(f"DEBUG: Loaded sparse matrix shape: {self.full_activation_matrix.shape}")
            
            self.num_sequences = self.full_activation_matrix.shape[0]
            self.num_neurons = self.full_activation_matrix.shape[1]
            print(f"DEBUG: Set num_sequences={self.num_sequences}, num_neurons={self.num_neurons}")


            print("DEBUG: Zeroing out bottom percentile of activations for each neuron")
            self.full_activation_matrix = zero_out_bottom_percentile(self.full_activation_matrix)

            
            # Apply min-max normalization to non-zero columns
            activations_min = self.full_activation_matrix.min(axis=0, keepdims=True)
            # Calculate maximum value for each neuron across all sequences
            activations_max = self.full_activation_matrix.max(axis=0, keepdims=True)
            # Add small epsilon to prevent division by zero during normalization
            eps = 1e-10
            # Calculate denominator for normalization, ensuring it's never zero
            scaling_denominator = np.maximum(activations_max - activations_min, eps)
            # Identify columns (neurons) that have any non-zero values
            non_zero_cols = np.any(self.full_activation_matrix != 0, axis=0)
            # Apply min-max normalization only to non-zero columns to avoid numerical issues
            self.full_activation_matrix[:, non_zero_cols] = (
                (self.full_activation_matrix[:, non_zero_cols] - activations_min[:, non_zero_cols]) / 
                scaling_denominator[:, non_zero_cols]
            )






    # Input: neuron_idx (which neuron to analyze), and various sampling parameters for different activation blocks
    # Output: Returns three arrays - sampled active sequences, their activation values, and sampled inactive sequences
    def get_sequences_for_neuron(self, neuron_idx, high_block_samples=6, mid_block_samples=3, 
                           low_block_samples=2, inactive_samples=15):

        # Print a separator line for better readability in debug output
        print("\n======================================================================")
        # Print debug information about which neuron is being processed and in what mode
        print(f"DEBUG [Neuron {neuron_idx}]: Processing in {'embedding' if self.embedding_mode else 'sparse'} mode")
        
        # Extract all activation values for the specified neuron
        neuron_activations = self.full_activation_matrix[:, neuron_idx]
        
        # Define activation ranges for different blocks (0-0.2, 0.2-0.4, etc.)
        blocks = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        # Define how many samples to take from each block
        block_sizes = [low_block_samples, mid_block_samples, mid_block_samples, mid_block_samples, high_block_samples]
        
        # Helper function to get indices of activations falling within a specific range
        # Input: activation values, lower and upper bounds
        # Output: array of indices where activations fall within the bounds
        def get_block_indices(acts, lower, upper):
            return np.where((acts >= lower) & (acts < upper))[0]
        
        # Complex sampling function that can "spill over" to adjacent blocks if needed
        # Input: activation values, current block index, how many samples needed, and already used indices
        # Output: sampled indices, their activation values, and remaining samples needed
        def sample_from_block_with_spillover(acts, block_idx, target_size, used_indices=None):

            # Initialize used_indices set if not provided
            if used_indices is None:
                used_indices = set()
            
            sampled_indices = []
            sampled_values = []
            
            # First try the current block
            current_selected, current_values, target_size = sample_from_block(
                acts, block_idx, target_size, used_indices)
            sampled_indices.extend(current_selected)
            sampled_values.extend(current_values)
            
            # If we still need more samples, try adjacent blocks alternately
            if target_size > 0:
                distance = 1
                max_distance = len(blocks) - 1  # Maximum possible distance to check
                
                while target_size > 0 and distance <= max_distance:
                    # Try lower block if available
                    lower_idx = block_idx - distance
                    if lower_idx >= 0:
                        lower_selected, lower_values, remaining = sample_from_block(
                            acts, lower_idx, target_size, used_indices)
                        sampled_indices.extend(lower_selected)
                        sampled_values.extend(lower_values)
                        target_size = remaining
                    
                    # Try higher block if still needed and available
                    higher_idx = block_idx + distance
                    if target_size > 0 and higher_idx < len(blocks):
                        higher_selected, higher_values, remaining = sample_from_block(
                            acts, higher_idx, target_size, used_indices)
                        sampled_indices.extend(higher_selected)
                        sampled_values.extend(higher_values)
                        target_size = remaining
                    
                    distance += 1  # Look one block further in each direction next time
            
            return sampled_indices, sampled_values, target_size
        
        # Helper function to sample from a single block (to avoid recursion)
        def sample_from_block(acts, block_idx, target_size, used_indices):
            lower, upper = blocks[block_idx]
            block_indices = set(get_block_indices(acts, lower, upper)) - used_indices
            
            sampled_indices = []
            sampled_values = []
            
            if block_indices:
                sample_size = min(len(block_indices), target_size)
                selected = np.random.choice(list(block_indices), sample_size, replace=False)
                sampled_indices.extend(selected)
                sampled_values.extend(acts[selected])
                used_indices.update(selected)
                target_size -= sample_size
            
            return sampled_indices, sampled_values, target_size
        
        # Handle embedding mode (when working with continuous activation values)
        if hasattr(self, 'embedding_mode') and self.embedding_mode:
            # Special handling for ReLU-activated embeddings
            if hasattr(self, 'embedding_relu') and self.embedding_relu:
                # Split into active (>0) and inactive (=0) sequences
                active_indices = np.where(neuron_activations > 0)[0]
                inactive_indices = np.where(neuron_activations == 0)[0]
                
                # Print debug information about found sequences
                print(f"DEBUG [Neuron {neuron_idx}]: Found {len(active_indices)} total active sequences")
                print(f"DEBUG [Neuron {neuron_idx}]: Found {len(inactive_indices)} total inactive sequences")
                
                # Check if we have enough sequences to sample from
                if len(active_indices) < 2*(high_block_samples + (3*mid_block_samples) + low_block_samples) or len(inactive_indices) < inactive_samples:
                    print(f"Warning [Neuron {neuron_idx}]: Insufficient sequences (needs {2*(high_block_samples + (3*mid_block_samples) + low_block_samples)} active samples and {inactive_samples} inactive samples). Skipping...")
                    return None, None, None
                
                # Initialize collection variables
                sampled_active = []
                sampled_active_acts = []
                used_indices = set()
                
                # Find sequences in highest activation block
                high_block_mask = ((neuron_activations[active_indices] >= 0.8) & 
                                  (neuron_activations[active_indices] <= 1.0))
                high_block_count = np.sum(high_block_mask)
                print(f"DEBUG [Neuron {neuron_idx}]: High block (0.8-1.0): {high_block_count} sequences")
                
                # Sample from highest block
                selected, values, _ = sample_from_block_with_spillover(
                    neuron_activations[active_indices], len(blocks)-1, high_block_samples, used_indices)
                print(f"DEBUG [Neuron {neuron_idx}]: Sampled {len(selected)} sequences from high block")
                sampled_active.extend(active_indices[selected])
                sampled_active_acts.extend(values)
                
                # Sample from remaining blocks
                for block_idx in range(len(blocks)-1):  # Skip the last block
                    # Find sequences in current block
                    block_mask = ((neuron_activations[active_indices] >= blocks[block_idx][0]) & 
                                 (neuron_activations[active_indices] < blocks[block_idx][1]))
                    block_count = np.sum(block_mask)
                    print(f"DEBUG [Neuron {neuron_idx}]: Block {block_idx} ({blocks[block_idx][0]:.1f}-{blocks[block_idx][1]:.1f}): {block_count} sequences")
                    
                    # Sample from current block
                    selected, values, _ = sample_from_block_with_spillover(
                        neuron_activations[active_indices], block_idx, block_sizes[block_idx], used_indices)
                    print(f"DEBUG [Neuron {neuron_idx}]: Sampled {len(selected)} sequences from block {block_idx}")
                    
                    sampled_active.extend(active_indices[selected])
                    sampled_active_acts.extend(values)
                
                # Randomly sample from inactive sequences
                sampled_inactive = np.random.choice(inactive_indices, inactive_samples, replace=False)
                print(f"DEBUG [Neuron {neuron_idx}]: Sampled {len(sampled_inactive)} inactive sequences")
                
                # Return None if no active sequences were found
                if not sampled_active:
                    return None, None, None
                
                # Final validation check to ensure we have enough samples
                if len(sampled_active) < (high_block_samples + (3*mid_block_samples) + low_block_samples) or len(sampled_inactive) < inactive_samples:
                    print(f"FAILURE!! [Neuron {neuron_idx}]: Final sample counts insufficient (got {len(sampled_active)} active, {len(sampled_inactive)} inactive). Skipping...")
                    return None, None, None
                # Check for duplicate active indices
                if len(set(sampled_active)) != len(sampled_active):
                    print(f"DUPLICATION failure [Neuron {neuron_idx}]: Duplicate active indices found!")
                    return None, None, None
                # Check for duplicate inactive indices
                if len(set(sampled_inactive)) != len(sampled_inactive):
                    print(f"DUPLICATION failure [Neuron {neuron_idx}]: Duplicate inactive indices found!")
                    return None, None, None

                print(f"DEBUG [Neuron {neuron_idx}]: Final counts - Active: {len(sampled_active)}, Inactive: {len(sampled_inactive)}")
                return np.array(sampled_active), np.array(sampled_active_acts), sampled_inactive
            else:
                # Handle embedding mode without ReLU 
                # Find sequences with very low activation (0.0-0.2)
                inactive_mask = (neuron_activations >= 0.0) & (neuron_activations < 0.2)
                inactive_candidates = np.where(inactive_mask)[0]
                if len(inactive_candidates) < inactive_samples:
                    print(f"Warning [Neuron {neuron_idx}]: Only found {len(inactive_candidates)} inactive sequences")
                    return None, None, None
                sampled_inactive = np.random.choice(inactive_candidates, inactive_samples, replace=False)

                # Initialize collection variables
                sampled_active = []
                sampled_active_acts = []
                used_indices = set()

                # Sample from highest activation block first
                selected, values, _ = sample_from_block_with_spillover(
                    neuron_activations, len(blocks)-1, high_block_samples, used_indices)
                print(f"DEBUG [Neuron {neuron_idx}]: Sampled {len(selected)} sequences from high block (0.8-1.0)")
                sampled_active.extend(selected)
                sampled_active_acts.extend(values)

                # Sample from middle blocks
                for block_idx in range(1, len(blocks)-1):
                    selected, values, _ = sample_from_block_with_spillover(
                        neuron_activations, block_idx, mid_block_samples, used_indices)
                    print(f"DEBUG [Neuron {neuron_idx}]: Sampled {len(selected)} sequences from block {blocks[block_idx][0]}-{blocks[block_idx][1]}")
                    sampled_active.extend(selected)
                    sampled_active_acts.extend(values)

                # Sample from lowest activation block
                selected, values, _ = sample_from_block_with_spillover(
                    neuron_activations, 0, low_block_samples, used_indices)
                print(f"DEBUG [Neuron {neuron_idx}]: Sampled {len(selected)} sequences from low block (0.0-0.2)")
                sampled_active.extend(selected)
                sampled_active_acts.extend(values)

                # Return None if no active sequences were found
                if not sampled_active:
                    print(f"Warning [Neuron {neuron_idx}]: Could not sample any active sequences")
                    return None, None, None

                return np.array(sampled_active), np.array(sampled_active_acts), sampled_inactive
        else:  # Handle sparse activation mode 
            # Split into active and inactive sequences
            active_indices = np.where(neuron_activations > 0)[0]
            inactive_indices = np.where(neuron_activations == 0)[0]
            
            # Print debug information
            print(f"DEBUG [Neuron {neuron_idx}]: Found {len(active_indices)} total active sequences")
            print(f"DEBUG [Neuron {neuron_idx}]: Found {len(inactive_indices)} total inactive sequences")
            
            # Check if we have enough sequences to sample from
            if len(active_indices) < 2*(high_block_samples + (3*mid_block_samples) + low_block_samples) or len(inactive_indices) < inactive_samples:
                print(f"Warning [Neuron {neuron_idx}]: Insufficient sequences (needs {2*(high_block_samples + (3*mid_block_samples) + low_block_samples)} active samples and {inactive_samples} inactive samples). Skipping...")
                return None, None, None
            
            # Initialize collection variables
            sampled_active = []
            sampled_active_acts = []
            used_indices = set()
            
            # Find sequences in highest activation block
            high_block_mask = ((neuron_activations[active_indices] >= 0.8) & 
                              (neuron_activations[active_indices] <= 1.0))
            high_block_count = np.sum(high_block_mask)
            print(f"DEBUG [Neuron {neuron_idx}]: High block (0.8-1.0): {high_block_count} sequences")
            
            # Sample from highest block
            selected, values, _ = sample_from_block_with_spillover(
                neuron_activations[active_indices], len(blocks)-1, high_block_samples, used_indices)
            print(f"DEBUG [Neuron {neuron_idx}]: Sampled {len(selected)} sequences from high block")
            sampled_active.extend(active_indices[selected])
            sampled_active_acts.extend(values)
            
            # Sample from remaining blocks
            for block_idx in range(len(blocks)-1):
                # Find sequences in current block
                block_mask = ((neuron_activations[active_indices] >= blocks[block_idx][0]) & 
                             (neuron_activations[active_indices] < blocks[block_idx][1]))
                block_count = np.sum(block_mask)
                print(f"DEBUG [Neuron {neuron_idx}]: Block {block_idx} ({blocks[block_idx][0]:.1f}-{blocks[block_idx][1]:.1f}): {block_count} sequences")
                
                # Sample from current block
                selected, values, _ = sample_from_block_with_spillover(
                    neuron_activations[active_indices], block_idx, block_sizes[block_idx], used_indices)
                print(f"DEBUG [Neuron {neuron_idx}]: Sampled {len(selected)} sequences from block {block_idx}")
                
                sampled_active.extend(active_indices[selected])
                sampled_active_acts.extend(values)
            
            # Randomly sample from inactive sequences
            sampled_inactive = np.random.choice(inactive_indices, inactive_samples, replace=False)
            print(f"DEBUG [Neuron {neuron_idx}]: Sampled {len(sampled_inactive)} inactive sequences")
            
            # Return None if no active sequences were found
            if not sampled_active:
                return None, None, None
            
            # Final validation check to ensure we have enough samples
            if len(sampled_active) < (high_block_samples + (3*mid_block_samples) + low_block_samples) or len(sampled_inactive) < inactive_samples:
                print(f"FAILURE!! [Neuron {neuron_idx}]: Final sample counts insufficient (got {len(sampled_active)} active, {len(sampled_inactive)} inactive). Skipping...")
                return None, None, None
            # Check for duplicate active indices
            if len(set(sampled_active)) != len(sampled_active):
                print(f"DUPLICATION failure [Neuron {neuron_idx}]: Duplicate active indices found!")
                return None, None, None
            # Check for duplicate inactive indices
            if len(set(sampled_inactive)) != len(sampled_inactive):
                print(f"DUPLICATION failure [Neuron {neuron_idx}]: Duplicate inactive indices found!")
                return None, None, None

            print(f"DEBUG [Neuron {neuron_idx}]: Final counts - Active: {len(sampled_active)}, Inactive: {len(sampled_inactive)}")
            return np.array(sampled_active), np.array(sampled_active_acts), sampled_inactive









    def format_sequence_data(self, seq_idx, activation=None, neuron_idx=None):

        # Get sequence metadata from DataFrame
        seq_data = self.metadata.iloc[seq_idx]
        
        # Convert activation to standard Python float if it exists
        if activation is not None:
            activation = float(activation)  # Convert from float32/float64 to Python float
        

        
        # Helper function to safely split comma-separated strings
        def safe_split(column_name):
            if column_name in seq_data.index and pd.notna(seq_data[column_name]):
                return seq_data[column_name].split(',')
            return []
        
        # Helper function to safely get string values
        def safe_get(column_name):
            if column_name in seq_data.index and pd.notna(seq_data[column_name]):
                return str(seq_data[column_name])
            return ''
        
        # Build formatted dictionary 
        formatted_data = {
            'sequence_id': safe_get('sequence_id'),
            'activation': activation if activation is not None else 0,
            'protein_families': safe_split('protein_families'),
            'go_biological': safe_split('go_biological'),
            'go_cellular': safe_split('go_cellular'),
            'go_molecular': safe_split('go_molecular'),            
            'protein_names': safe_get('protein_names'),
            'gene_names': safe_get('gene_names'),
            'entry': safe_get('entry'),
            'reviewed': safe_get('reviewed'),
            'entry_name': safe_get('entry_name'),
            'organism': safe_get('organism'),
            'length': safe_get('length')
        }
        
        return formatted_data 






    def get_claude_interpretation(self, train_data):

        # Add debug prints for activation ranges
        positive_activations = [seq['activation'] for seq in train_data['positive']]
        negative_activations = [seq['activation'] for seq in train_data['negative']]
        print(f"\nDEBUG [Neuron {train_data['neuron_idx']}] Activation ranges:")
        print(f"Active sequences - Min: {min(positive_activations):.10f}, Max: {max(positive_activations):.10f}")
        print(f"Inactive sequences - Min: {min(negative_activations):.10f}, Max: {max(negative_activations):.10f}")

        # Construct the prompt
        prompt = f"""You are an expert at interpreting the neurons of a neural network. 
        
I am giving you the metedata for protein sequences that activate neuron #{train_data['neuron_idx']} in a neural network, along with their scaled activation levels. The activation levels lie between 0 and 1. 

ACTIVE EXAMPLES (activate the neuron):"""

        for seq in train_data['positive']:
            prompt += f"""

Sequence ID: {seq['sequence_id']}
Activation: {seq['activation']:.10f}
Protein Names: {seq['protein_names']}
Gene Names: {seq['gene_names']}
Protein Families: {', '.join(seq['protein_families']) if seq['protein_families'] else 'None'}
GO Terms:
  Biological Process: {', '.join(seq['go_biological']) if seq['go_biological'] else 'None'}
  Cellular Component: {', '.join(seq['go_cellular']) if seq['go_cellular'] else 'None'}
  Molecular Function: {', '.join(seq['go_molecular']) if seq['go_molecular'] else 'None'}"""

        prompt += """

INACTIVE EXAMPLES (do not activate the neuron):"""

        for seq in train_data['negative']:
            prompt += f"""

Sequence ID: {seq['sequence_id']}
Activation: {seq['activation']:.10f}
Protein Names: {seq['protein_names']}
Gene Names: {seq['gene_names']}
Protein Families: {', '.join(seq['protein_families']) if seq['protein_families'] else 'None'}
GO Terms:
  Biological Process: {', '.join(seq['go_biological']) if seq['go_biological'] else 'None'}
  Cellular Component: {', '.join(seq['go_cellular']) if seq['go_cellular'] else 'None'}
  Molecular Function: {', '.join(seq['go_molecular']) if seq['go_molecular'] else 'None'}"""

        prompt += """

There is something within the metadata that is causing this neuron to fire. That is what this neuron is looking for. Your job is the following: based on the active and inactive examples above, provide a human-readable interpretation of what this neuron appears to be capturing. Consider the metadata available. 
Your response should be no longer than a single sentence."""


        # Save prompt to a file (new addition)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        prompt_filename = os.path.join(self.prompts_dir, f'neuron_{train_data["neuron_idx"]}_training_prompt_{timestamp}.txt')
        with open(prompt_filename, 'w') as f:
            f.write(prompt)
        print(f"\nSaved training interpretation prompt to: {prompt_filename}")

        # Handle offline mode where user manually interacts with Claude
        if self.offline_mode:
            neuron_idx = train_data['neuron_idx']
            train_file = os.path.join(self.prompts_dir, f'neuron_{neuron_idx}_train.txt')
            # Save prompt to file for manual processing
            with open(train_file, 'w') as f:
                f.write(prompt)
            print(f"\nSaved interpretation prompt to {train_file}")
            print("Please paste the content to Claude and enter the interpretation:")
            response = input().strip()
            return response
        # Handle online mode with direct API calls
        else:
            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=100,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content

    def get_claude_predictions(self, interpretation, test_data, neuron_idx):

        # Extract plain text from interpretation if it's a TextBlock response
        if isinstance(interpretation, str):
            clean_interpretation = interpretation
        else:
            # Handle case where interpretation contains TextBlock
            clean_interpretation = interpretation[0].text if isinstance(interpretation, list) else interpretation.text
        prompt = f"""You are incredibly intelligent and you are an expert at being able to predict the activation level of a neuron in a neural network, if you are told what the neuron's interpretation is. This neuron is looking for something in the metadata of the protein sequence.
        
INTERPRETATION:
{clean_interpretation}

Your job is as follows: carefully read the interpretation (above) which tells you what this neuron is doing, and use it to predict the activation level of this neuron on the following protein sequences, whose metadata I have provided below. 

Keep in mind that the activation levels lie between 0 and 1. I am giving you metadata for exactly {len(test_data)} protein sequences in the test set below. So you must return exactly {len(test_data)} numbers, one per line, representing the predicted activation level corresponding to each of the {len(test_data)} protein sequences in the test set.

You must not return anything else or any other text! You will be penalised if you return anything other than {len(test_data)} numbers, one per line. 

Test sequences:"""

        # Add test sequences
        for seq in test_data:
            prompt += f"""

Sequence ID: {seq['sequence_id']}
Protein Names: {seq['protein_names']}
Gene Names: {seq['gene_names']}
Protein Families: {', '.join(seq['protein_families']) if seq['protein_families'] else 'None'}
GO Terms:
  Biological Process: {', '.join(seq['go_biological']) if seq['go_biological'] else 'None'}
  Cellular Component: {', '.join(seq['go_cellular']) if seq['go_cellular'] else 'None'}
  Molecular Function: {', '.join(seq['go_molecular']) if seq['go_molecular'] else 'None'}"""

        # Save prediction prompt to a file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        prompt_filename = os.path.join(self.prompts_dir, f'neuron_{neuron_idx}_prediction_prompt_{timestamp}.txt')
        with open(prompt_filename, 'w') as f:
            f.write(prompt)
        print(f"\nSaved prediction prompt to: {prompt_filename}")

        if self.offline_mode:
            test_file = os.path.join(self.prompts_dir, f'neuron_{neuron_idx}_test.txt')
            # Save prompt to file for manual processing
            with open(test_file, 'w') as f:
                f.write(prompt)
            print(f"\nSaved prediction prompt to {test_file}")
            print(f"Expected {len(test_data)} predictions (one per test sequence)")
            print("Enter Claude's predictions (one number per line):")
            print("Press Enter twice (i.e., submit an empty line) when done:")
            
            # Collect predictions from user input
            predictions = []
            while True:
                line = input().strip()
                if not line:  # If empty line, break the loop
                    break
                try:
                    predictions.append(float(line))
                except ValueError:
                    print(f"Skipping invalid input: {line}")
                    continue
            
            print(f"Received {len(predictions)} predictions.")
            return np.array(predictions)
        else:
            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            # Update: Handle the new response format
            predictions = []
            # Access the content string from the Message object
            content = response.content[0].text if isinstance(response.content, list) else response.content
            # Parse predictions line by line
            for line in content.split('\n'):
                try:
                    pred = float(line.strip())
                    predictions.append(pred)
                except:
                    continue
            
            # Save predictions and actual values to a file
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            predictions_filename = os.path.join(self.prompts_dir, f'neuron_{neuron_idx}_predictions_vs_actual_{timestamp}.txt')
            with open(predictions_filename, 'w') as f:
                f.write(f"Neuron {neuron_idx} Predictions vs Actual Values\n")
                f.write("Format: sequence_id\tprediction\tactual\n")
                f.write("-" * 50 + "\n")
                for pred, seq in zip(predictions, test_data):
                    try:
                        # Just store the sequence_id as is, without trying to use it as an index
                        f.write(f"{seq['sequence_id']}\t{pred:.10f}\t{seq['activation']:.10f}\n")
                    except Exception as e:
                        print(f"Warning: Could not write prediction for sequence {seq.get('sequence_id', 'unknown')}: {str(e)}")
                        continue
            print(f"\nSaved predictions and actual values to: {predictions_filename}")

            return np.array(predictions)






    def save_result(self, result):

        if result:
            # Extract plain text from interpretation if needed
            interpretation = result['interpretation']
            if not isinstance(interpretation, str):
                interpretation = interpretation[0].text if isinstance(interpretation, list) else interpretation.text
            
            # Save to text file as before
            with open(self.results_file, 'a') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Neuron: {result['neuron_idx']}\n")
                f.write(f"Correlation: {result['correlation']:.3f}\n")
                f.write(f"P-value: {result['p_value']:.3f}\n")
                f.write(f"Interpretation: {interpretation}\n")
            
            # Save to CSV file
            csv_file = self.results_file.replace('.txt', '.csv')
            # Create CSV file with headers if it doesn't exist
            if not os.path.exists(csv_file):
                with open(csv_file, 'w') as f:
                    f.write("neuron_idx,correlation,p_value,interpretation\n")
            
            # Append result to CSV file
            with open(csv_file, 'a') as f:
                # Wrap interpretation in quotes and escape any existing quotes
                safe_interpretation = '"' + interpretation.replace('"', '""') + '"'
                f.write(f"{result['neuron_idx']},{result['correlation']:.3f},{result['p_value']:.3f},{safe_interpretation}\n")

    def analyze_neuron(self, neuron_idx, high_block_samples=6, mid_block_samples=3, 
                            low_block_samples=2, inactive_samples=15):

            # Calls helper method to retrieve sequences that activate this neuron at different levels
            # Returns active sequences, their activation values, and inactive sequences
            result = self.get_sequences_for_neuron(neuron_idx, high_block_samples=high_block_samples, 
                                                mid_block_samples=mid_block_samples, 
                                                low_block_samples=low_block_samples, 
                                                inactive_samples=inactive_samples)
            
            # If insufficient sequences were found, skip analysis of this neuron
            # if result is None or result == (None, None, None):
            if (result is None or 
                (isinstance(result, tuple) and all(x is None for x in result))):
                print(f"Skipping neuron {neuron_idx} due to insufficient sequences")
                return None
            
            # Extracts the three components returned by get_sequences_for_neuron
            # active_indices = indices of sequences that activate the neuron
            # scaled_activations = activation values for those sequences
            # inactive_indices = indices of sequences where neuron is inactive
            active_indices, scaled_activations, inactive_indices = result
            
            # Simple random split for active sequences
            num_active = len(active_indices)
            train_size_active = int(num_active * self.train_ratio)
            
            # Randomly shuffle active sequences and their activations together
            perm = np.random.permutation(num_active)
            shuffled_active_indices = np.array(active_indices)[perm]
            shuffled_active_values = np.array(scaled_activations)[perm]
            
            # Split into train/test
            train_active_indices = shuffled_active_indices[:train_size_active]
            train_active_values = shuffled_active_values[:train_size_active]
            test_active_indices = shuffled_active_indices[train_size_active:]
            test_active_values = shuffled_active_values[train_size_active:]
            
            # Simple random split for inactive sequences
            train_size_inactive = int(len(inactive_indices) * self.train_ratio)
            perm_inactive = np.random.permutation(len(inactive_indices))
            shuffled_inactive = np.array(inactive_indices)[perm_inactive]
            train_inactive_indices = shuffled_inactive[:train_size_inactive]
            test_inactive_indices = shuffled_inactive[train_size_inactive:]
            
            # Creates training data dictionary for Claude
            # Each sequence is formatted using format_sequence_data helper method
            train_data = {
                'neuron_idx': neuron_idx,
                'positive': [self.format_sequence_data(idx, act, neuron_idx) 
                            for idx, act in zip(train_active_indices, train_active_values)],
                'negative': [self.format_sequence_data(idx, self.full_activation_matrix[idx, neuron_idx], neuron_idx) 
                            for idx in train_inactive_indices]
            }
            
            # Creates test data list combining active and inactive sequences
            # Each sequence is formatted using format_sequence_data helper method
            test_data = (
                [self.format_sequence_data(idx, act, neuron_idx) 
                for idx, act in zip(test_active_indices, test_active_values)] +
                [self.format_sequence_data(idx, self.full_activation_matrix[idx, neuron_idx], neuron_idx) 
                for idx in test_inactive_indices]
            )
            
            # Creates array of actual activation values for test sequences
            # Randomly shuffles test data and corresponding actual values
            actual = np.concatenate([
                test_active_values,
                [self.full_activation_matrix[idx, neuron_idx] for idx in test_inactive_indices]
            ])
            shuffle_indices = np.random.permutation(len(test_data))
            test_data = [test_data[i] for i in shuffle_indices]
            actual = actual[shuffle_indices]
            
            interpretation = self.get_claude_interpretation(train_data)
            
            predictions = self.get_claude_predictions(interpretation, test_data, neuron_idx)
            
            # Validates that number of predictions matches number of test sequences
            if len(predictions) != len(actual):
                print(f"\nWarning: Number of predictions ({len(predictions)}) doesn't match number of test sequences ({len(actual)})")
                print("Expected predictions for all test sequences. Skipping this neuron.")
                return None
            
            # Calculates Pearson correlation between predicted and actual activation values
            correlation, p_value = pearsonr(actual, predictions)
            
            # Creates results dictionary with all analysis outputs
            result = {
                'neuron_idx': neuron_idx,
                'interpretation': interpretation,
                'correlation': correlation,
                'p_value': p_value,
                'actual_values': actual.tolist(),
                'predicted_values': predictions.tolist()
            }
            
            # Saves results to file if running in offline mode
            self.save_result(result)
            return result




def main():
    
    # Parse command line arguments
    args = parse_args()

    # Set random seeds at start of main
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Find POOLING files in the specified folder
    files = find_pooling_files(args.data_folder)
    if files is None:
        print("Aborting due to file discovery error")
        return
        
    embeddings_file, sparse_matrix_file, metadata_file = files
    
    # Initialize the neuron interpreter with discovered files
    interpreter = NeuronInterpreter(
        embeddings_file=embeddings_file,
        sparse_matrix_file=sparse_matrix_file,
        metadata_file=metadata_file,
        claude_api_key=args.claude_api_key,
        offline_mode=args.offline_mode,
        output_dir=args.output_dir,
        embedding_mode=args.embedding_mode,
        include_sequence=args.include_sequence,
        train_ratio=args.train_ratio,
        embedding_relu=args.embedding_relu,
        specific_neuron=args.specific_neuron,
        num_neurons=args.num_neurons,
        high_block_samples=args.high_block_samples,
        mid_block_samples=args.mid_block_samples,
        low_block_samples=args.low_block_samples,
        inactive_samples=args.inactive_samples,
        random_seed=args.random_seed
    )
    
    # Handle specific neuron case
    if args.specific_neuron is not None:
        print(f"\nAnalyzing specific neuron {args.specific_neuron}")
        if args.specific_neuron >= interpreter.num_neurons:
            print(f"Error: Specified neuron {args.specific_neuron} exceeds maximum neuron index {interpreter.num_neurons-1}")
            return
            
        try:
            result = interpreter.analyze_neuron(
                args.specific_neuron,
                high_block_samples=args.high_block_samples,
                mid_block_samples=args.mid_block_samples,
                low_block_samples=args.low_block_samples,
                inactive_samples=args.inactive_samples
            )
            if result is not None:
                # Save result to JSON
                with open('neuron_interpretations.json', 'w') as f:
                    json.dump([result], f, indent=2)
                print("\nAnalysis Summary:")
                print(f"Neuron {args.specific_neuron}:")
                print(f"Correlation: {result['correlation']:.3f}")
                print(f"Interpretation: {result['interpretation']}")
            else:
                print(f"Could not analyze neuron {args.specific_neuron} due to insufficient data")
        except Exception as e:
            print(f"\nError processing neuron {args.specific_neuron}: {str(e)}")
        return

    # Create a deterministic sampling plan upfront instead of incrementally sampling
    all_neurons = list(range(interpreter.num_neurons))
    # Shuffle the entire list of neurons deterministically
    rng = np.random.RandomState(args.random_seed)
    rng.shuffle(all_neurons)
    attempted_neurons = set()
    results = []
    
    # Add debug prints
    print(f"DEBUG: Running in embedding_mode={args.embedding_mode}")
    print(f"DEBUG: Total number of neurons available: {interpreter.num_neurons}")
    print(f"DEBUG: Number of neurons requested to analyze: {args.num_neurons}")
    
    # Set up progress bar
    pbar = tqdm(total=args.num_neurons)
    # Main analysis loop - now uses the pre-shuffled list instead of random sampling each time
    for neuron_idx in all_neurons:
        if len(results) >= args.num_neurons:
            break
        
        if neuron_idx in attempted_neurons:
            continue 
        attempted_neurons.add(neuron_idx)
        
        # Update progress bar with current neuron
        pbar.set_description(f"Processing neuron {neuron_idx}")
        try:
            # Analyze the selected neuron with explicit sampling parameters
            result = interpreter.analyze_neuron(
                neuron_idx,
                high_block_samples=args.high_block_samples,
                mid_block_samples=args.mid_block_samples,
                low_block_samples=args.low_block_samples,
                inactive_samples=args.inactive_samples
            )
            if result is not None:
                results.append(result)
                pbar.update(1)
                ####If ever run into weird memory issues, then uncomment the two lines below:
                #Save results after each successful analysis
                # with open('neuron_interpretations.json', 'w') as f:
                #     json.dump(results, f, indent=2)
        except Exception as e:
            print(f"\nError processing neuron {neuron_idx}: {str(e)}")
            continue
    
    # Close progress bar
    pbar.close()
    
    # Print analysis summary
    print("\nAnalysis Summary:")
    print(f"Successfully analyzed: {len(results)}")
    print(f"Total neurons attempted: {len(attempted_neurons)}")
    
    # Handle case where no neurons could be analyzed
    if not results:
        print("\nNo neurons could be analyzed. All sampled neurons had insufficient data.")
        print("Try decreasing the required number of sequences.")
        return
    
    # Calculate statistics
    correlations = [r['correlation'] for r in results]
    p_values = [r['p_value'] for r in results]
    # Filter out nan values before computing statistics
    valid_correlations = [c for c in correlations if not np.isnan(c)]
    avg_correlation = np.mean(valid_correlations) if valid_correlations else float('nan')
    median_correlation = np.median(valid_correlations) if valid_correlations else float('nan')
    valid_p_values = [p for p in p_values if not np.isnan(p)]
    significant_neurons = sum(p < 0.05 for p in valid_p_values)
    
    # Print summary to terminal
    print(f"\nAverage correlation: {avg_correlation:.3f}")
    print(f"Median correlation: {median_correlation:.3f}")
    print(f"Neurons with p < 0.05: {significant_neurons} out of {len(valid_p_values)} ({(significant_neurons/len(valid_p_values))*100:.1f}% of valid results)")
    
    # Add summary statistics to results file
    with open(interpreter.results_file, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Analysis Summary:\n")
        f.write(f"Successfully analyzed: {len(results)}\n")
        f.write(f"Total neurons attempted: {len(attempted_neurons)}\n")
        f.write(f"Average Pearson correlation: {avg_correlation:.3f}\n")
        f.write(f"Median Pearson correlation: {median_correlation:.3f}\n")
        f.write(f"Neurons with p < 0.05: {significant_neurons} out of {len(valid_p_values)} ({(significant_neurons/len(valid_p_values))*100:.1f}% of valid results)\n")
    
    # Display detailed activation statistics for each analyzed neuron
    print("\nNeuron Activation Summary:")
    for result in results:
        neuron_idx = result['neuron_idx']
        neuron_activations = interpreter.full_activation_matrix[:, neuron_idx]
        max_act = np.max(neuron_activations)
        mean_act = np.mean(neuron_activations)
        active_pct = np.mean(neuron_activations > 0) * 100
        print(f"Neuron {neuron_idx}: Max={max_act:.10f}, Mean={mean_act:.10f}, Active={active_pct:.1f}%")
        
        # Print detailed correlation analysis
        print(f"\nNeuron {neuron_idx} Correlation Details:")
        actual = result.get('actual_values', [])
        predicted = result.get('predicted_values', [])
        if actual and predicted:
            print("Actual vs Predicted values:")
            for a, p in zip(actual, predicted):
                print(f"Actual: {a:.10f}, Predicted: {p:.10f}")

    # Add summary statistics to CSV file
    csv_file = interpreter.results_file.replace('.txt', '.csv')
    with open(csv_file, 'a') as f:
        f.write("\n# Analysis Summary\n")
        f.write(f"# Successfully analyzed,{len(results)}\n")
        f.write(f"# Total neurons attempted,{len(attempted_neurons)}\n")
        f.write(f"# Average Pearson correlation,{avg_correlation:.3f}\n")
        f.write(f"# Median Pearson correlation,{median_correlation:.3f}\n")
        f.write(f"# Significant neurons (p < 0.05),{significant_neurons}\n")
        f.write(f"# Percentage significant,{(significant_neurons/len(valid_p_values))*100:.1f}%\n")

# Entry point of the script
if __name__ == "__main__":
    main()