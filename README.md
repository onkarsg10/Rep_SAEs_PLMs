# Interpreting SAEs and transcoders trained on PLM representations

This repository helps interpret SAEs and transcoders trained on protein-level representations from ESM2 using both GO Analysis and automated LLM-assisted interpretations, and SAEs trained on amino acid-level representations using automated LLM-assisted interpretations.  

## Prerequisites

The following is necessary for all the SAEs and transcoders:

1. `swissprot.tsv` needs to be present in your current working directory, or the path to it must be provided in the `swissprot_filtered_uniref_dataset.py` script. This is simply the Swissprot dataset downloadable as a TSV file (https://www.uniprot.org/uniprotkb?query=*&facets=reviewed%3Atrue) in order to exclude its entries from SAE training. This is because the TSV  later gets used for analysis.

2. `Topk_weights` folder must exist in your current working directory. This folder is where the trained model weights will get saved.

3. The path to the uniref_file (`uniref50.fasta.gz`) needs to be provided as an argument. It can be downloaded from: https://www.uniprot.org/help/downloads as a .fasta.gz file.

4. In addition to the above, the following files are also used for GO analysis: [go.obo file](https://geneontology.org/docs/download-ontology/) and the [goa_human.gaf file](https://current.geneontology.org/products/pages/downloads.html) which can be downloaded from the links. 


## Training Commands

### Training SAE on Protein-Level Representations

The following is a sample of the command used to train the SAE on the protein-level representations (formed by mean-pooling the token-level representations). Irrespective of the number of epochs, max_steps dictates how many steps of training occur. You would need to provide your wandb API key as an argument. 

```bash
python Folder_Random_Seed_Regular_Pooling_Scripts/main_script.py --uniref_file uniref50.fasta.gz --batch_size 128 --learning_rate 0.0001 --k 64 --hidden_dim 20000 --epochs 100 --encoder_decoder_init 1 --max_seq_len 1024 --esm_model esm2_t12_35M_UR50D --inactive_threshold 200 --val_check_interval 300 --limit_val_batches 40 --max_samples 50000000 --max_steps 180000 --esm_layer 8 --cuda_device 1
```

Note: in case Wandb causes timeout issues, you may need to uncomment the line in main_script.py about os.environ["WANDB__SERVICE_WAIT"]. 

### Training Transcoder on Protein-Level Representations

Similarly, the following is a sample of the command for training the transcoder on the protein-level representations. Notice the additional argument `return_difference`, which we set to 1 in our transcoder training runs. This simply indicates that the target that the autoencoder aims to predict subtracts out the previous layer's representation.

```bash
python Folder_Random_Seed_TC_Pooling_Scripts/main_script.py --uniref_file uniref50.fasta.gz --batch_size 128 --learning_rate 0.0001 --k 64 --hidden_dim 20000 --epochs 100 --encoder_decoder_init 1 --max_seq_len 1024 --esm_model esm2_t12_35M_UR50D --inactive_threshold 200 --val_check_interval 300 --limit_val_batches 40 --max_samples 50000000 --max_steps 180000 --esm_layer 8 --cuda_device 0 --return_difference 1
```

### Training SAE on AA-Level Representations

This is a sample of the command for training the SAE on the AA-level representations:

```bash
python Flatten_instead_of_Pool/main_script.py --uniref_file uniref50.fasta.gz --batch_size 128 --learning_rate 0.0001 --k 16 --hidden_dim 20000 --epochs 100 --encoder_decoder_init 1 --max_seq_len 1024 --esm_model esm2_t12_35M_UR50D --inactive_threshold 50 --val_check_interval 300 --limit_val_batches 40 --max_samples 50000000 --max_steps 70000 --esm_layer 10 --cuda_device 2
```

## Data Extraction & GO Analysis Commands

### Extracting Data from Protein-Level Representations

The following is a sample of the command that is used to extract data from the SAEs and transcoders trained on protein-level representations for automated interpretation:

```bash
python Folder_Results_Storing_Scripts/numpy_store_results.py \
    --uniref_file "swissprot.tsv" \
    --batch_size 64 \
    --esm_model "esm2_t12_35M_UR50D" \
    --max_seq_len 1024 \
    --random_seed 42 \
    --separate 1 \
    --human_filter 0 \
    --max_samples 50000 \
    --cuda_device 0 \
    --ckpt_file "Topk_weights/saved_checkpoint.ckpt"\
    --esm_layer 9 \
    --output_dir "output_directory"
```

Here, `Topk_weights/saved_checkpoint.ckpt` is the path to the saved checkpoint saved after SAE/transcoder training, and `output_directory` is the path where you want to save the outputs from the data extraction.

For the GO analysis, the same command is used, with `human_filter 1` and `max_samples = 20000` instead.

Running the command results in a CSV file containing metadata of samples from Swissprot, and numpy files containing activation information (in the `output_directory`) that gets used for the automated interpretation/GO analysis.

### GO Analysis Commands

There are three main files for GOE analysis: `evals_goe.py`, `evals_goe_pairwise_analysis.py`, and `evals_plots.py`, and each file should be run in this order. The only argument is the `--input_dir_path`, which is the directory that stores the CSV metadata and .npy files that contain activation information (outputted from the aforementioned step). Example commands would be of the form:

```bash
python goe_analysis/evals_goe.py --input_dir_path= # Path to outputted metadata and activations
python goe_analysis/evals_goe_pairwise_analysis.py  --input_dir_path= # Path to outputted metadata and activations
python goe_analysis/evals_plots.py --input_dir_path= # Path to outputted metadata and activations
```

### Extracting Data from AA-Level Representations

The following is a sample of the command that is used to extract data from the SAEs trained on AA-level representations for automated interpretation:

```bash
python FINAL_AA_Results_Storing_Scripts/numpy_store_results.py \
    --uniref_file "swissprot.tsv" \
    --batch_size 64 \
    --esm_model "esm2_t12_35M_UR50D" \
    --max_seq_len 1024 \
    --random_seed 42 \
    --separate 1 \
    --human_filter 0 \
    --max_samples 50000 \
    --cuda_device 0 \
    --ckpt_file "Topk_weights/AA_saved_checkpoint.ckpt"\
    --esm_layer 9 \
    --output_dir "AA_output_directory"
```

## Automated Interpretation Commands

### Interpreting Protein-Level Representations

The following is a sample of the command that is used for the automated interpretation of sparse features for protein-level representations. It makes use of the `output_directory` where data was extracted, and needs to be provided with the Claude API key you intend to use.

```bash
python Folder_Pooling_Autointerp/Pooling_Auto_Interp.py \
    --high_block_samples 6 \
    --mid_block_samples 3 \
    --low_block_samples 2 \
    --inactive_samples 15 \
    --include_sequence 0 \
    --claude_api_key "" \
    --train_ratio 0.5 \
    --random_seed 42 \
    --num_neurons 200 \
    --embedding_mode 0 \
    --data_folder output_directory
```

A similar command can be used for the automated interpretation of ESM neurons on protein-level representations.

```bash
python Folder_Pooling_Autointerp/Pooling_Auto_Interp.py \
    --high_block_samples 6 \
    --mid_block_samples 3 \
    --low_block_samples 2 \
    --inactive_samples 15 \
    --include_sequence 0 \
    --claude_api_key "" \
    --train_ratio 0.5 \
    --random_seed 42 \
    --num_neurons 200 \
    --embedding_mode 1 \
    --embedding_relu 1 \
    --data_folder output_directory
```

Both of these commands produce a CSV file each (among other data) in `output_directory` that contains the automated interpretations of the 200 neurons analysed, along with their scores.

### Interpreting AA-Level Representations

The following is a sample of the command that is used for the automated interpretation of sparse features for AA-level representations. It makes use of the `AA_output_directory` where data was extracted, and needs to be provided with the Claude API key you intend to use. 

```bash
python FINAL_Folder_AA_Autointerp/AA_Auto_Interp.py\
    --high_block_samples 6 \
    --mid_block_samples 3 \
    --low_block_samples 2 \
    --inactive_samples 15 \
    --claude_api_key "" \
    --train_ratio 0.5 \
    --random_seed 42 \
    --num_neurons 200 \
    --embedding_mode 0 \
    --data_folder AA_output_directory
```

A similar command can be used for the automated interpretation of ESM neurons on AA-level representations.

```bash
python FINAL_Folder_AA_Autointerp/AA_Auto_Interp.py\
    --high_block_samples 6 \
    --mid_block_samples 3 \
    --low_block_samples 2 \
    --inactive_samples 15 \
    --claude_api_key "" \
    --train_ratio 0.5 \
    --random_seed 42 \
    --num_neurons 200 \
    --embedding_mode 1 \
    --embedding_relu 1 \
    --data_folder AA_output_directory
```

Both of these commands yield a CSV file each (among other data) in `output_directory` that contains the automated interpretations of the 200 neurons analysed, along with their scores.

## Dependencies

Main dependencies along with versions that were used for SAE/transcoder training and automated interpretation experiments:

- torch: 2.4.1+cu121
- pytorch-lightning: 2.2.0
- wandb: 0.17.7
- numpy: 1.24.4
- pandas: 2.0.1
- scipy: 1.10.1
- tqdm: 4.66.5
- h5py: 3.11.0
- biopython: 1.83
- anthropic: 0.39.0
- fair-esm: 2.0.1
- Python 3.8.18

Dependencies used for GO analysis:

- numpy: 1.24.1
- pandas: 2.0.3
- plotly: 5.24.1
- goatools: 1.4.12
- networkx: 3.0
- rich: 13.9.4
- scikit-learn: 1.3.2
- h5py: 3.11.0
- umap-learn: 0.5.7
- ipywidgets: 7.7.1
- notebook: 7.2.2
- statsmodels: 0.14.1

## Github Acknowledgements/References

Our final SAE/transcoder training implementation draws on implementation ideas and various components/modules of code from the following open-source implementations (along with modifications):

- https://github.com/DanielKerrigan/saefarer/tree/41b5c6789952310f515c461804f12e1dfd1fd32d  
- https://github.com/openai/sparse_autoencoder/tree/4965b941e9eb590b00b253a2c406db1e1b193942 
- https://github.com/EleutherAI/sae/tree/main 
- https://github.com/bartbussmann/BatchTopK/tree/main
- https://github.com/tylercosgrove/sparse-autoencoder-mistral7b/tree/main
- https://github.com/etowahadams/interprot/tree/cbe293403e6427b3a7891f8dde15d2f8d7f81d96/interprot
- https://github.com/tim-lawson/mlsae/tree/main/mlsae/model
- https://github.com/jbloomAus/SAELens 
- https://github.com/neelnanda-io/1L-Sparse-Autoencoder 
