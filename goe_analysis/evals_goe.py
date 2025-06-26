# 1

import numpy as np
import pandas as pd
import ast
from goatools.obo_parser import GODag
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.anno.gaf_reader import GafReader
from goatools.anno.factory import get_objanno
import multiprocessing as mp
from rich.progress import Progress
import os
import argparse
import sys
from contextlib import contextmanager

@contextmanager
def silence():
    original_stdout = sys.stdout
    sys.stdout = None 
    try:
        yield
    finally:
        sys.stdout = original_stdout 


parser = argparse.ArgumentParser(description="Gene ontology enrichment analysis per feature")
# parser.add_argument("--sparse_path", type=str, required=True,
#                     help="Path to the sparse embeddings file.")
# parser.add_argument("--sparse_indices_path", type=str, required=True,
#                     help="Path to the sparse indices file.")
# parser.add_argument("--metadata_path", type=str, required=True,
#                     help="Path to the metadata file.")
parser.add_argument("--input_dir_path", type=str, required=True,
                    help="Path to the input directory.")
# parser.add_argument("--out_dir_path", type=str, required=True,
#                     help="Path to the output directory.")

args = parser.parse_args()

        
# human_sparse_path = "/data/cb/scratch/onkar/viral-mutation/Extracted_Data/POOLING_activations_time20241116_231918_ckptUPDATED_USED_CHECKPOINT_MeanPooled_sae_20241114_034556_esmt12_k64_md0_hd20000_lr0.0001_ep100_maxsamples20000.npy"
# human_sparse_indices_path = "/data/cb/scratch/onkar/viral-mutation/Extracted_Data/POOLING_indices_time20241116_231918_ckptUPDATED_USED_CHECKPOINT_MeanPooled_sae_20241114_034556_esmt12_k64_md0_hd20000_lr0.0001_ep100_maxsamples20000.npy"
# human_metdata_path = "/data/cb/scratch/onkar/viral-mutation/Extracted_Data/metadata_time20241116_225206_ckptUSED_CHECKPOINT_sae_20241114_030518_esmt12_k32_md0_hd20000_lr0.0001_ep100.csv"

# sparse_path = args.sparse_path
# sparse_indices_path = args.sparse_indices_path
# metadata_path = args.metadata_path
# out_dir_path = args.out_dir_path
out_dir_path = os.path.join(args.input_dir_path, "out")


for file in os.listdir(args.input_dir_path):
    if "activations" in file:
        sparse_path= os.path.join(args.input_dir_path, file)
    if "indices" in file:
        sparse_indices_path= os.path.join(args.input_dir_path, file)
    if "metadata" in file:
        metadata_path= os.path.join(args.input_dir_path, file)
        
save_path = os.path.join(out_dir_path, "goe")

if not os.path.isfile(sparse_path):
    raise FileNotFoundError(f"The sparse file '{sparse_path}' does not exist.")
if not os.path.isfile(sparse_indices_path):
    raise FileNotFoundError(f"The sparse indices file '{sparse_indices_path}' does not exist.")

if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path)
    print(f"Output directory '{out_dir_path}' created.")

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Output GOE directory '{save_path}' created.")


print(f"GO Enrichment of sparse embeddings from {sparse_path}")
print(f"Using sparse indices from {sparse_indices_path}")
print(f"Saving output to {save_path}")


go_dag = GODag("go.obo")
gaf_reader = get_objanno("goa_human.gaf")
go_annotations = gaf_reader.get_id2gos(namespace="bp")                  # change namespace for bp, mf, cc
sparse_global_embeddings = np.load(sparse_path, mmap_mode='r')          # shape (50000, 64)
sparse_global_indices = np.load(sparse_indices_path, mmap_mode='r')     # shape (50000, 64)
metadata = pd.read_csv(metadata_path)
# metadata["go_biological"] = metadata["go_biological"].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) and ast.literal_eval(x) else None)
# metadata["go_cellular"] = metadata["go_cellular"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and ast.literal_eval(x) else None)
# metadata["go_molecular"] = metadata["go_molecular"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and ast.literal_eval(x) else None)

human_indices = metadata[metadata.organism.str.contains("Human")].index
human_protein_set = set(metadata["sequence_id"][human_indices])

# using all activated proteins
sample_size = len(human_indices)
sparse_embeddings_sample = np.zeros((sample_size,20000))
sparse_embeddings_sample[np.arange(sample_size).reshape(-1,1),sparse_global_indices[human_indices]] = sparse_global_embeddings[human_indices]

total_activation_per_neuron = np.sum(sparse_embeddings_sample, axis=0)  # computing total activation of each neuron (sum of per protein activations (axis=0)) for later filteirng
activated_neuron_mask = total_activation_per_neuron > 0
activated_neuron_indices = np.arange(20000)[activated_neuron_mask]

activations_per_neuron_df = pd.DataFrame(sparse_embeddings_sample[:,activated_neuron_mask] , columns=activated_neuron_indices)


def go_enrichment(protein_set):
    population_proteins = human_protein_set
    goea = GOEnrichmentStudy(population_proteins, go_annotations, go_dag, methods=["bonferroni"])
    goea_results = goea.run_study(protein_set)
    return goea_results


def go_per_neuron(idx):
    filename = os.path.join(save_path,f'feature{idx}_goe.csv')
    if os.path.exists(filename):
        return
    neuron_key = idx
    activated_proteins_per_neuron = metadata["sequence_id"][activations_per_neuron_df[neuron_key] != 0.0]   # maybe better threshold?
    with silence():
        results = go_enrichment(set(activated_proteins_per_neuron.tolist()))
        feature = []
        names = []
        terms = []
        pvalues = []
        if len(results) != 0:
            for result in results:
                if result.get_pvalue() < 0.05:
                    # print(result.GO, result.get_pvalue(), result.name)
                    feature.append(idx)
                    terms.append(result.GO)
                    names.append(result.name)
                    pvalues.append(result.get_pvalue())
    
        result_df = pd.DataFrame({"Feature":feature,"Term":terms, "Name":names, "pvalues":pvalues })

        result_df.to_csv(filename, index=False)


with Progress() as progress:
    task = progress.add_task("GO Enrichment tests...", total=len(activated_neuron_indices))
    
    with mp.Pool() as pool:
        for neuron_id in pool.imap_unordered(go_per_neuron, activated_neuron_indices):
            progress.update(task, advance=1)

print("GO Processing complete. All CSVs have been saved.")
