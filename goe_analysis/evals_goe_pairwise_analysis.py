# 2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
from goatools.obo_parser import GODag
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.anno.gaf_reader import GafReader
from goatools.anno.factory import get_objanno
from sklearn.cluster import KMeans
import os
import networkx as nx
from rich.progress import Progress
import multiprocessing as mp
import argparse
import glob


parser = argparse.ArgumentParser(description="Analyzing GO sets per feature (computing pairwise shortest paths and lcas)")
# parser.add_argument("--metadata_path", type=str, required=True,
                    # help="Path to the sparse indices file.")
parser.add_argument("--input_dir_path", type=str, required=True,
                    help="Path to the output directory.")
# parser.add_argument("--out_dir_path", type=str, required=True,
#                     help="Path to the output directory.")

args = parser.parse_args()

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

save_path = os.path.join(out_dir_path, "metrics")

if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path)
    print(f"Output directory '{out_dir_path}' created.")

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Output metrics directory '{save_path}' created.")

go_dag = GODag("go.obo")

def contruct_go_tree_nx(go_dag):
    print("Constructing Gene Ontology DAGs")
    subgraphs = {
        'biological_process': nx.DiGraph(),
        'molecular_function': nx.DiGraph(),
        'cellular_component': nx.DiGraph()
    }
    for go_id, term in go_dag.items():
        if term.namespace in subgraphs:
            subgraphs[term.namespace].add_node(term.id, name=term.name, namespace=term.namespace)
            for parent in term.parents:
                subgraphs[term.namespace].add_edge(parent.id, term.id)


    return subgraphs

go_trees = contruct_go_tree_nx(go_dag)

for type,tree in go_trees.items():
    print(type, len(tree.nodes),len(tree.edges), [len(cc) for cc in nx.connected_components(tree.to_undirected())])
    
gaf_reader = get_objanno("goa_human.gaf")
go_annotations = gaf_reader.get_id2gos(namespace="bp")                  # change namespace for bp, mf, cc
metadata = pd.read_csv(metadata_path)
# metadata["go_biological"] = metadata["go_biological"].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) and ast.literal_eval(x) else None)
# metadata["go_cellular"] = metadata["go_cellular"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and ast.literal_eval(x) else None)
# metadata["go_molecular"] = metadata["go_molecular"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and ast.literal_eval(x) else None)
print(metadata["go_ids"][0] )
# metadata["go_ids"] = metadata["go_ids"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and ast.literal_eval(x) else None)
metadata["go_ids"] = metadata["go_ids"].apply(lambda x: x.split(",") if isinstance(x, str) else None)

print(metadata.columns)

goe_allcsv_path = os.path.join(out_dir_path,"goe_per_feature.csv")
if os.path.exists(goe_allcsv_path):
    goe = pd.read_csv(goe_allcsv_path, index_col=None)
else:
    goe = pd.concat([pd.read_csv(file) for file in glob.glob(os.path.join(out_dir_path,"goe","feature*.csv"))], ignore_index=True)
    goe.to_csv(goe_allcsv_path)
    
goe["-log(p)"] = (-np.log10(goe["pvalues"])).clip(upper=50)
goe["namespace"] = goe["Term"].apply(lambda x: go_dag[x].namespace)
goe_heatmap = goe.pivot(index='Term', columns='Feature', values='-log(p)')
goe_heatmap.fillna(0, inplace=True)


features = goe_heatmap.columns
analyses_columns = ["Feature","Namespace", "Term 1","Term 2",  "Harmonic pvalue", "Shortest Path", "Shortest Path Length", "Weighted Shortest Path Length", "LCA", "LCA Depth", "Weighted LCA Depth", "1 to LCA dist", "2 to LCA dist"]

def random_pairwise_analyses(go_terms,neglogp, type, random_type):
    print(f"Random baseline for {type}")
    types = {"biological_process":"bp", "molecular_function":"mf","cellular_component":"cc"}
    type_short = types[type]
    analyses_per_termset = pd.DataFrame(columns=analyses_columns)
    tree_directed = go_trees[type]
    tree_undirected = tree_directed.to_undirected()
    for i in range(len(go_terms)):
        for j in range(i+1, len(go_terms)):
            term1 = go_terms[i]
            term2 = go_terms[j]
            harmmean = harmonic_mean(neglogp[i],neglogp[j])  if neglogp is not None else None
            path = nx.shortest_path(tree_undirected,term1,term2)
            path_length = len(path)
            lca_name = lca(tree_directed,term1, term2)
            lca_to_1 = len(nx.shortest_path(tree_undirected,term1,lca_name))
            lca_to_2 = len(nx.shortest_path(tree_undirected,term2,lca_name))
            analyses_per_termset = pd.concat([analyses_per_termset,pd.DataFrame({
                "Namespace":[type],
                "Term 1": [term1], "Term 2":[term2],
                "Harmonic pvalue":[harmmean],
                "Shortest Path":[path], "Shortest Path Length": [path_length],
                "Weighted Shortest Path Length":[path_length / harmmean] if harmmean is not None else [None],
                "LCA": [lca_name], "LCA Depth": [None] if lca_name is None else [go_dag[lca_name].depth],
                "Weighted LCA Depth":[go_dag[lca_name].depth * harmmean] if harmmean is not None else [None],
                "1 to LCA dist": [lca_to_1], "2 to LCA dist":[lca_to_2],
            })], ignore_index=True)
    analyses_per_termset.to_csv(os.path.join(out_dir_path,f"random_analyses_{type_short}_{random_type}.csv"),index=None)
    return analyses_per_termset


def pairwise_comparisons(feature, type):
    go_terms,neglogp = goe[(goe["Feature"]==feature) & (goe["namespace"]==type)][["Term","-log(p)"]].T.to_numpy()
    analyses_per_termset = pd.DataFrame(columns=analyses_columns)
    if len(go_terms) > 0:
        tree_directed = go_trees[type]
        tree_undirected = tree_directed.to_undirected()
        for i in range(len(go_terms)):
            for j in range(i+1, len(go_terms)):
                term1 = go_terms[i]
                term2 = go_terms[j]
                harmmean = harmonic_mean(neglogp[i],neglogp[j])
                path = nx.shortest_path(tree_undirected,term1,term2)
                path_length = len(path)
                lca_name = lca(tree_directed,term1, term2)
                lca_to_1 = len(nx.shortest_path(tree_undirected,term1,lca_name))
                lca_to_2 = len(nx.shortest_path(tree_undirected,term2,lca_name))
                analyses_per_termset = pd.concat([analyses_per_termset,pd.DataFrame({
                    "Feature":[feature],"Namespace":[type],
                    "Term 1": [term1], "Term 2":[term2],
                    "Harmonic pvalue":[harmmean],
                    "Shortest Path":[path], "Shortest Path Length": [path_length],
                    "Weighted Shortest Path Length":[path_length / harmmean],
                    "LCA": [lca_name], "LCA Depth": [None] if lca_name is None else [go_dag[lca_name].depth],
                    "Weighted LCA Depth":[go_dag[lca_name].depth * harmmean],
                    "1 to LCA dist": [lca_to_1], "2 to LCA dist":[lca_to_2],
                })], ignore_index=True)
    return analyses_per_termset


def harmonic_mean(a,b):
    return 2/(1/a+1/b)
            
def lca(graph, node1, node2):
    ancestors1 = nx.ancestors(graph, node1) | {node1}
    ancestors2 = nx.ancestors(graph, node2) | {node2}
    
    common_ancestors = ancestors1 & ancestors2
    if not common_ancestors:
        return None
    
    lca = max(common_ancestors, key=lambda node: go_dag[node].depth)
    return lca


def analyses_for_one_feature(feature_idx):
    analyses_df = pd.concat([pairwise_comparisons(feature_idx, "biological_process"), 
                           pairwise_comparisons(feature_idx, "molecular_function"), 
                           pairwise_comparisons(feature_idx, "cellular_component"),], 
                          ignore_index=True)
    analyses_df.to_csv(os.path.join(save_path,f"feature{feature_idx}_metrics.csv"), index=None)


# Random samples analysis on constrained dag
all_constrained_gos = pd.concat([pd.Series(lis) for lis in metadata["go_ids"]]).drop_duplicates()
bpdf = pd.Series([go_dag[term].id for term in all_constrained_gos if term in go_dag and go_dag[term].namespace == "biological_process"]).drop_duplicates().sample(n=150, random_state=42)
mfdf = pd.Series([go_dag[term].id for term in all_constrained_gos if term in go_dag and go_dag[term].namespace == "molecular_function"]).drop_duplicates().sample(n=150, random_state=42)
ccdf = pd.Series([go_dag[term].id for term in all_constrained_gos if  term in go_dag and go_dag[term].namespace == "cellular_component"]).drop_duplicates().sample(n=150, random_state=42)

random_pairwise_analyses(bpdf.tolist(),None, "biological_process", "constrained_dag")
random_pairwise_analyses(mfdf.tolist(),None, "molecular_function", "constrained_dag")
random_pairwise_analyses(ccdf.tolist(),None, "cellular_component", "constrained_dag")

# Random samples analysis on entire dag
bpdf = pd.Series([go_dag[term].id for term in go_dag.keys() if go_dag[term].namespace == "biological_process"]).drop_duplicates().sample(n=150, random_state=42)
mfdf = pd.Series([go_dag[term].id for term in go_dag.keys() if go_dag[term].namespace == "molecular_function"]).drop_duplicates().sample(n=150, random_state=42)
ccdf = pd.Series([go_dag[term].id for term in go_dag.keys() if go_dag[term].namespace == "cellular_component"]).drop_duplicates().sample(n=150, random_state=42)

random_pairwise_analyses(bpdf.tolist(),None, "biological_process", "entire_dag")
random_pairwise_analyses(mfdf.tolist(),None, "molecular_function", "entire_dag")
random_pairwise_analyses(ccdf.tolist(),None, "cellular_component", "entire_dag")


# Real samples analysis
with Progress() as progress:
    task = progress.add_task("Analyzing Feature GO sets...", total=len(features))
    
    with mp.Pool(mp.cpu_count()) as pool:
        for feature_idx in pool.imap_unordered(analyses_for_one_feature, features):
            progress.update(task, advance=1)

print("Analysis complete. All CSVs have been saved.")

