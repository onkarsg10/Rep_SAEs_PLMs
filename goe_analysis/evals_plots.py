import numpy as np
import pandas as pd
import h5py
import umap
import plotly.express as px
import plotly.graph_objects as go
import ast
import ipywidgets as widgets
from IPython.display import display
import notebook
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


parser = argparse.ArgumentParser(description="Plotting script")
# parser.add_argument("--metadata_path", type=str, required=True,
#                     help="Path to the metadata file.")
# parser.add_argument("--out_dir_path", type=str, required=True,
#                     help="Path to the output directory.")
parser.add_argument("--input_dir_path", type=str, required=True,
                    help="Path to the output directory.")

args = parser.parse_args()

# metadata_path = args.metadata_path
# out_dir_path = args.out_dir_path
out_dir_path = os.path.join(args.input_dir_path, "out")

save_path = os.path.join(out_dir_path, "plots")

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Output plotting directory '{save_path}' created.")


# metadata = pd.read_csv(metadata_path)
# metadata["go_bp"] = metadata["go_biological"].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) and ast.literal_eval(x) else None)
# metadata["go_cc"] = metadata["go_cellular"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and ast.literal_eval(x) else None)
# metadata["go_mf"] = metadata["go_molecular"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and ast.literal_eval(x) else None)
# metadata["go_ids"] = metadata["go_ids"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and ast.literal_eval(x) else [])


if not os.path.exists(os.path.join(out_dir_path,"pairwise_analyses_bp.csv")):
    analyses_per_feature = pd.concat([pd.read_csv(file) for file in glob.glob(os.path.join(out_dir_path,"metrics","feature*.csv"))], ignore_index=True)
    analyses_per_feature.to_csv(os.path.join(out_dir_path,"pairwise_analyses_all.csv"),index=None)
    analyses_per_feature[analyses_per_feature["Namespace"] == "cellular_component"].to_csv(os.path.join(out_dir_path,"pairwise_analyses_cc.csv"),index=None)
    analyses_per_feature[analyses_per_feature["Namespace"] == "biological_process"].to_csv(os.path.join(out_dir_path,"pairwise_analyses_bp.csv"),index=None)
    analyses_per_feature[analyses_per_feature["Namespace"] == "molecular_function"].to_csv(os.path.join(out_dir_path,"pairwise_analyses_mf.csv"),index=None)
    del analyses_per_feature


def plot_sp_lca(df, weighted=False, random=False):
    prefix = "Weighted " if weighted else ""
    rand = "(Random) " if random else ""
    fig = px.scatter(
        df,
        x=f"{prefix}LCA Depth", 
        y=f"{prefix}Shortest Path Length",
        hover_data=["Feature"], 
        marginal_x="histogram", 
        marginal_y="histogram",
        title=f"{rand}Mean SP vs. Mean LCA Depth ({type.upper()})"
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40), 
        height=700,
        width=800
        # xaxis=dict(showgrid=True, zeroline=True),
        # yaxis=dict(showgrid=True, zeroline=True),
    )
    return fig


from plotly.subplots import make_subplots


import scipy.stats as stats
import statsmodels.stats.multitest as smm
import plotly.graph_objects as go

def plot_combined_w_signficance(analyses_per_feature, random_analyses_df, type):
    """
    Creates a combined plot with subplots (joint scatter plot and paired box plot) for each type.
    Includes significance bars on the paired box plot.
    """
    # Joint scatter plot
    joint_plot = px.scatter(
        analyses_per_feature,
        x="LCA Depth",
        y="Shortest Path Length",
        hover_data=["Feature"],
        marginal_x="histogram",
        marginal_y="histogram",
        title=f"Mean SP vs. Mean LCA Depth ({type.upper()})",
        color_discrete_sequence=["#fc035e"]
    )

    joint_plot.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
    joint_plot.update_layout(
        height=700,
        width=800,
        template="plotly_white",
        font=dict(color="black"),
        title=dict(font=dict(size=25)),
        xaxis=dict(
            title_font=dict(color="black", size=20),
            tickfont=dict(color="black"),
            linecolor="black",
            showline=True,
        ),
        yaxis=dict(
            title_font=dict(color="black", size=20),
            tickfont=dict(color="black"),
            linecolor="black",
            showline=True,
        ),
    )

    # Bin data for box plot
    bins = np.arange(analyses_per_feature["LCA Depth"].max() + 1).astype(int)
    labels = [f"({bins[i]}-{bins[i+1]}]" for i in range(len(bins) - 1)]
    analyses_per_feature["LCA Depth Bin"] = pd.cut(analyses_per_feature["LCA Depth"], bins=bins, labels=labels, right=False)
    random_analyses_df["LCA Depth Bin"] = pd.cut(random_analyses_df["LCA Depth"], bins=bins, labels=labels, right=False)

    real_data = analyses_per_feature[["Feature", "LCA Depth Bin", "Shortest Path Length"]].copy()
    real_data["Source"] = "Real"

    random_data = random_analyses_df[["Feature", "LCA Depth Bin", "Shortest Path Length"]].copy()
    random_data["Source"] = "Random"

    combined_data = pd.concat([real_data, random_data]).sort_values(by=["LCA Depth Bin", "Source"])

    # Create the paired box plot
    traces = []
    annotations = []
    source_colors = {"Random": "#029cbf", "Real": "#fc035e"}
    bins = combined_data["LCA Depth Bin"].cat.categories

    for source, color in source_colors.items():
        subset = combined_data[combined_data["Source"] == source]
        traces.append(
            go.Box(
                x=subset["LCA Depth Bin"],
                y=subset["Shortest Path Length"],
                name=f"{type.upper()} {source}",
                marker=dict(color=color),
                # boxpoints=False,
                # boxmean="sd",
                notched=True,
                offsetgroup=source,
            )
        )

    for i, bin_label in enumerate(bins):
        real_values = combined_data[(combined_data["Source"] == "Real") & (combined_data["LCA Depth Bin"] == bin_label)]["Shortest Path Length"]
        random_values = combined_data[(combined_data["Source"] == "Random") & (combined_data["LCA Depth Bin"] == bin_label)]["Shortest Path Length"]
        print(real_values)
        print(random_values)
        if len(real_values) > 0 and len(random_values) > 0:
            t_stat, p_value = stats.ttest_ind(real_values, random_values, equal_var=False)
            _, corrected_p_value, _, _ = smm.multipletests([p_value], method="fdr_bh")

            if corrected_p_value[0] <= 0.05:
                text = "*"
            if corrected_p_value[0] <= 0.001:
                text = "**"
            if corrected_p_value[0] <= 0.0001:
                text = "***"

            if corrected_p_value[0] <= 0.05:
                y_max = max(real_values.max(), random_values.max())
                annotations.append(
                    dict(
                        x=bin_label,
                        y=y_max + 1,  # Position above the max value
                        text=text,
                        showarrow=False,
                        font=dict(size=16, color="black"),
                    )
                )


    # Add traces to the figure
    paired_box_plot = go.Figure(data=traces)
    paired_box_plot.update_layout(
        title=f"{type.upper()} Paired Boxplot of SP Lengths over LCA Depths",
        height=700,
        width=1000,
        template="plotly_white",
        font=dict(color="black"),
        # title=dict(font=dict(size=25)),
        xaxis=dict(
            title="LCA Depth Bin (GO DAG level)",
            title_font=dict(color="black", size=20),
            tickfont=dict(color="black"),
            linecolor="black",
            showline=True,
        ),
        yaxis=dict(
            title="Shortest Path Lengths (Monosemanticity)",
            title_font=dict(color="black", size=20),
            tickfont=dict(color="black"),
            linecolor="black",
            showline=True,
        ),
        boxmode='group',
        annotations=annotations,  # Add significance annotations
    )

    return joint_plot, paired_box_plot


def plot_combined(analyses_per_feature, random_analyses_df, type):
    """
    Creates a combined plot with subplots (joint scatter plot and paired box plot) for each type.
    """
    # Joint scatter plot
    joint_plot = px.scatter(
        analyses_per_feature,
        x="LCA Depth",
        y="Shortest Path Length",
        hover_data=["Feature"],
        marginal_x="histogram",
        marginal_y="histogram",
        title=f"Mean SP vs. Mean LCA Depth ({type.upper()})",
    )

    joint_plot.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
    joint_plot.update_layout(
        # margin=dict(l=40, r=40, t=10, b=10),
        height=700,
        width=800,
        template="plotly_white",
        font=dict(color="black"),
        title=dict(font=dict(size=25)),
        xaxis=dict(title_font=dict(color="black",size=20), tickfont=dict(color="black"), linecolor="black", showline=True),
        yaxis=dict(title_font=dict(color="black",size=20), tickfont=dict(color="black"), linecolor="black", showline=True),
    )

    # Bin data for box plot
    bins = np.arange(analyses_per_feature['LCA Depth'].max() + 1).astype(int)
    labels = [f"({bins[i]}-{bins[i+1]}]" for i in range(len(bins)-1)] 
    analyses_per_feature['LCA Depth Bin'] = pd.cut(analyses_per_feature['LCA Depth'], bins=bins, labels=labels, right=False)
    random_analyses_df['LCA Depth Bin'] = pd.cut(random_analyses_df['LCA Depth'], bins=bins, labels=labels, right=False)

    real_data = analyses_per_feature[['Feature', 'LCA Depth Bin', 'Shortest Path Length']].copy()
    real_data['Source'] = 'Real'

    random_data = random_analyses_df[['Feature', 'LCA Depth Bin', 'Shortest Path Length']].copy()
    random_data['Source'] = 'Random'

    combined_data = pd.concat([real_data, random_data]).sort_values(by=['LCA Depth Bin', 'Source'])

    paired_box_plot = px.box(
        combined_data,
        x="LCA Depth Bin",
        y="Shortest Path Length",
        color="Source",
        labels={
            "LCA Depth Bin": "LCA Depth Bin (GO DAG level)",
            "Shortest Path Length": "Shortest Path Lengths (Monosemanticity)",
            "Source": "Data Type",
        },
        title=f"{type.upper()} Paired Boxplot of SP Lengths over LCA Depths",
        # points=True,  # Hide individual points
        boxmode="group"  # Group boxplots side by side
    )

    paired_box_plot.update_layout(
        # margin=dict(l=40, r=40, t=40, b=40),
        height=700,
        width=1000,
        template="plotly_white",
        font=dict(color="black"),
        title=dict(font=dict(size=25)),
        xaxis=dict(title_font=dict(color="black",size=20), tickfont=dict(color="black"), linecolor="black", showline=True),
        yaxis=dict(title_font=dict(color="black",size=20), tickfont=dict(color="black"), linecolor="black", showline=True),
        # boxgroupgap=0.4,  
    )

    return joint_plot, paired_box_plot


# Creating the combined figure
# fig = make_subplots(
#     rows=3, cols=2,
#     # subplot_titles=[
#     #     "BP Shortest Path vs. LCA Depth", "BP GrBoxplot",
#     #     "MF Shortest Path vs. LCA Depth", "MF Boxplot",
#     #     "CC Shortest Path vs. LCA Depth", "CC Boxplot",
#     # ],
#     horizontal_spacing=0.15, vertical_spacing=0.1
# )

for random_type in ["entire","constrained"]:
    for idx, type in enumerate(["bp", "mf", "cc"]):
    # for idx, type in enumerate(["bp"]):
        print(f"Plotting jointplot/grouped boxplots for {type}")
        analyses_per_feature = pd.read_csv(os.path.join(out_dir_path, f"pairwise_analyses_{type}.csv")).groupby("Feature").agg({
            "Feature": 'first',
            "LCA Depth": "mean",
            "Shortest Path Length": "mean",
        })

        random_analyses_df = pd.read_csv(os.path.join(out_dir_path, f"random_analyses_{type}_{random_type}_dag.csv"))

        joint_plot, paired_box_plot = plot_combined_w_signficance(analyses_per_feature, random_analyses_df, type)
        # paired_box_plot.write_image(os.path.join(save_path, "test.png"))

        # # Adding subplots
        # fig.add_trace(joint_plot.data[0], row=idx + 1, col=1)  # Jointplot scatter
        # fig.add_traces(paired_box_plot.data, rows=idx + 1, cols=2)  # Paired boxplot
        # i=idx+1
        # for trace in joint_plot["data"]:
        #     fig.add_trace(trace, row=i, col=1)

        #     # Add the traces from paired_box_plot
        # for trace in paired_box_plot["data"]:
        #     fig.add_trace(trace, row=i, col=2)

        # fig.update_xaxes(joint_plot.layout.xaxis, row=i, col=1)
        # fig.update_yaxes(joint_plot.layout.yaxis, row=i, col=1)
        # fig.update_xaxes(paired_box_plot.layout.xaxis, row=i, col=2)
        joint_plot.write_image(os.path.join(save_path, f"jointplot_{type}_{random_type}_dag.png"))
        joint_plot.write_image(os.path.join(save_path, f"jointplot_{type}_{random_type}_dag.svg"))
        paired_box_plot.write_image(os.path.join(save_path, f"groupplot_{type}_{random_type}_dag.png"))
        paired_box_plot.write_image(os.path.join(save_path, f"groupplot_{type}_{random_type}_dag.svg"))
        
# Update layout for combined figure
# fig.update_layout(
#     height=1200,
#     width=1400,
#     showlegend=True,
#     title_text="GO Analysis Results: Jointplots and Boxplots",
#     template="plotly_white",
#     font=dict(color="black"),
# )
# fig.update_traces(col=2,margin=dict(l=10, r=10, t=10, b=10),
#         template="plotly_white",
#         font=dict(color="black"),
#         xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"), linecolor="black", showline=True),
#         yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"), linecolor="black", showline=True),
#         boxgroupgap=0.4,boxmode="group")

# # Save the figure as high-quality SVG/PDF
# fig.write_image(os.path.join(save_path, "combined_plot.png"))
# fig.write_image(os.path.join(save_path, "combined_plot.svg"))
# fig.write_image(os.path.join(save_path, "combined_plot.pdf"))



# for type in ["bp","mf","cc"]:
#     analyses_per_feature = pd.read_csv(os.path.join(out_dir_path,f"pairwise_analyses_{type}.csv"), index_col=None).groupby("Feature").agg({
#         "Feature":'first',
#         "LCA Depth": "mean",
#         "Weighted LCA Depth":"mean",
#         "Shortest Path Length":"mean",
#         "Weighted Shortest Path Length":"mean",
#         "1 to LCA dist": "mean",
#         "2 to LCA dist": "mean",
#     })
#     print(f"Loaded {type} analyses...")

#     plot_sp_lca(analyses_per_feature, random=False).write_image(os.path.join(save_path,f"sp_lca_jointplot_{type}.png")) 

#     bins = np.arange(analyses_per_feature['LCA Depth'].max()+1).astype(int)
#     labels = [f"({bins[i]}-{bins[i+1]}]" for i in range(len(bins)-1)] 
#     analyses_per_feature['LCA Depth Bin'] = pd.cut(analyses_per_feature['LCA Depth'], bins=bins, labels=labels, right=False)
    
#     for random_type in ["constrained_dag","entire_dag"]:
#         random_analyses_df = pd.read_csv(os.path.join(out_dir_path,f"random_analyses_{type}_{random_type}.csv"), index_col=None)

#         plot_sp_lca(random_analyses_df, random=True).write_image(os.path.join(save_path,f"random_sp_lca_jointplot_{type}_{random_type}.png"))
         
#         random_analyses_df['LCA Depth Bin'] = pd.cut(random_analyses_df['LCA Depth'], bins=bins, labels=labels, right=False)

#         real_data = analyses_per_feature[['Feature', 'LCA Depth Bin', 'Shortest Path Length']].copy()
#         real_data['Source'] = 'Real' 

#         random_data = random_analyses_df[['Feature', 'LCA Depth Bin', 'Shortest Path Length']].copy()
#         random_data['Source'] = 'Random' 

#         combined_data = pd.concat([real_data, random_data])

#         combined_data = combined_data.sort_values(by=['LCA Depth Bin', 'Source'])
#         fig = px.box(combined_data,
#                     x="LCA Depth Bin",           
#                     y="Shortest Path Length",    
#                     color="Source",             
#                     labels={"LCA Depth Bin": "LCA Depth Bin (represents level of GO DAG)", "Shortest Path Length": "Shortest Path Lengths (represents monosemanticity)", "Source": "Data Type"},
#                     title=f"{type.upper()} Paired Boxplot of Shortest Path Lengths over all LCA Depths")
        
#         fig.write_image(os.path.join(save_path,f"sp_boxplots_{type}_{random_type}.png"))