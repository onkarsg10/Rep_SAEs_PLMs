import subprocess

input_dir_paths = [
    "./Minotaur_Final_Reproducible_Mihir_Extracted_Data/regular_layer7",      # running on dragon
    "./Minotaur_Final_Reproducible_Mihir_Extracted_Data/regular_layer8",
    "./Minotaur_Final_Reproducible_Mihir_Extracted_Data/regular_layer9",
    "./Minotaur_Final_Reproducible_Mihir_Extracted_Data/regular_layer10",
    "./Minotaur_Final_Reproducible_Mihir_Extracted_Data/TC_layer7",   # running on uni
    "./Minotaur_Final_Reproducible_Mihir_Extracted_Data/TC_layer8",
    "./Minotaur_Final_Reproducible_Mihir_Extracted_Data/TC_layer9",     # running on kyogre
    "./Minotaur_Final_Reproducible_Mihir_Extracted_Data/TC_layer10",  
    "./Dementor_Final_Reproducible_Mihir_Extracted_Data/regular_layer6",        
    "./Dementor_Final_Reproducible_Mihir_Extracted_Data/TC_layer6",
]

# script_path = "evals_goe.py"
# script_path = "evals_goe_pairwise_analysis.py"
script_path = "evals_plots.py"

for input_dir in input_dir_paths:
    try:
        print(f"Running {script_path} for input_dir_path: {input_dir}")
        subprocess.run(
            ["python", script_path, "--input_dir_path", input_dir],
            check=True,
        )
        print(f"Finished processing: {input_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while processing {input_dir}: {e}")
