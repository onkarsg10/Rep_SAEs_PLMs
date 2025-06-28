####Same binning - Regular

import datetime
import pandas as pd
import matplotlib.pyplot as plt

file_path1 = ''
file_path2 = ''

# Function to process data and extract 'correlation' column
def process_file(file_path):
    data = pd.read_csv(file_path)
    filtered_data = data.iloc[:200].copy()  # Ignore rows after row 200
    filtered_data.loc[:, 'correlation'] = pd.to_numeric(filtered_data['correlation'], errors='coerce')
    cleaned_data = filtered_data.dropna(subset=['correlation'])
    return cleaned_data['correlation']

# Process each file
correlations1 = process_file(file_path1)
correlations2 = process_file(file_path2)

# Print the counts of correlation values
print(f"Number of correlation values in file 1: {len(correlations1)}")
print(f"Number of correlation values in file 2: {len(correlations2)}")

# Plot histograms
plt.figure(figsize=(10, 6))

# Get common range for both histograms
min_val = min(correlations1.min(), correlations2.min())
max_val = max(correlations1.max(), correlations2.max())

plt.hist(correlations1, bins=30, alpha=0.5, label='Sparse (Layer 9)', edgecolor='black', range=(min_val, max_val))
plt.hist(correlations2, bins=30, alpha=0.5, label='ESM (Layer 9)', edgecolor='black', range=(min_val, max_val))

# Add legend and labels
plt.title('')
plt.xlabel('Pearson Correlation')
plt.ylabel('Neurons (total: 200 neurons evaluated)')
plt.legend()

# Save the figure
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# plt.savefig(f'Correlation_Histogram_Sparse_layer9_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'Correlation_Histogram_Sparse_layer9_{timestamp}.pdf', bbox_inches='tight')


plt.show()
