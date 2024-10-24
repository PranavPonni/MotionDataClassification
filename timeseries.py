import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# File path to your CSV file
csv_file_path = # add own file path

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Create a figure and axis for plotting
plt.figure(figsize=(12, 8))

# Plot the force commands
plt.plot(df.index, df['x force command'], label='X Force Command', color='blue')
plt.plot(df.index, df['y force command'], label='Y Force Command', color='green')
plt.plot(df.index, df['z force command'], label='Z Force Command', color='red')

# Highlight regions based on 'swiping in motion'
for i in range(len(df)):
    if df['swiping in motion'][i] == 'swiping':
        plt.axvspan(i - 0.5, i + 0.5, color='yellow', alpha=0.3)
    else:
        plt.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.3)

# Create custom legend handles for swiping regions
legend_elements = [
    Line2D([0], [0], color='yellow', alpha=0.3, lw=6, label='Swiping'),
    Line2D([0], [0], color='gray', alpha=0.3, lw=6, label='No Swiping'),
    Line2D([0], [0], color='blue', lw=2, label='X Force Command'),
    Line2D([0], [0], color='green', lw=2, label='Y Force Command'),
    Line2D([0], [0], color='red', lw=2, label='Z Force Command')
]

# Adding labels and title
plt.xlabel('Time (Index)')
plt.ylabel('Force Command')
plt.title('Time Series of Force Commands with Swipe Annotations')
plt.legend(handles=legend_elements)

# Show the plot
plt.show()
