Here’s a `README.md` for your data visualization script:

---

# Force Command Time Series Visualization with Swipe Annotations

This script visualizes time series data for x, y, and z force commands from a CSV file, with annotated regions indicating whether swiping is in motion. The output is a plot with color-coded force commands and swiping/no-swiping regions.

## Features

- **Visualize Force Commands**: Plots the x, y, and z force commands over time.
- **Swipe Annotation**: Highlights regions where swiping is detected in yellow, and no-swiping regions in gray.
- **Custom Legend**: Includes a legend showing the force commands and swiping annotations for clear interpretation.

## Prerequisites

Before running this script, make sure you have the following dependencies installed:

- Python 3.x
- Pandas (for data handling)
- Matplotlib (for plotting)

To install the required libraries, use:

```bash
pip install pandas matplotlib
```

## Usage

1. **Input CSV File**: Make sure the CSV file containing force commands and swiping annotations is available.
2. **Configure File Path**: Update the `csv_file_path` variable in the script to point to the location of your CSV file.
3. **Run the Script**: Execute the script to visualize the data.

### Example Command:

```bash
python visualize_force_commands.py
```

## Input Data

The input CSV file should contain the following columns:
- `x force command`: Force command data for the x-axis.
- `y force command`: Force command data for the y-axis.
- `z force command`: Force command data for the z-axis.
- `swiping in motion`: Annotation of whether swiping is occurring (`swiping` or `no swiping`).

## Output

The script generates a time series plot that includes:
- A line plot for the x, y, and z force commands.
- Highlighted regions showing where swiping is in motion (yellow) and where it is not (gray).

## Customization

You can adjust the following in the script:
- **Colors**: Change the colors for force commands or swiping regions in the `plt.plot()` and `plt.axvspan()` functions.
- **Labels and Title**: Modify axis labels and the plot title in the `plt.xlabel()`, `plt.ylabel()`, and `plt.title()` functions.

## Example Plot

The plot will show how force commands change over time, with swiping events clearly marked for easier analysis.

---

This README should help users understand how to use the script and interpret the plot it generates.
