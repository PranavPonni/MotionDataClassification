Here’s a `README.md` for your force command classification script:

---

# Force Command Classification Script

This Python script processes force command data from a CSV file, classifying movements and generating new labels for the data based on specific thresholds for the x, y, and z force commands. It also includes feature engineering to create binary labels and combined force analysis.

## Features

- **Input**: A CSV file containing force command data for the x, y, and z axes.
- **Output**: A new CSV file that includes additional labeled columns:
  - `x force label`: Classification of x-axis force commands.
  - `y force label`: Classification of y-axis force commands.
  - `z force label`: Classification of z-axis force commands.
  - `swiping in motion`: Indicates whether swiping is happening based on force commands in all three axes.
  - `swiping_binary`: A binary label (1 for swiping, 0 for no swiping).
  - `combined force label`: A custom label based on combinations of force thresholds (e.g., high x, low z).

## Prerequisites

Ensure you have Python 3.x installed and the following libraries:

- Pandas

You can install the necessary libraries using:

```bash
pip install pandas
```

## Usage

1. **Input CSV File**: Place your CSV file with force command data in a known location.
2. **Configure Paths**: Update the `input_file` and `output_file` variables in the script to point to your input file and desired output location.
3. **Run the Script**: Execute the script to generate the classified output CSV file with the newly labeled data.

### Example Command:

```bash
python classify_force_commands.py
```

## Output

The output CSV will contain the following columns:
- `x force command`, `y force command`, `z force command`: Original force commands from the input file.
- `x force label`, `y force label`, `z force label`: Labels classifying the force commands.
- `swiping in motion`: Label indicating if swiping occurred based on all three axes.
- `swiping_binary`: A binary indicator for swiping (1 for swiping, 0 for no swiping).
- `combined force label`: A custom label based on combinations of force thresholds.

## Performance

The script will print the time taken to classify and save the data after processing.

--- 

This file should guide users on how to execute the script and understand its functionality.
