Here’s a `README.md` for your classification script without code explanation:

---

# Force Command Classification

This script processes force command data from a CSV file and classifies the movements into different labels based on specified criteria. It also identifies when swiping is occurring based on the combined classification of the x, y, and z force commands.

## Features

- **Input**: A CSV file containing force command data for the x, y, and z axes.
- **Output**: A new CSV file with the following additional columns:
  - `x force label`: Classification of the x-axis force command.
  - `y force label`: Classification of the y-axis force command.
  - `z force label`: Classification of the z-axis force command.
  - `swiping in motion`: Indicates if swiping is happening based on the force commands.
  
## Prerequisites

To run this script, you will need:

- Python 3.x
- Pandas library

You can install Pandas with the following command:

```bash
pip install pandas
```

## Usage

1. Place your input CSV file containing the force command data in a known location.
2. Set the `input_file` and `output_file` variables with the paths to your input and desired output files.
3. Run the script to classify the data and save the results to a new CSV file.

## Example

```bash
python classify_force_commands.py
```

## Performance

The script will print out the time it takes to process and classify the data once completed.

