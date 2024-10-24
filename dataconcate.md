Combine Multiple CSV Files into a Single File

This script merges all CSV files from a given folder into a single DataFrame and exports the combined data to a new CSV file. It's useful for scenarios where you have several CSV files with similar structures and need to process them as a single dataset.

Requirements

Before you run the script, make sure you have the following dependencies installed:

Python 3.x
Pandas (for data manipulation)
To install Pandas, use the following command:

bash
Copy code
pip install pandas
Script Overview

Functionality
Input: A folder containing CSV files.
Output: A new CSV file that consolidates all the individual CSV files in the folder.
The script uses Python’s glob module to find all .csv files in the directory.
pandas is used to read and concatenate the data.
The final merged data is saved as a single CSV file.
