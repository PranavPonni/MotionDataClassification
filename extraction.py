import pandas as pd
import time

def classify_force_commands(input_file, output_file):
    start_time = time.time()  # Start the timer

    # Load the CSV file
    data = pd.read_csv(input_file)

    # Define the classification criteria
    def classify_x_force(x):
        if (3 < x < 20) or (-20 < x < -3):
            return "Swiping over obstacle"
        return "Movement in progress"

    def classify_y_force(y):
        if -1.4 < y < -0.4:
            return "Swiping over obstacle"
        return "Movement in progress"

    def classify_z_force(z):
        if -5.0 < z < -1.6:
            return "Swiping over obstacle"
        return "Movement in progress"

    # Apply the classification to the respective columns
    data['x force label'] = data['x force command'].apply(classify_x_force)
    data['y force label'] = data['y force command'].apply(classify_y_force)
    data['z force label'] = data['z force command'].apply(classify_z_force)

    # Define the condition for "swiping in motion"
    def classify_swiping_in_motion(row):
        if (row['x force label'] == "Swiping over obstacle" and
            row['y force label'] == "Swiping over obstacle" and
            row['z force label'] == "Swiping over obstacle"):
            return "swiping"
        return "no swiping"

    # Apply the condition to create the new column
    data['swiping in motion'] = data.apply(classify_swiping_in_motion, axis=1)

    # Select only the necessary columns for output
    output_data = data[['x force command', 'y force command', 'z force command', 
                        'x force label', 'y force label', 'z force label', 
                        'swiping in motion']]

    # Save the output to a new CSV file
    output_data.to_csv(output_file, index=False)

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"Classified data saved to {output_file} in {elapsed_time:.2f} seconds.")

# Example usage
input_file =   # Replace with your actual file path
output_file =   # Replace with desired output file name
classify_force_commands(input_file, output_file)
