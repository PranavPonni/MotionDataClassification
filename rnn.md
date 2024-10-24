Here’s a `README.md` for your RNN model script:

---

# RNN for Swiping Detection (Force Command Classification)

This Python script uses a Recurrent Neural Network (RNN) implemented in PyTorch to detect whether a robot's motion involves "swiping over an obstacle" based on x, y, and z force command data. The script preprocesses the data, applies SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance, trains an RNN model, and evaluates its performance.

## Features

- **Swiping Classification**: Predicts whether a motion event is classified as "swiping over an obstacle" (`1`) or "no swiping" (`0`).
- **Class Imbalance Handling**: Uses SMOTE to balance the training dataset for swiping and no swiping classes.
- **Model Performance Metrics**: Outputs accuracy, classification report, and confusion matrix.
- **Predictions Export**: Saves test set predictions along with true labels to a CSV file.

## Prerequisites

Before running the script, ensure that Python 3.x and the following libraries are installed:

- pandas
- numpy
- PyTorch
- scikit-learn
- imbalanced-learn
- os
- time

You can install the required packages using the following:

```bash
pip install pandas numpy torch scikit-learn imbalanced-learn
```

## Data Structure

The script expects CSV files in the specified training and testing folders. Each CSV file should include the following columns:

- `x force command`: Force data along the x-axis.
- `y force command`: Force data along the y-axis.
- `z force command`: Force data along the z-axis.

### Labels:

- The script applies feature engineering to generate the binary target column (`swiping_binary`), where:
  - `1` indicates "swiping over obstacle"
  - `0` indicates "no swiping"

The data files are loaded and processed automatically from the specified `train_folder` and `test_folder`.

## How to Run

1. **Set Data Paths**:
   Update the following variables in the script with your directory paths:
   - `train_folder`: Path to the folder containing the training data CSV files.
   - `test_folder`: Path to the folder containing the testing data CSV files.

2. **Run the Script**:
   Execute the script to train the RNN model and evaluate its performance.

### Example Command:

```bash
python rnn_swiping_detection.py
```

## Main Steps

1. **Data Loading and Preprocessing**:
   - Loads and concatenates CSV files from the train and test folders.
   - Applies feature engineering to classify force commands into movement labels and generate the target column (`swiping_binary`).
   - The data is normalized using `StandardScaler` before feeding it to the model.

2. **SMOTE for Class Imbalance**:
   - Balances the training data using SMOTE to account for class imbalance between swiping and no swiping events.

3. **RNN Model Architecture**:
   - A Recurrent Neural Network with 2 layers, each containing 128 hidden units.
   - The model takes force command data (`x`, `y`, `z`) as input and outputs two classes (swiping and no swiping).

4. **Training the RNN Model**:
   - The model is trained using the Adam optimizer and Cross Entropy Loss for 1000 epochs.

5. **Model Evaluation**:
   - The model makes predictions on the test data and outputs the following metrics:
     - **Accuracy**: Proportion of correct predictions.
     - **Classification Report**: Precision, recall, f1-score, and support for both classes.
     - **Confusion Matrix**: Breakdown of true positives, true negatives, false positives, and false negatives.

6. **Saving Model and Predictions**:
   - The trained RNN model is saved to `rnn_model.pth`.
   - The predictions are saved to a CSV file containing the true and predicted labels for the test data.

## Hyperparameters

The script uses the following hyperparameters:

- **Input Size**: 3 (for x, y, z force commands)
- **Hidden Size**: 128
- **Output Size**: 2 (swiping or no swiping)
- **Number of Layers**: 2
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Number of Epochs**: 1000

You can adjust these values in the script to experiment with different configurations.

## Output

The script generates the following outputs:

- **Accuracy**: Overall accuracy of the RNN model on the test set.
- **Classification Report**: Detailed performance metrics for each class (swiping/no swiping).
- **Confusion Matrix**: Summary of prediction results.

### Output Files:

- **Model File**: The trained RNN model is saved to `rnn_model.pth`.
- **Predictions CSV**: The test set predictions, along with true labels and force commands, are saved to `predictions_rnn.csv`.

## Customization

- **Model Architecture**: You can modify the RNN architecture, such as the number of layers or hidden units.
- **Model Hyperparameters**: Fine-tune the learning rate, batch size, or number of epochs to improve performance.

---

This README provides a guide to setting up, running, and interpreting the outputs of your RNN model for swiping detection using force command data.
