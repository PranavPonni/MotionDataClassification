Here’s a `README.md` for your logistic regression model script:

---

# Logistic Regression for Force Command Classification (Swiping Detection)

This Python script trains a logistic regression model to classify motion data as either swiping or no swiping, based on force command data for x, y, and z axes. It also handles class imbalance using SMOTE and provides model evaluation metrics.

## Features

- **Binary Classification**: Predicts whether swiping is in motion (1) or not (0) based on force command data.
- **Class Imbalance Handling**: Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data.
- **Performance Metrics**: Outputs model accuracy, classification report, and confusion matrix.
- **Model Coefficients**: Displays the trained model's coefficients for interpretation.
- **Predictions Export**: Saves predictions along with true labels and features to a CSV file.

## Prerequisites

Make sure to have Python 3.x installed and the following libraries:

- pandas
- scikit-learn
- imbalanced-learn
- time

You can install the necessary libraries using:

```bash
pip install pandas scikit-learn imbalanced-learn
```

## Data Input

The script expects a CSV file with the following structure:
- **Features**:
  - `x force command`: Force command data for the x-axis.
  - `y force command`: Force command data for the y-axis.
  - `z force command`: Force command data for the z-axis.
- **Target**:
  - `swiping_binary`: Binary target indicating swiping (`1`) or no swiping (`0`).

## Usage

1. **Configure Input and Output Paths**: 
   - Set the `input_file` variable to the path of your input CSV file containing the motion data and labels.
   - Update the `output_predictions_file` to specify where the predictions CSV should be saved.

2. **Run the Script**: Execute the script to train the logistic regression model, predict the test labels, and save the results.

### Example Command:

```bash
python logistic_regression_swiping.py
```

## Key Steps

1. **Data Preprocessing**: 
   - Selects force command features (`x force command`, `y force command`, `z force command`) and the binary target (`swiping_binary`).
   - Splits the dataset into training and testing sets.
   - Applies SMOTE to balance the training data.

2. **Model Training**: 
   - Trains a logistic regression model on the SMOTE-balanced dataset.

3. **Prediction and Evaluation**:
   - Makes predictions on the test set.
   - Evaluates the model using accuracy, classification report, and confusion matrix.

4. **Results Saving**: 
   - Saves predictions along with true labels and features to a CSV file.

## Output

The script will output the following:
- **Model Accuracy**: A measure of the model’s performance on the test set.
- **Classification Report**: Precision, recall, f1-score, and support for each class (swiping/no swiping).
- **Confusion Matrix**: A matrix showing true positive, true negative, false positive, and false negative predictions.
- **Model Coefficients**: Displays the logistic regression coefficients for each feature.
- **Predictions CSV**: A CSV file containing the true and predicted labels alongside the input force commands.

## Customization

- Adjust the **test size** for splitting the dataset by changing the `test_size` parameter in `train_test_split`.
- Tune the **logistic regression model** by adding parameters when initializing `LogisticRegression()`.

---

This README should guide users on how to use the script for logistic regression classification on motion data and understand the generated output.
