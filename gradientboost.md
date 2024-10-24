Here’s a `README.md` for your Gradient Boosting classifier script:

---

# Gradient Boosting Classifier for Swiping Detection (Force Command Classification)

This Python script utilizes a Gradient Boosting Classifier to detect whether a robot's motion involves "swiping over an obstacle" based on x, y, and z force command data. The script handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique), trains a Gradient Boosting model, and evaluates its performance.

## Features

- **Swiping Classification**: Predicts whether a motion event is classified as "swiping over an obstacle" (`1`) or "no swiping" (`0`).
- **Class Imbalance Handling**: Uses SMOTE to balance the training dataset for swiping and no swiping classes.
- **Model Performance Metrics**: Outputs accuracy, classification report, and confusion matrix.
- **Feature Importances**: Displays the relative importance of each feature (x, y, z force commands) in the model.
- **Predictions Export**: Saves test set predictions along with true labels to a CSV file.

## Prerequisites

Before running the script, ensure that Python 3.x and the following libraries are installed:

- pandas
- scikit-learn
- imbalanced-learn
- os
- time

Install the necessary dependencies via pip:

```bash
pip install pandas scikit-learn imbalanced-learn
```

## Data Structure

The script expects CSV files in the specified training and testing folders. Each CSV file should include the following columns:

- `x force command`: Force data along the x-axis.
- `y force command`: Force data along the y-axis.
- `z force command`: Force data along the z-axis.

### Labels:
- The script applies feature engineering to create additional labels for classification:
  - **`swiping_binary`**: A binary target label where `1` means swiping over an obstacle, and `0` means no swiping.

The data files will be loaded and processed automatically from the specified `train_folder` and `test_folder`.

## How to Run

1. **Set Data Paths**: 
   Update the following variables in the script with your directory paths:
   - `train_folder`: Path to the folder containing the training data CSV files.
   - `test_folder`: Path to the folder containing the testing data CSV files.

2. **Run the Script**:
   Execute the script to train the Gradient Boosting Classifier and evaluate the model.

### Example Command:

```bash
python gradient_boosting_swiping.py
```

## Main Steps

1. **Data Loading and Preprocessing**:
   - Loads and concatenates CSV files from the train and test folders.
   - Applies feature engineering to classify force commands into movement labels and generate the target column (`swiping_binary`).

2. **SMOTE for Class Imbalance**:
   - Balances the training data to account for class imbalance using SMOTE.

3. **Training the Gradient Boosting Model**:
   - The Gradient Boosting Classifier is trained on the SMOTE-balanced training data.

4. **Model Evaluation**:
   - The model makes predictions on the test data and outputs the following metrics:
     - **Accuracy**: Proportion of correct predictions.
     - **Classification Report**: Precision, recall, f1-score, and support for both classes.
     - **Confusion Matrix**: Breakdown of true positives, true negatives, false positives, and false negatives.

5. **Feature Importances**:
   - Outputs the importance of each feature (x, y, and z force commands) in the trained model.

6. **Saving Predictions**:
   - Saves the predicted labels for the test set, along with the true labels and corresponding force commands, to a CSV file.

## Output

The script generates the following outputs:

- **Accuracy**: Overall accuracy of the Gradient Boosting model on the test set.
- **Classification Report**: Detailed performance metrics for each class (swiping/no swiping).
- **Confusion Matrix**: Summary of prediction results.
- **Feature Importances**: A list of the most important features used by the model.
- **Predictions CSV**: File containing true labels, predicted labels, and the associated x, y, z force commands.

### Output Files:

- `gb_predictions.csv`: Contains the test set predictions, saved to the specified output file path.

## Customization

- **Class Imbalance**: You can modify the SMOTE parameters to experiment with different oversampling strategies.
- **Model Parameters**: Fine-tune the hyperparameters of the Gradient Boosting model to improve performance (e.g., `n_estimators`, `learning_rate`, `max_depth`).

---

This README provides guidance on setting up, running, and interpreting the outputs of your Gradient Boosting classification model for swiping detection using force command data.
