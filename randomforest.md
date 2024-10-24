Here’s a `README.md` for your Random Forest classifier script:

---

# Random Forest Classifier for Swiping Detection (Force Command Classification)

This Python script uses a Random Forest Classifier to predict whether a motion event involves "swiping over an obstacle" based on force command data. The classifier is trained on x, y, and z force commands with class imbalance handled using SMOTE (Synthetic Minority Over-sampling Technique).

## Features

- **Swiping Classification**: Classifies whether the robot is swiping over an obstacle (`1`) or performing a normal movement (`0`).
- **Class Imbalance Handling**: Applies SMOTE to the training data for balancing the swiping (minority) and non-swiping (majority) classes.
- **Model Performance Metrics**: Outputs accuracy, classification report, and confusion matrix.
- **Feature Importances**: Displays the importance of each force command feature in the model.
- **Predictions Export**: Saves the predicted labels and corresponding force commands to a CSV file.

## Prerequisites

Ensure that Python 3.x is installed along with the required libraries:

- pandas
- scikit-learn
- imbalanced-learn
- os
- time

You can install the necessary dependencies with:

```bash
pip install pandas scikit-learn imbalanced-learn
```

## Data Structure

The script expects CSV files with the following structure in both the training and testing data folders:

- **Features**:
  - `x force command`: Force data for the x-axis.
  - `y force command`: Force data for the y-axis.
  - `z force command`: Force data for the z-axis.
- **Labels**:
  - `swiping_binary`: A binary label where `1` indicates swiping over an obstacle and `0` indicates no swiping.

The data will be automatically loaded from the specified training and testing folders, and additional labels will be generated based on force command thresholds.

## Usage

1. **Set Data Paths**: 
   - Update the `train_folder` and `test_folder` variables to the respective directories containing your training and testing CSV files.

2. **Run the Script**:
   Simply run the script to train the Random Forest classifier, predict labels for the test data, and save the predictions.

### Example Command:

```bash
python random_forest_swiping.py
```

## Key Steps

1. **Data Loading and Preprocessing**:
   - Loads all CSV files from the specified folders.
   - Applies feature engineering to classify the force commands (`x`, `y`, and `z`) into labels such as "Swiping over obstacle" or "Movement in progress."
   - Combines the force command labels into a binary target (`swiping_binary`).

2. **Handling Class Imbalance**:
   - Applies SMOTE to the training data to handle the class imbalance between swiping and no swiping.

3. **Model Training**:
   - Trains a Random Forest Classifier using the SMOTE-balanced dataset with higher weightage assigned to the swiping class.

4. **Model Evaluation**:
   - Makes predictions on the test set and evaluates the model's performance using accuracy, classification report, and confusion matrix.

5. **Feature Importances**:
   - Outputs the importance of each feature in the trained model.

6. **Predictions Saving**:
   - Saves the test set predictions, including true labels and force commands, to a CSV file.

## Output

The script generates the following outputs:
- **Model Accuracy**: The accuracy of the model on the test set.
- **Classification Report**: Precision, recall, f1-score, and support for swiping/no swiping classes.
- **Confusion Matrix**: A matrix showing the true positive, true negative, false positive, and false negative predictions.
- **Feature Importances**: A DataFrame showing the importance of each feature in the model.
- **Predictions CSV**: A CSV file containing the true labels, predicted labels, and force commands for each test example.

## Customization

- You can modify the **class weight** parameter in the Random Forest model to give different importance to swiping (`1`) vs. no swiping (`0`).
- Adjust the **SMOTE parameters** or explore different resampling techniques for handling class imbalance.
- Fine-tune the **Random Forest model hyperparameters** to improve performance.

---

This README guides users through setting up, running, and understanding the outputs of your Random Forest classification script for force command data.
