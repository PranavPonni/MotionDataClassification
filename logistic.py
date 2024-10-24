import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import time

# Load the combined CSV file (after feature engineering)
input_file = '/Users/pranavponnivalavan/Documents/TCS Motion Study/TCS motiondata/datacombined/combinedoutputlabel.csv'
data = pd.read_csv(input_file)

# 1. Data Preprocessing
# Select features and the target variable
features = ['x force command', 'y force command', 'z force command']
target = 'swiping_binary'  # Binary classification target (1: swiping, 0: no swiping)

# Split the data into training and testing sets (before applying SMOTE)
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=42, stratify=data[target])

# Apply SMOTE to the training set to handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 2. Model Training
# Initialize the logistic regression model
model = LogisticRegression()

# Train the model on the SMOTE-balanced dataset
start_time = time.time()  # Start the timer
model.fit(X_train_smote, y_train_smote)
end_time = time.time()  # End the timer

print(f"Model training completed in {end_time - start_time:.2f} seconds.")

# 3. Making Predictions
# Predict the swiping/no swiping labels for the test set
y_pred = model.predict(X_test)

# 4. Model Evaluation
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Generate a classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Generate a confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Example: Saving the model's coefficients for interpretation
coefficients = pd.DataFrame(model.coef_, columns=features)
coefficients['intercept'] = model.intercept_
print("Model coefficients:")
print(coefficients)

# Example: Save the predictions to a CSV file
output_predictions_file = #own file path
predictions_df = pd.DataFrame({
    'x force command': X_test['x force command'],
    'y force command': X_test['y force command'],
    'z force command': X_test['z force command'],
    'True Label': y_test,
    'Predicted Label': y_pred
})
predictions_df.to_csv(output_predictions_file, index=False)
print(f"Predictions saved to {output_predictions_file}")
