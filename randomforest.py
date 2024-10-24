import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import os
import time

# File paths
train_folder = #own file path
test_folder = #own file path

# Function to load and preprocess CSV files
def load_and_preprocess_files(folder):
    data_list = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder, file_name)
            data = pd.read_csv(file_path)
            # Apply feature engineering
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

            def classify_swiping_in_motion(row):
                if (row['x force label'] == "Swiping over obstacle" and
                    row['y force label'] == "Swiping over obstacle" and
                    row['z force label'] == "Swiping over obstacle"):
                    return "swiping"
                return "no swiping"

            data['x force label'] = data['x force command'].apply(classify_x_force)
            data['y force label'] = data['y force command'].apply(classify_y_force)
            data['z force label'] = data['z force command'].apply(classify_z_force)
            data['swiping in motion'] = data.apply(classify_swiping_in_motion, axis=1)
            data['swiping_binary'] = data['swiping in motion'].apply(lambda x: 1 if x == 'swiping' else 0)
            
            data_list.append(data)
    
    combined_data = pd.concat(data_list, ignore_index=True)
    return combined_data

# Load and preprocess training and testing data
train_data = load_and_preprocess_files(train_folder)
test_data = load_and_preprocess_files(test_folder)

# Select features and target variable
features = ['x force command', 'y force command', 'z force command']
target = 'swiping_binary'

# Split features and target
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize the random forest model
rf_model = RandomForestClassifier(class_weight={0: 1, 1: 10}, random_state=42)

# Train the model
start_time = time.time()
rf_model.fit(X_train_smote, y_train_smote)
end_time = time.time()

print(f"Random Forest model training completed in {end_time - start_time:.2f} seconds.")

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model feature importances
importances = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_})
print("Feature Importances:")
print(importances)

# Save predictions to CSV
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
