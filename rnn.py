import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
X_train = train_data[features].values
y_train = train_data[target].values
X_test = test_data[features].values
y_test = test_data[target].values

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_smote, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_smote, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 128
output_size = 2
num_layers = 2
learning_rate = 0.001
batch_size = 64
num_epochs = 1000

# Initialize model, loss function, and optimizer
model = RNNModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor.unsqueeze(1))
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

end_time = time.time()
print(f"Model training completed in {end_time - start_time:.2f} seconds.")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor.unsqueeze(1)).argmax(dim=1)

# Convert predictions and targets to numpy arrays for evaluation
y_pred = y_pred.numpy()
y_test = y_test_tensor.numpy()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
torch.save(model.state_dict(), '#file path')
print("Model saved to rnn_model.pth")

# Save predictions to CSV
output_predictions_file = #own file path
predictions_df = pd.DataFrame({
    'x force command': test_data['x force command'],
    'y force command': test_data['y force command'],
    'z force command': test_data['z force command'],
    'True Label': y_test,
    'Predicted Label': y_pred
})
predictions_df.to_csv(output_predictions_file, index=False)
print(f"Predictions saved to {output_predictions_file}")
