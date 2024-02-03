import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import logging
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_cols = [
    "p1_sets",
    "p2_sets",
    "p1_games",
    "p2_games",
    "p1_score",
    "p2_score",
    "server",
    "serve_no",
    # "p1_points_won",
    # "p2_points_won",
    # "game_victor",
    # "set_victor",
    # "p1_ace",
    # "p2_ace",
    # "p1_winner",
    # "p2_winner",
    # "winner_shot_type",
    # "p1_double_fault",
    # "p2_double_fault",
    # "p1_unf_err",
    # "p2_unf_err",
    # "p1_net_pt",
    # "p2_net_pt",
    # "p1_net_pt_won",
    # "p2_net_pt_won",
    "p1_break_pt",
    "p2_break_pt",
    # "p1_break_pt_won",
    # "p2_break_pt_won",
    # "p1_break_pt_missed",
    # "p2_break_pt_missed",
    # "p1_distance_run",
    # "p2_distance_run",
    # "rally_count",
    # "speed_mph",
    # "serve_width",
    # "serve_depth",
    # "return_depth",
]
output_col = "point_victor"

# # Define your LSTM model
# class TennisLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(TennisLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
#         c0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
#         out, _ = self.lstm(x.to(torch.float32), (h0, c0))
#         # print(out.shape)
#         # Add activation function
#         # out = torch.relu(out[:,  :])
#         out = self.fc(out)
#         return out

# Define Logistic Regression model
class TennisLogistic(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TennisLogistic, self).__init__()
        self.fc = nn.Linear(input_size, 32)
        # self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.fc(x)
        out = torch.relu(out)
        # out = self.fc2(out)
        # out = torch.relu(out)
        out = self.fc3(out)
        return out


# Prepare your data (X_train, y_train, X_test, y_test)
# X_train, y_train, X_test, y_test = ...

# Define hyperparameters
input_size = len(input_cols) # Define input size based on your features
hidden_size = 128
num_layers = 2
num_classes = 2  # output param
num_epochs = 50
batch_size = 32
learning_rate = 0.1

# Initialize your model, loss function, and optimizer
# model = TennisLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
model = TennisLogistic(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create DataLoader from csv files
df = pd.read_csv("Data/Wimbledon_featured_matches.csv")
# take column winner_shot_type and convert to int with F as 1 and B as 0
serve_width_encoder = {"B": 0, "BC":1, "BW":2, "C":3, "W":4}
df["winner_shot_type"] = df["winner_shot_type"].apply(lambda x: 1 if x == "F" else 0)
df["serve_width"] = df["serve_width"].apply(lambda x: serve_width_encoder.get(x,0))
df["serve_depth"] = df["serve_depth"].apply(lambda x: 1 if x == "CTL" else 0)
df["return_depth"] = df["return_depth"].apply(lambda x: 1 if x == "D" else 0)
df["p1_score"] = df["p1_score"].apply(lambda x: 50 if x == "AD" else x)
df["p2_score"] = df["p1_score"].apply(lambda x: 50 if x == "AD" else x)

# print(np.unique(df["winner_shot_type"].values))
# print(np.unique(df["serve_width"].values))
# print(np.unique(df["serve_depth"].values))
# print(np.unique(df["return_depth"].values))
# logging.info("Asssertions passed")

# Add new columns p1_current_score and p2_current_score to the dataframe by obtaining p1_points_won and p2_points_won
df["p1_current_score"] = df["p1_points_won"].apply(lambda x: 0 if x == 0 else 15 if x == 1 else 30 if x == 2 else 40 if x == 3 else 50)


# counvert point_victor to 1 if p1 and 0 if p2
df["point_victor"] = df["point_victor"].apply(lambda x: 0 if x == 1 else 1)
X = df[input_cols].values
y = df[output_col].values

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

train_data = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test))
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

val_data = TensorDataset(torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(y_val))
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    best_val_acc = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.to(torch.long)
        
        model.train()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # exit()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Add validation loss
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.to(torch.long)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Val Acc: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model.ckpt")
    

        # if (i + 1) % 100 == 0:
        #     print(
        #         f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
        #     )

# Test the model
model.load_state_dict(torch.load("model.ckpt"))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

# Save the model checkpoint
# torch.save(model.state_dict(), "model.ckpt")

