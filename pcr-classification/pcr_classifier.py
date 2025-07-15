# MAMA-MIA Radiomics + SNN Classification Pipeline

import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from glob import glob
import SimpleITK as sitk
import torch.optim as optim
import matplotlib.pyplot as plt
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from mlxtend.feature_selection import SequentialFeatureSelector

# -----------------------------------
# Radiomics Feature Extraction
# -----------------------------------

def get_paths_pair(root_dir):
    image_paths = sorted(glob(os.path.join(root_dir, "**", "*_0001.nii.gz"), recursive=True))
    mask_paths = sorted(glob(os.path.join(root_dir, "**", "*.nii.gz"), recursive=True))
    return image_paths, mask_paths

def feature_extraction(root_dir, extractor):
    features_df = pd.DataFrame([])
    image_paths, mask_paths = get_paths_pair(root_dir)

    for image, mask in zip(image_paths, mask_paths):
        image_name = os.path.basename(image).replace("_0001.nii.gz", "")
        print(f"Extracting Case: {image_name}")

        mask = sitk.ReadImage(mask)
        image = sitk.ReadImage(image)
        feats = extractor.execute(image, mask)
        feats["ID"] = image_name
        features_df = pd.concat([features_df, pd.DataFrame([feats])], ignore_index=True)

    return features_df

# Run feature extraction
extractor = featureextractor.RadiomicsFeatureExtractor()
data = feature_extraction(root_dir="/content", extractor=extractor)

# Remove diagnostic features
diagnostic_cols = [col for col in data.columns if 'diagnostics' in col]
data = data.drop(columns=diagnostic_cols)
data.to_csv("MAMA-MIA-Radiomics-Features.csv", index=False)

# -----------------------------------
# Correlation Analysis and Feature Selection
# -----------------------------------

radiomics_data = pd.read_csv("MAMA-MIA-Radiomics-Features.csv")

# Correlation heatmap
plt.figure(figsize=(16, 12))
corr = radiomics_data.drop(columns=['ID', 'labels']).corr()
sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Remove highly correlated features (>0.99)
corr_matrix = radiomics_data.drop(columns=['ID', 'labels']).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
print(f"Features suggested to drop due to high correlation: {to_drop}")

# Sequential Feature Selection
x = radiomics_data.drop(columns=['ID', 'labels'] + to_drop)
y = radiomics_data['labels']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)

sfs = SequentialFeatureSelector(RandomForestClassifier(),
           k_features=60,
           forward=True,
           floating=False,
           scoring='accuracy',
           cv=2)

sfs = sfs.fit(x_train, y_train)
selected_features = x_train.columns[list(sfs.k_feature_idx_)]
print("Selected features:", selected_features)

# -----------------------------------
# Self-Normalizing Network (SNN)
# -----------------------------------

class SelfNormalizingNetwork(nn.Module):
    def __init__(self, input_size=40, hidden=64, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden*2)
        self.fc4 = nn.Linear(hidden*2, hidden*2)
        self.fc5 = nn.Linear(hidden*2, output_size)

        self.selu = nn.SELU()
        self.alpha = nn.AlphaDropout(0.1)

    def forward(self, x):
        x = self.selu(self.fc1(x))
        x = self.alpha(x)
        x = self.selu(self.fc2(x))
        x = self.alpha(x)
        x = self.selu(self.fc4(x))
        x = self.alpha(x)
        x = self.fc5(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = SelfNormalizingNetwork(input_size=60, hidden=64, output_size=1).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)


scaler = StandardScaler()
x = scaler.fit_transform(x[selected_features])
y = y.values

dataset = TensorDataset(torch.tensor(x, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.float32).view(-1,1))
dataloader = DataLoader(dataset, shuffle=True, batch_size=32)

num_epochs = 3000

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = loss_fn(outputs, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == yb).sum().item()
        total += yb.numel()

    train_loss /= len(dataloader)
    accuracy = correct / total

    if (epoch+1) % 100 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")
