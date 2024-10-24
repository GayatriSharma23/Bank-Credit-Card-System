import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class TabTransformer(nn.Module):
    def __init__(self, num_features, num_classes, dim_embedding=64, num_heads=4, num_layers=4):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, dim_embedding)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim_embedding, num_classes)

    def forward(self, x):
        # Check the input shape
        print("Input shape to TabTransformer:", x.shape)
        
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Adding a sequence length dimension
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # Pooling
        x = self.classifier(x)
        return x

# Predict function
def predict_in_batches(model, data_tensor, batch_size=1024):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    
    # Create a DataLoader for the input tensor
    data_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=False)

    with torch.no_grad():  # Disable gradient calculation for inference
        for batch_X in data_loader:
            batch_X = batch_X[0].to(device)  # Move to GPU if available
            print("Batch input shape:", batch_X.shape)  # Print the shape of the batch input
            outputs = model(batch_X)  # Get model predictions
            predictions.append(outputs.cpu().numpy())  # Move to CPU and convert to numpy

    return np.concatenate(predictions, axis=0)  # Combine all predictions

# Assume X_train_tensor is already created and has the correct number of features
num_features = X_train_tensor.shape[1]  # Ensure this matches the embedding layer's input

# Prepare the TabTransformer model
tab_transformer_model = TabTransformer(num_features=num_features, num_classes=2).to(device)

# Get transformed features for training
transformed_features_train = predict_in_batches(tab_transformer_model, X_train_tensor, batch_size=1024)
print("Shape of transformed_features_train:", transformed_features_train.shape)  # Check shape

# Prepare the test tensor
X_test_tensor = torch.FloatTensor(tabnet_features_test).to(device)

# Get transformed features for testing
transformed_features_test = predict_in_batches(tab_transformer_model, X_test_tensor, batch_size=1024)
print("Shape of transformed_features_test:", transformed_features_test.shape)  # Check shape

# Ensure that transformed features have the same shape
if transformed_features_train.shape[1] != transformed_features_test.shape[1]:
    raise ValueError("Transformed feature dimensions do not match: "
                     f"Train shape: {transformed_features_train.shape}, "
                     f"Test shape: {transformed_features_test.shape}")

# Convert transformed features to tensors for TabNet
transformed_train_tensor = torch.FloatTensor(transformed_features_train).to(device)
transformed_test_tensor = torch.FloatTensor(transformed_features_test).to(device)

# Print shapes to verify consistency
print("Shape of transformed_train_tensor:", transformed_train_tensor.shape)
print("Shape of transformed_test_tensor:", transformed_test_tensor.shape)

# Train TabNet using the latent features from TabTransformer
clf = TabNetClassifier()
clf.fit(transformed_features_train, smote_y_train.values, max_epochs=100, patience=10)

# Batch predictions with TabNet
tabnet_test_preds = clf.predict(transformed_features_test)
print(tabnet_test_preds)

