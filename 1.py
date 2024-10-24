import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def predict_in_batches(model, data_tensor, batch_size=1024):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    
    # Create a DataLoader for the input tensor
    data_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=False)

    with torch.no_grad():  # Disable gradient calculation for inference
        for batch_X in data_loader:
            batch_X = batch_X[0].to(device)  # Move to GPU if available
            outputs = model(batch_X)  # Get model predictions
            predictions.append(outputs.cpu().numpy())  # Move to CPU and convert to numpy

    return np.concatenate(predictions, axis=0)  # Combine all predictions

# 1. Prepare the input tensor
X_train_tensor = torch.FloatTensor(smote_X_train.values).to(device)

# 2. Get transformed features in batches for training
transformed_features_train = predict_in_batches(tab_transformer_model, X_train_tensor, batch_size=1024)
print("Shape of transformed_features_train:", transformed_features_train.shape)  # Check shape

# 3. Prepare the test tensor (assumed to be already in the right format)
X_test_tensor = torch.FloatTensor(tabnet_features_test).to(device)

# 4. Get transformed features in batches for testing
transformed_features_test = predict_in_batches(tab_transformer_model, X_test_tensor, batch_size=1024)
print("Shape of transformed_features_test:", transformed_features_test.shape)  # Check shape

# 5. Ensure that the output dimensions match
if transformed_features_train.shape[1] != transformed_features_test.shape[1]:
    raise ValueError("Transformed feature dimensions do not match: "
                     f"Train shape: {transformed_features_train.shape}, "
                     f"Test shape: {transformed_features_test.shape}")

# 6. Convert transformed features to tensors for TabNet
transformed_train_tensor = torch.FloatTensor(transformed_features_train).to(device)
transformed_test_tensor = torch.FloatTensor(transformed_features_test).to(device)

# 7. Print shapes to verify consistency
print("Shape of transformed_train_tensor:", transformed_train_tensor.shape)
print("Shape of transformed_test_tensor:", transformed_test_tensor.shape)

# 8. Train TabNet using the latent features from TabTransformer
clf = TabNetClassifier()
clf.fit(transformed_features_train, smote_y_train.values, max_epochs=100, patience=10)

# 9. Batch predictions with TabNet
tabnet_test_preds = clf.predict(transformed_features_test)
print(tabnet_test_preds)


