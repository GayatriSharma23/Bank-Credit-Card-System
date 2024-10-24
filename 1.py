# Function to predict in batches using DataLoader
def predict_in_batches(model, data_tensor, batch_size=1024):
    data_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=False)
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            batch_X = batch_data[0].to(device)  # Extract input features from batch
            batch_preds = model(batch_X)        # Perform batch predictions
            all_preds.append(batch_preds.cpu().numpy())  # Collect predictions

    # Concatenate all batch predictions into a single array
    return np.concatenate(all_preds, axis=0)

# TabTransformer Model - Latent feature extraction in batches
tab_transformer_model.eval()
X_train_tensor = torch.FloatTensor(smote_X_train.values).to(device)  # Convert to tensor
X_test_tensor = torch.FloatTensor(smote_X_test.values).to(device)

# Batch prediction for the training set
transformed_features_train = predict_in_batches(tab_transformer_model, X_train_tensor, batch_size=1024)
# Batch prediction for the test set
transformed_features_test = predict_in_batches(tab_transformer_model, X_test_tensor, batch_size=1024)

# Now that we have latent features, pass them to TabNet
transformed_train_tensor = torch.FloatTensor(transformed_features_train).to(device)
transformed_test_tensor = torch.FloatTensor(transformed_features_test).to(device)

# Train TabNet using the latent features from TabTransformer
clf = TabNetClassifier()

# Fit TabNet on the transformed features from TabTransformer
clf.fit(transformed_features_train, smote_y_train.values, max_epochs=100, patience=10)

# Batch predictions with TabNet
tabnet_test_preds = clf.predict(transformed_features_test)
print(tabnet_test_preds)
