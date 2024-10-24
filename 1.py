# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# TabTransformer model definition
class TabTransformer(nn.Module):
    def __init__(self, num_features, num_classes, dim_embedding=64, num_heads=4, num_layers=4):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, dim_embedding)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim_embedding, num_classes)
 
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Adding a sequence length dimension
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # Pooling
        x = self.classifier(x)
        return x

# Initialize TabNetClassifier
clf = TabNetClassifier()
clf.fit(smote_X_train.values, smote_y_train.values,
        max_epochs=100,
        patience=10,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False)

# Initialize TabTransformer
num_features = smote_X_train.shape[1]
num_classes = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tab_transformer_model = TabTransformer(num_features, num_classes).to(device)

# Define criterion and optimizer for TabTransformer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(tab_transformer_model.parameters(), lr=0.001)

# Convert training data to tensors for TabTransformer
X_train_tensor = torch.FloatTensor(smote_X_train.values).to(device)
y_train_tensor = torch.LongTensor(smote_y_train.values).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop for TabTransformer
num_epochs = 100
for epoch in range(num_epochs):
    tab_transformer_model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = tab_transformer_model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, TabTransformer Loss: {loss.item()}')

# Final prediction: combine outputs of both models
def final_prediction(tabnet_model, transformer_model, X_test):
    # TabNet predictions
    tabnet_preds = tabnet_model.predict_proba(X_test.values)
    
    # TabTransformer predictions
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    transformer_model.eval()  # Set to evaluation mode
    with torch.no_grad():
        transformer_preds = transformer_model(X_test_tensor).cpu().numpy()
    
    # Combine predictions (simple average for example, you can modify to use voting or weighted average)
    combined_preds = (tabnet_preds + transformer_preds) / 2.0
    final_preds = np.argmax(combined_preds, axis=1)
    
    return final_preds

# Example of using final_prediction function
final_preds = final_prediction(clf, tab_transformer_model, smote_X_test)
print(final_preds)
