# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.utils.data import DataLoader, TensorDataset

# TabTransformer model definition
class TabTransformer(nn.Module):
    def __init__(self, input_size, num_classes, dim_embedding=64, num_heads=4, num_layers=4):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, dim_embedding)
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

# Step 1: Train TabNetClassifier
clf = TabNetClassifier()
clf.fit(smote_X_train.values, smote_y_train.values,
        max_epochs=100,
        patience=10,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False)

# Step 2: Generate new features from TabNet model
tabnet_features_train = clf.predict_proba(smote_X_train.values)
tabnet_features_test = clf.predict_proba(smote_X_test.values)

# Step 3: Initialize TabTransformer using TabNet output features as input
input_size = tabnet_features_train.shape[1]  # The input size becomes the TabNet output size
num_classes = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tab_transformer_model = TabTransformer(input_size, num_classes).to(device)

# Define criterion and optimizer for TabTransformer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(tab_transformer_model.parameters(), lr=0.001)

# Convert TabNet-generated features to tensors for TabTransformer
X_train_tensor = torch.FloatTensor(tabnet_features_train).to(device)
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

# Step 4: Final prediction using TabTransformer
X_test_tensor = torch.FloatTensor(tabnet_features_test).to(device)
tab_transformer_model.eval()
with torch.no_grad():
    final_preds = tab_transformer_model(X_test_tensor).cpu().numpy()

# Convert predictions to class labels
final_class_preds = np.argmax(final_preds, axis=1)
print(final_class_preds)
