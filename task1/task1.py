import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch.nn import Embedding
from torch_geometric.data import DataLoader, Data
import torch_geometric.nn
import os
import pickle
import matplotlib.pyplot as plt

class GATNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=True, dropout=0.6)
        self.pool = torch_geometric.nn.global_mean_pool

    def forward(self, x, edge_index, batch):
        # print(x.shape, edge_index.shape)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.pool(x, batch)
        return x.squeeze(1)

def move_to_device(data, device):
    # 如果是张量，则直接移动到设备
    if isinstance(data, torch.Tensor):
        return data.to(device)
    # 如果是字典，则递归移动每个值到设备
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    # 如果是列表或元组，则递归移动每个元素到设备
    elif isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    return data

train_dataset = []
train_labels = []
for filename in os.listdir('dataset/train'):
    if filename.endswith('.pkl'):
        with open('dataset/train/' + filename, 'rb') as f:
            data = pickle.load(f)
            train_dataset.append(Data(x=data['node_type'].unsqueeze(1).float(), edge_index=data['edge_index'], y=data['label']))

test_dataset = []
test_labels = []
for filename in os.listdir('dataset/test'):
    if filename.endswith('.pkl'):
        with open('dataset/test/' + filename, 'rb') as f:
            data = pickle.load(f)
            # print(data['label'])
            test_dataset.append(Data(x=data['node_type'].unsqueeze(1).float(), edge_index=data['edge_index'], y=data['label']))
# print(data['node_type'].shape)
# print(data['edge_index'].shape)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Training on GPU.")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Training on CPU.")

# with open('dataset/train/apex3_1060.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(data)
#     print(data['node_type'].shape)
#     print(data['edge_index'].shape)
#     print(data['num_inverted_predecessors'].shape)

model = GATNet(input_dim=1, hidden_dim=32, output_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.MSELoss()

train_losses = []
test_losses = []
print('Start Training')
model.train()
for epoch in range(50):  # 50 个训练周期
    total_loss = 0
    for data in train_loader:
        # 移动数据到 GPU
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_train_loss}")

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y)
            total_loss += loss.item()
        avg_test_loss = total_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f"Average Test Loss: {avg_test_loss}")
    model.train()

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('result.png')