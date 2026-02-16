# Import libraries
import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 2. Data Loading and Graph Encoding

def load_data():
    dataset = KarateClub()
    data = dataset[0]  # Get the single graph object
    
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Analyze graph structure
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    print(f"Average degree: {data.edge_index.shape[1] / data.num_nodes:.2f}")
    print(f"Training nodes: {data.train_mask.sum().item()}")
    
    return dataset, data

# 3. Non-Geometric Baseline (MLP)

class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        # MLP ignores graph structure (edge_index).
        # It treats every node as an independent sample.
        self.lin1 = torch.nn.Linear(num_features, 16)
        self.lin2 = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index=None):
        # edge_index is unused
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

# 4. GCN Implementation

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        # First Graph Convolutional Layer
        self.conv1 = GCNConv(num_features, 16)
        # Second Graph Convolutional Layer
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        # Layer 1: Aggregates info from 1-hop neighbors
        # H(1) = ReLU( D^-0.5 A D^-0.5 X W(0) )
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        
        # Layer 2: Aggregates info from neighbors of neighbors (2-hops)
        # H(2) = D^-0.5 A D^-0.5 H(1) W(1)
        out = self.conv2(h, edge_index)
        
        return out, h # Return both final output and hidden embedding

# Training and Evaluation Helpers

def train_node_classifier(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    if isinstance(model, GCN):
        out, _ = model(data.x, data.edge_index)
    else: # MLP
        out = model(data.x, data.edge_index)
        
    # Semi-supervised: Calculate loss ONLY on training nodes (4 nodes usually)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        if isinstance(model, GCN):
            out, _ = model(data.x, data.edge_index)
        else:
            out = model(data.x, data.edge_index)
            
        pred = out.argmax(dim=1)
        correct = (pred == data.y).sum()
        acc = int(correct) / int(data.num_nodes)
    return acc

# 6. Visualization

def visualize_embeddings(h, color):
    z = TSNE(n_components=2, perplexity=5, random_state=42, init='pca', learning_rate=200).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.title('2D Visualization of Node Embeddings (Hidden Layer)')
    plt.show()
    
# Main Execution

def main():
    torch.manual_seed(42)
    dataset, data = load_data()
    criterion = torch.nn.CrossEntropyLoss()

    # Train MLP Baseline 
    print("\n[MLP Baseline]")
    mlp = MLP(dataset.num_features, dataset.num_classes)
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        loss = train_node_classifier(mlp, data, optimizer_mlp, criterion)
        if epoch % 50 == 0:
            current_acc = evaluate(mlp, data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {current_acc:.4f}')

    final_mlp_acc = evaluate(mlp, data)
    print(f"Final MLP Accuracy: {final_mlp_acc:.4f}")

    # Train GCN 
    print("\n[GCN Model]")
    gcn = GCN(dataset.num_features, dataset.num_classes)
    optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        loss = train_node_classifier(gcn, data, optimizer_gcn, criterion)
        if epoch % 50 == 0:
            current_acc = evaluate(gcn, data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {current_acc:.4f}')
    
    final_gcn_acc = evaluate(gcn, data)
    print(f"Final GCN Accuracy: {final_gcn_acc:.4f}")

    # Visual Comparison 
    print(f"\nImprovement: {(final_gcn_acc - final_mlp_acc) * 100:.2f}%")
    
    # Visualize GCN embeddings
    gcn.eval()
    _, hidden_emb = gcn(data.x, data.edge_index)
    visualize_embeddings(hidden_emb, data.y)

if __name__ == "__main__":
    main()
