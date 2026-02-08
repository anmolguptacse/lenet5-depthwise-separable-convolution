import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

#  PART A: Regular LeNet-5 
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Original LeNet-5 architecture 
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # Output: 28x28x6
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)      # Output: 14x14x6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)           # Output: 10x10x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)      # Output: 5x5x16
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)         
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, 120)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# PART B: LeNet-5 with Depthwise Separable Convolutions 
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LeNet5_Depthwise(nn.Module):
    def __init__(self):
        super(LeNet5_Depthwise, self).__init__()
        # Replace all conv layers with depthwise separable conv
        self.conv1 = DepthwiseSeparableConv(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DepthwiseSeparableConv(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = DepthwiseSeparableConv(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, 120)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, test_loader, model_name="LeNet5", epochs=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Testing phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * correct / total
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return train_losses, train_accuracies, test_accuracies

# TRAIN BOTH MODELS 

print("Training Regular LeNet-5 (with MaxPool)")

model_regular = LeNet5().to(device)
regular_losses, regular_train_acc, regular_test_acc = train_model(
    model_regular, train_loader, test_loader, "Regular LeNet-5", epochs=15
)


print("Training Depthwise Separable LeNet-5 (with MaxPool)")

model_depthwise = LeNet5_Depthwise().to(device)
depthwise_losses, depthwise_train_acc, depthwise_test_acc = train_model(
    model_depthwise, train_loader, test_loader, "Depthwise LeNet-5", epochs=15
)

# PLOT RESULTS 
plt.figure(figsize=(14, 5))

# Plot Accuracy vs Epochs
plt.subplot(1, 2, 1)
plt.plot(range(1, 16), regular_test_acc, 'b-', label='Regular LeNet-5 (Test)', linewidth=2)
plt.plot(range(1, 16), depthwise_test_acc, 'r-', label='Depthwise (Test)', linewidth=2)
plt.plot(range(1, 16), regular_train_acc, 'b--', label='Regular LeNet-5 (Train)', alpha=0.7)
plt.plot(range(1, 16), depthwise_train_acc, 'r--', label='Depthwise (Train)', alpha=0.7)
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Epochs (MaxPool)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Loss vs Epochs
plt.subplot(1, 2, 2)
plt.plot(range(1, 16), regular_losses, 'b-', label='Regular LeNet-5', linewidth=2)
plt.plot(range(1, 16), depthwise_losses, 'r-', label='Depthwise', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Loss vs Epochs (MaxPool)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lenet5_comparison_maxpool.png', dpi=150)
plt.show()

#  PART C: CALCULATE OPERATIONS

print("PART C: Operation Count Comparison (Multiply-Add Operations)")


def calculate_operations_regular(C_in, C_out, H_in, W_in, K, padding=0, stride=1):
    """
    Calculate operations for regular convolution
    For regular conv: Each output pixel requires C_in * K² multiply-accumulate operations
    Multiply-accumulate (MAC) = 1 multiply + 1 add
    Total MACs = C_out * H_out * W_out * C_in * K²
    """
    H_out = (H_in + 2*padding - K) // stride + 1
    W_out = (W_in + 2*padding - K) // stride + 1
    
    # Total multiply-accumulate operations
    macs = C_out * H_out * W_out * C_in * K * K
    

    multiplies = C_out * H_out * W_out * C_in * K * K
    additions = C_out * H_out * W_out * (C_in * K * K - 1)
    
    return macs, multiplies, additions, H_out, W_out

def calculate_operations_depthwise(C_in, C_out, H_in, W_in, K, padding=0, stride=1):
    """
    Calculate operations for depthwise separable convolution
    """
    H_out = (H_in + 2*padding - K) // stride + 1
    W_out = (W_in + 2*padding - K) // stride + 1
    
    # Depthwise convolution
    depthwise_macs = C_in * H_out * W_out * K * K
    depthwise_mult = C_in * H_out * W_out * K * K
    depthwise_add = C_in * H_out * W_out * (K * K - 1)
    
    # Pointwise convolution (1x1 convolution)
    pointwise_macs = C_out * H_out * W_out * C_in
    pointwise_mult = C_out * H_out * W_out * C_in
    pointwise_add = C_out * H_out * W_out * (C_in - 1)
    
    total_macs = depthwise_macs + pointwise_macs
    total_mult = depthwise_mult + pointwise_mult
    total_add = depthwise_add + pointwise_add
    
    return total_macs, total_mult, total_add

# LeNet-5 Layer specifications (using MNIST 28x28 input)
layers = [
    # (C_in, C_out, H_in, W_in, K, padding)
    (1, 6, 28, 28, 5, 2),    # Conv1: 1->6, 5x5, padding=2
    (6, 16, 14, 14, 5, 0),   # Conv2: 6->16, 5x5, after pool
    (16, 120, 5, 5, 5, 0),   # Conv3: 16->120, 5x5, after pool
]

print("\nOperation Count for Each Layer (Multiply-Add Operations):")

print(f"{'Layer':<10} {'Type':<20} {'MACs':<15} {'Multiplies':<15} {'Additions':<15}")


total_macs_regular = 0
total_mult_regular = 0
total_add_regular = 0
total_macs_depthwise = 0
total_mult_depthwise = 0
total_add_depthwise = 0

for i, (C_in, C_out, H_in, W_in, K, padding) in enumerate(layers, 1):
    # Regular convolution
    macs_reg, mult_reg, add_reg, H_out, W_out = calculate_operations_regular(C_in, C_out, H_in, W_in, K, padding)
    total_macs_regular += macs_reg
    total_mult_regular += mult_reg
    total_add_regular += add_reg
    
    # Depthwise separable convolution
    macs_dw, mult_dw, add_dw = calculate_operations_depthwise(C_in, C_out, H_in, W_in, K, padding)
    total_macs_depthwise += macs_dw
    total_mult_depthwise += mult_dw
    total_add_depthwise += add_dw
    
    print(f"Conv{i}:")
    print(f"{'':<10} {'Regular':<20} {macs_reg:<15,} {mult_reg:<15,} {add_reg:<15,}")
    print(f"{'':<10} {'Depthwise Sep':<20} {macs_dw:<15,} {mult_dw:<15,} {add_dw:<15,}")
    print()

# Print totals

print("TOTAL OPERATIONS FOR ALL CONVOLUTIONAL LAYERS:")

print(f"{'Network Type':<25} {'MACs':<20} {'Multiplies':<20} {'Additions':<20} {'Total Ops':<20}")

print(f"{'Regular LeNet-5':<25} {total_macs_regular:<20,} {total_mult_regular:<20,} {total_add_regular:<20,} {total_mult_regular+total_add_regular:<20,}")
print(f"{'Depthwise LeNet-5':<25} {total_macs_depthwise:<20,} {total_mult_depthwise:<20,} {total_add_depthwise:<20,} {total_mult_depthwise+total_add_depthwise:<20,}")
