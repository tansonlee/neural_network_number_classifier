import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np
from model import NeuralNetwork, cross_entropy_loss
import matplotlib.pyplot as plt

mnist_train = datasets.MNIST(
    root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(
    root='./data', train=False, download=True, transform=transforms.ToTensor())

# Extract images and labels from the dataset
train_images = torch.stack([img for img, _ in mnist_train])   # shape: (60000, 1, 28, 28)
train_labels = torch.tensor([label for _, label in mnist_train])  # shape: (60000,)

test_images = torch.stack([img for img, _ in mnist_test])   # shape: (60000, 1, 28, 28)
test_labels = torch.tensor([label for _, label in mnist_test])  # shape: (60000,)

def transform_data(images, labels):
    # Flatten each image to a 784-length vector
    X = images.view(-1, 28*28).numpy().T  # shape: (60000, 784)
    y = labels.numpy()                 # shape: (60000,)
    y = np.eye(10)[y].T
    return X, y

def compute_accuracy(y, y_hat):
    y = np.argmax(y, axis=0)  # shape: (batch_size,)
    predicted = np.argmax(y_hat, axis=0)  # shape: (batch_size,)
    correct = np.sum(predicted == y)
    accuracy = correct / y.shape[0]
    return accuracy

X, y = transform_data(train_images, train_labels)

nn = NeuralNetwork()
step = 0.01
epochs = 30
batch_size = 64
losses = []

for epoch in range(epochs):
    perm = np.random.permutation(X.shape[1])
    X_shuffled = X[:, perm]
    y_shuffled = y[:, perm]

    for i in range(0, X.shape[1], batch_size):
        X_batch = X_shuffled[:, i:i+batch_size]
        y_batch = y_shuffled[:, i:i+batch_size]
        
        output = nn.forward(X_batch)
        nn.backward(y_batch, step)

        if i % 1000 == 0:
            loss = cross_entropy_loss(y_batch, output)
            print(f"Loss: {loss:.4f}")
            losses.append(loss)

X, y = transform_data(test_images, test_labels)

result = nn.forward(X)
output_labels = np.argmax(result, axis=0)
correct = np.sum(test_labels.numpy() == output_labels)
accuracy = correct / test_labels.shape[0] 
print(accuracy)

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.show()

# Pick N random test samples
N = 20
indices = np.random.choice(test_images.shape[0], N, replace=False)
sample_images = test_images[indices]             # shape: (N, 1, 28, 28)
sample_labels = test_labels[indices].numpy()     # shape: (N,)

# Preprocess images for model (reshape to 784xN)
X_sample = sample_images.view(N, -1).numpy().T   # shape: (784, N)

# Forward pass
predictions = np.argmax(nn.forward(X_sample), axis=0)  # shape: (N,)

# Plot
plt.figure(figsize=(20, 4))
for i in range(N):
    plt.subplot(1, N, i + 1)
    plt.imshow(sample_images[i][0], cmap='gray')
    plt.axis('off')
    plt.title(f"True: {sample_labels[i]}\nPred: {predictions[i]}")
plt.suptitle("MNIST Predictions")
plt.show()