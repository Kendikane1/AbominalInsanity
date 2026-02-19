# --- START OF FILE AE_CNN_Reconstruction_digits_SequentialSplit.py ---

# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# --- Scikit-learn imports ---
from sklearn.datasets import load_digits
# Removed train_test_split as we are doing sequential split
import numpy as np
# --- Visualization Import ---
import matplotlib.pyplot as plt

# --- Define the Autoencoder ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: 1x8x8 -> 8x4x4 -> 16x2x2
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1), # (N, 1, 8, 8) -> (N, 8, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (N, 8, 8, 8) -> (N, 8, 4, 4)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # (N, 8, 4, 4) -> (N, 16, 4, 4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (N, 16, 4, 4) -> (N, 16, 2, 2)
        )
        # Decoder: 16x2x2 -> 8x4x4 -> 1x8x8
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2), # (N, 16, 2, 2) -> (N, 8, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2), # (N, 8, 4, 4) -> (N, 1, 8, 8)
            nn.Sigmoid() # Output pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- Define the CNN (Modified for 8x8 input from previous step) ---
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=(3,3),
            stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(3,3),
            stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16 * 2 * 2, num_classes) # 16 * 2 * 2 = 64

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# --- Set device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Hyperparameters ---
# General
num_classes = 10
batch_size = 32
num_train_samples = 1000 # Define the split point

# Autoencoder specific
ae_learning_rate = 0.002
ae_num_epochs = 1000 # AE might need more epochs for good reconstruction

# CNN specific
cnn_in_channels = 1
cnn_learning_rate = 0.023
cnn_num_epochs = 500


# --- Load and Preprocess Original Data ---
digits = load_digits()
X = digits.data # (1797, 64) numpy array
y = digits.target # (1797,) numpy array
print(f"Total dataset size: {X.shape[0]} samples")

# --- Sequential Split ---
# Use the first num_train_samples for training, the rest for testing
X_train_raw = X[:num_train_samples]
y_train = y[:num_train_samples]
X_test_raw = X[num_train_samples:]
y_test = y[num_train_samples:]

print(f"Training samples: {X_train_raw.shape[0]}")
print(f"Testing samples: {X_test_raw.shape[0]}")

# Reshape and Normalize (Original data)
# Apply to the split data
X_train = X_train_raw.reshape(-1, 1, 8, 8) / 16.0
X_test = X_test_raw.reshape(-1, 1, 8, 8) / 16.0

# Convert original data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device) # Move to device early
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)


# --- Create DataLoaders FOR AUTOENCODER TRAINING ---
# Input: original, Target: original (for reconstruction learning)
ae_train_dataset = TensorDataset(X_train_tensor, X_train_tensor) # Target is itself
ae_train_loader = DataLoader(dataset=ae_train_dataset, batch_size=batch_size, shuffle=True)
# Optional: AE test loader for validation loss monitoring (using the actual test split)
ae_test_dataset = TensorDataset(X_test_tensor, X_test_tensor)
ae_test_loader = DataLoader(dataset=ae_test_dataset, batch_size=batch_size, shuffle=False)


# --- Initialise Models ---
autoencoder = Autoencoder().to(device)
model_cnn = CNN(in_channels=cnn_in_channels, num_classes=num_classes).to(device)

# --- Loss and Optimisers ---
criterion_ae = nn.MSELoss() # Mean Squared Error for reconstruction
optimiser_ae = optim.Adam(autoencoder.parameters(), lr=ae_learning_rate)

criterion_cnn = nn.CrossEntropyLoss() # For classification
optimiser_cnn = optim.Adam(model_cnn.parameters(), lr=cnn_learning_rate)


# =========================================
# --- Train the Autoencoder (Reconstruction) ---
# =========================================
print("\n--- Training Autoencoder (for Reconstruction) ---")
autoencoder.train() # Ensure model is in training mode
for epoch in range(ae_num_epochs):
    epoch_loss_ae = 0.0
    for batch_idx, (data_in, data_target) in enumerate(ae_train_loader):
        # Data is already on device

        # Forward
        outputs = autoencoder(data_in)
        loss = criterion_ae(outputs, data_target) # Compare output to original
        epoch_loss_ae += loss.item()

        # Backward & Optimize
        optimiser_ae.zero_grad()
        loss.backward()
        optimiser_ae.step()

    # Calculate validation loss on the test set
    val_loss_ae = 0.0
    autoencoder.eval()
    with torch.no_grad():
        for data_in_val, data_target_val in ae_test_loader:
            outputs_val = autoencoder(data_in_val)
            loss_val = criterion_ae(outputs_val, data_target_val)
            val_loss_ae += loss_val.item()
    autoencoder.train() # Set back to train mode

    # Print less frequently for long trainings
    if (epoch + 1) % 50 == 0 or epoch == ae_num_epochs - 1:
      print(f"AE Epoch [{epoch+1}/{ae_num_epochs}], Train Loss: {epoch_loss_ae/len(ae_train_loader):.6f}, Val Loss: {val_loss_ae/len(ae_test_loader):.6f}")

# ============================================================
# --- Generate Reconstructed Data using Trained Autoencoder ---
# ============================================================
print("\n--- Generating Reconstructed Data using Trained AE ---")
autoencoder.eval() # Set AE to evaluation mode
with torch.no_grad(): # No gradients needed for this step
    X_train_reconstructed = autoencoder(X_train_tensor).detach() # Detach from graph
    X_test_reconstructed = autoencoder(X_test_tensor).detach()

print(f"Shape of reconstructed training data: {X_train_reconstructed.shape}")
print(f"Shape of reconstructed testing data: {X_test_reconstructed.shape}")

# ============================================================
# --- Create DataLoaders FOR CNN TRAINING (using RECONSTRUCTED data) ---
# ============================================================
cnn_train_dataset = TensorDataset(X_train_reconstructed, y_train_tensor)
cnn_train_loader = DataLoader(dataset=cnn_train_dataset, batch_size=batch_size, shuffle=True)

cnn_test_dataset = TensorDataset(X_test_reconstructed, y_test_tensor)
cnn_test_loader = DataLoader(dataset=cnn_test_dataset, batch_size=batch_size, shuffle=False)


# ==================================
# --- Train the CNN (on Reconstructed Data) ---
# ==================================
print("\n--- Training CNN (on AE Reconstructed data) ---")
model_cnn.train() # Ensure model is in training mode
for epoch in range(cnn_num_epochs):
    epoch_loss_cnn = 0.0
    for batch_idx, (data, targets) in enumerate(cnn_train_loader):
        # Data (reconstructed) and targets should already be on the correct device

        # forward
        scores = model_cnn(data)
        loss = criterion_cnn(scores, targets)
        epoch_loss_cnn += loss.item()

        # backward
        optimiser_cnn.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimiser_cnn.step()

    # Print less frequently for long trainings
    if (epoch + 1) % 50 == 0 or epoch == cnn_num_epochs - 1:
      print(f"CNN Epoch [{epoch+1}/{cnn_num_epochs}], Loss: {epoch_loss_cnn/len(cnn_train_loader):.4f}")


# ==================================
# --- Evaluation Function (Tests CNN on Reconstructed Data) ---
# ==================================
def check_accuracy_cnn_on_reconstructed(loader, model, dataset_name="Reconstructed Test"):
    print(f"\nChecking CNN accuracy on {dataset_name} data")
    num_correct = 0
    num_samples = 0
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        for x, y in loader: # x is RECONSTRUCTED data here
            # Data already on device

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

    accuracy = (float(num_correct) / float(num_samples)) * 100 if num_samples > 0 else 0
    print(f"CNN on Reconstructed Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}%")
    model.train() # Set model back to training mode
    return accuracy


# --- Optional: Visualize Reconstruction Results ---
def visualize_reconstruction(ae_model, original_images_tensor, num_examples=5):
    print("\n--- Visualizing AE Reconstruction ---")
    ae_model.eval()
    if len(original_images_tensor) < num_examples:
        print(f"Warning: Requesting {num_examples} examples, but only {len(original_images_tensor)} available.")
        num_examples = len(original_images_tensor)
    indices = torch.randperm(len(original_images_tensor))[:num_examples]
    # Ensure original images are on the correct device for the model
    original_in = original_images_tensor[indices].to(device)

    with torch.no_grad():
        reconstructed_out = ae_model(original_in)

    # Move tensors to CPU for numpy conversion and plotting
    original_in_cpu = original_in.cpu().numpy()
    reconstructed_out_cpu = reconstructed_out.cpu().numpy()

    plt.figure(figsize=(10, 4))
    for i in range(num_examples):
        # Original Input
        ax = plt.subplot(2, num_examples, i + 1)
        plt.imshow(original_in_cpu[i, 0], cmap='gray')
        ax.set_title("Original In")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # AE Reconstruction
        ax = plt.subplot(2, num_examples, i + 1 + num_examples)
        plt.imshow(reconstructed_out_cpu[i, 0], cmap='gray')
        ax.set_title("Reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()
    ae_model.train() # Set back to train mode


# ==================================
# --- Run Evaluation ---
# ==================================
print("\n--- Evaluating Models ---")

# 1. Evaluate CNN on reconstructed test data
check_accuracy_cnn_on_reconstructed(cnn_test_loader, model_cnn, dataset_name="Reconstructed Test")

# 2. Visualize some reconstruction examples using the test set originals
visualize_reconstruction(autoencoder, X_test_tensor, num_examples=8)


print("\nDone.")

# --- END OF FILE ---