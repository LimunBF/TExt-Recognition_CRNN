import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from crnn_model import CRNN
from utils.dataset import OCRDataset
from utils.label_encoder import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Collate untuk handle sequence label dengan panjang berbeda
def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images)
    targets = pad_sequence(labels, batch_first=True, padding_value=0)
    return images, targets, torch.tensor(label_lengths)

# Load konfigurasi
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup
device = torch.device(config["device"])
img_h = config["dataset"]["img_height"]
img_w = config["dataset"]["img_width"]
charset = config["dataset"]["charset"]
label_encoder = LabelEncoder(charset)
num_classes = len(charset) + 1  # +1 untuk CTC blank

# Dataset & DataLoader
train_set = OCRDataset(config["dataset"]["train_labels"], img_h, img_w, label_encoder)
val_set = OCRDataset(config["dataset"]["val_labels"], img_h, img_w, label_encoder)

train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"],
    shuffle=True, num_workers=config["training"]["num_workers"], # num_workers can be > 0 if your system supports it
    collate_fn=collate_fn)

val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn) # num_workers 0 is fine for validation

# Model
model = CRNN(img_h, 1, num_classes).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

# --- NEW: Learning Rate Scheduler ---
# Reduces learning rate when validation loss stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# --- NEW: Early Stopping Variables ---
best_val_loss = float('inf')
epochs_no_improve = 0
patience = 10 # Number of epochs to wait for validation loss improvement before early stopping
min_delta = 0.001 # Minimum change to be considered an improvement

# --- NEW: Validation Function ---
def validate(model, val_loader, criterion, device):
    model.eval() # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad(): # Disable gradient calculations
        for images, targets, target_lengths in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(images)
            outputs = outputs.permute(1, 0, 2) # CTC: [T, B, C]
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

            loss = criterion(outputs, targets, input_lengths, target_lengths)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train():
    global best_val_loss, epochs_no_improve # Declare global to modify outside function scope

    for epoch in range(config["training"]["epochs"]):
        # --- Training Loop ---
        model.train() # Set model to training mode
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} (Train)")
        for images, targets, target_lengths in pbar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # [B, T, C]
            outputs = outputs.permute(1, 0, 2)  # CTC: [T, B, C]
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()

            # --- NEW: Gradient Clipping ---
            # Prevents exploding gradients, common in RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        current_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {current_train_loss:.4f}")

        # --- Validation Loop ---
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

        # --- NEW: Step the Learning Rate Scheduler ---
        scheduler.step(val_loss)

        # --- NEW: Early Stopping Logic ---
        if val_loss + min_delta < best_val_loss: # Check if validation loss has improved significantly
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model, not just the last one
            torch.save(model.state_dict(), config["training"]["model_save_path"])
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without significant improvement.")
                break # Exit the training loop

if __name__ == "__main__":
    train()
