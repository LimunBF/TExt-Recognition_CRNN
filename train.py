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
from Levenshtein import distance # Import Levenshtein for CER calculation

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
    shuffle=True, num_workers=config["training"]["num_workers"],
    collate_fn=collate_fn)

val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

# Model
model = CRNN(img_h, 1, num_classes).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Early Stopping Variables - now using values from config
best_val_loss = float('inf')
epochs_no_improve = 0
patience = config["training"]["early_stopping_patience"] # Mengambil dari config
min_delta = config["training"]["early_stopping_min_delta"] # Mengambil dari config

# Validation Function - now also calculates CER
def validate(model, val_loader, criterion, device, label_encoder): # Added label_encoder as argument
    model.eval()
    total_loss = 0
    total_cer_distance = 0
    total_chars = 0
    with torch.no_grad():
        for images, targets_encoded, target_lengths in val_loader:
            images = images.to(device)
            targets_encoded = targets_encoded.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(images)
            outputs = outputs.permute(1, 0, 2) # CTC: [T, B, C]
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

            loss = criterion(outputs, targets_encoded, input_lengths, target_lengths)
            total_loss += loss.item()

            # Calculate CER
            probs = torch.softmax(outputs, dim=2)
            preds_indices = torch.argmax(probs, dim=2)

            # Iterasi per item di dalam batch (penting karena batch_size=1 untuk val_loader)
            for i in range(preds_indices.size(1)):
                # Decode prediksi
                pred_text = label_encoder.decode(preds_indices[:, i].cpu().numpy())
                # Decode ground truth
                true_text = "".join([label_encoder.idx2char.get(idx.item(), "") for idx in targets_encoded[i, :target_lengths[i]]])

                total_cer_distance += distance(pred_text, true_text)
                total_chars += len(true_text)

    avg_loss = total_loss / len(val_loader)
    avg_cer = total_cer_distance / total_chars if total_chars > 0 else 0
    return avg_loss, avg_cer # Return CER as well

def train():
    global best_val_loss, epochs_no_improve

    for epoch in range(config["training"]["epochs"]):
        # --- Training Loop ---
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} (Train)")
        for images, targets, target_lengths in pbar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2) # CTC: [T, B, C]
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        current_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {current_train_loss:.4f}")

        # --- Validation Loop ---
        val_loss, val_cer = validate(model, val_loader, criterion, device, label_encoder) # Get CER from validate
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val CER: {val_cer:.4f}")

        # Step the Learning Rate Scheduler
        scheduler.step(val_loss)

        # Early Stopping Logic
        if val_loss + min_delta < best_val_loss:
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
                break

if __name__ == "__main__":
    train()
