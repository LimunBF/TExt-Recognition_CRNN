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
    shuffle=True, num_workers=config["training"]["num_workers"],
    collate_fn=collate_fn)

val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

# Model
model = CRNN(img_h, 1, num_classes).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

def train():
    model.train()
    for epoch in range(config["training"]["epochs"]):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
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
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")

        # Save checkpoint setiap epoch
        torch.save(model.state_dict(), config["training"]["model_save_path"])

if __name__ == "__main__":
    train()
