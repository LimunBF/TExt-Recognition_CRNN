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
from Levenshtein import distance  # untuk hitung CER

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return images, labels, torch.tensor(lengths, dtype=torch.long)

def ctc_decode(preds, blank=0):
    """
    Decode output from argmax for CTC:
    - Remove duplicates and blanks.
    """
    decoded = []
    prev = blank
    for p in preds:
        if p != blank and p != prev:
            decoded.append(p)
        prev = p
    return decoded

def train():
    # Load konfigurasi
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Set seed & device
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = torch.device(config["device"])

    # Dataset & DataLoader
    le = LabelEncoder(config["dataset"]["charset"])
    train_ds = OCRDataset(
        config["dataset"]["train_labels"],
        config["dataset"]["img_height"],
        config["dataset"]["img_width"],
        le
    )
    val_ds = OCRDataset(
        config["dataset"]["val_labels"],
        config["dataset"]["img_height"],
        config["dataset"]["img_width"],
        le
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Model, loss, optimizer
    num_classes = len(config["dataset"]["charset"]) + 1  # +1 for blank
    model = CRNN(
        config["dataset"]["img_height"], 1, num_classes
    ).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Early stopping & scheduler
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = config["training"]["early_stopping_patience"]
    min_delta = config["training"]["early_stopping_min_delta"]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=5, factor=0.5, min_lr=1e-6
    )

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        total_loss = 0
        for images, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch} (Train)"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)                           # [b, T, C]
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # [T, b, C]
            input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
            loss = criterion(log_probs, labels, input_lengths, lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

        # Validasi
        model.eval()
        val_loss = 0
        total_cer = 0
        with torch.no_grad():
            for images, labels, lengths in tqdm(val_loader, desc=f"Epoch {epoch} (Val)"):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                log_probs = logits.log_softmax(2).permute(1, 0, 2)
                input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
                loss = criterion(log_probs, labels, input_lengths, lengths)
                val_loss += loss.item()
                # Decode prediksi & hitung CER dengan ctc_decode
                preds_raw = logits.argmax(2)[0].cpu().numpy().tolist()
                preds = ctc_decode(preds_raw, blank=0)
                pred_text = le.decode(preds)
                true_text = le.decode(labels[0, :lengths[0]].cpu().numpy().tolist())
                total_cer += distance(pred_text, true_text) / max(len(true_text), 1)
        avg_val_loss = val_loss / len(val_loader)
        avg_cer = total_cer / len(val_loader)
        print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}, Val CER: {avg_cer:.4f}")

        # Contoh prediksi debug tiap 5 epoch
        if epoch % 5 == 0:
            print(f"Sample prediction: '{pred_text}'")
            print(f"Sample ground truth: '{true_text}'")

        # Scheduler & early stopping
        scheduler.step(avg_val_loss)
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config["training"]["model_save_path"])
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break

if __name__ == "__main__":
    train()
