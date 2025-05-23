import os
import torch
import cv2
import argparse
import yaml
from PIL import Image
import torchvision.transforms as T
from crnn_model import CRNN
from utils.label_encoder import LabelEncoder

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def preprocess_image(image, img_h, img_w):
    transform = T.Compose([
        T.Grayscale(),
        T.Resize((img_h, img_w)),
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # [1, 1, H, W]

def predict(model, image_tensor, label_encoder, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)  # [B, T, C]
        output = output.permute(1, 0, 2)  # [T, B, C]
        probs = torch.argmax(output, dim=2)
        probs = probs.squeeze(1).detach().cpu().numpy()
        text = label_encoder.decode(probs)
        return text

def infer_on_folder(config, model, label_encoder, device):
    image_dir = config["inference"]["image_folder"]
    result_path = config["inference"]["result_path"]
    img_h = config["dataset"]["img_height"]
    img_w = config["dataset"]["img_width"]

    with open(result_path, "w", encoding="utf-8") as out_file:
        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(image_dir, filename)
                image = Image.open(path).convert("RGB")
                image_tensor = preprocess_image(image, img_h, img_w)
                pred = predict(model, image_tensor, label_encoder, device)
                out_file.write(f"{filename} -> {pred}\n")
                print(f"{filename}: {pred}")

def infer_from_camera(config, model, label_encoder, device):
    cap = cv2.VideoCapture(0)
    img_h = config["dataset"]["img_height"]
    img_w = config["dataset"]["img_width"]

    if not cap.isOpened():
        print("❌ Kamera tidak ditemukan")
        return

    print("📸 Tekan 'q' untuk keluar")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = preprocess_image(pil_image, img_h, img_w)
        pred = predict(model, image_tensor, label_encoder, device)

        # Tampilkan teks hasil OCR
        cv2.putText(frame, pred, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("OCR Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action="store_true", help="Gunakan kamera untuk uji OCR")
    args = parser.parse_args()

    config = load_config()
    device = torch.device(config["device"])
    label_encoder = LabelEncoder(config["dataset"]["charset"])
    num_classes = len(config["dataset"]["charset"]) + 1

    model = CRNN(config["dataset"]["img_height"], 1, num_classes)
    model.load_state_dict(torch.load(config["training"]["model_save_path"], map_location=device))
    model = model.to(device)

    if args.camera:
        infer_from_camera(config, model, label_encoder, device)
    else:
        infer_on_folder(config, model, label_encoder, device)
