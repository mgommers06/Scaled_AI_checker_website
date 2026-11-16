import io, json, os, tempfile, time
import cv2
import numpy as np
import pika, redis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"))
RABBIT_HOST = os.getenv("RABBIT_HOST", "rabbitmq")
IMAGE_WEIGHTS = os.getenv("IMAGE_WEIGHTS", "trained_model.pth")
VIDEO_WEIGHTS = os.getenv("VIDEO_WEIGHTS", "video_cnn_weights.pth")
FRAME_STEP = 15          # analyse 1 frame out of every 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Photo model ----------
class ImageCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------- Video model ----------
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# LOADING: both models, once, at startup
photo_model = ImageCNN(num_classes=2).to(device)
photo_model.load_state_dict(torch.load(IMAGE_WEIGHTS, map_location=device))
photo_model.eval()

video_model = CustomCNN().to(device)
video_model.load_state_dict(torch.load(VIDEO_WEIGHTS, map_location=device))
video_model.eval()
print("Photo and video models loaded successfully.", flush=True)

photo_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
])
video_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image(data):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = photo_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(photo_model(tensor), dim=1)[0]
        confidence, idx = torch.max(probs, dim=0)
    return {"class_idx": idx.item(), "confidence": confidence.item()}


def predict_video(data):
    # OpenCV needs a file on disk, so use a temp file and always remove it
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(data)

        cap = cv2.VideoCapture(path)
        real_probs, count = [], 0
        with torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if count % FRAME_STEP == 0:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    x = video_transforms(img).unsqueeze(0).to(device)
                    probs = F.softmax(video_model(x), dim=1)
                    real_probs.append(probs[0][1].item())   # index 1 = Real
                count += 1
        cap.release()
    finally:
        if os.path.exists(path):
            os.remove(path)

    if not real_probs:
        raise ValueError("Could not extract any frames from video.")

    avg_real = float(np.mean(real_probs))
    verdict = 1 if avg_real > 0.5 else 0
    return {"class_idx": verdict, "confidence": max(avg_real, 1.0 - avg_real)}


def on_job(ch, method, props, body):
    msg = json.loads(body)
    job_id, kind = msg["job_id"], msg.get("type", "image")
    try:
        data = r.get(f"file:{job_id}")
        if data is None:
            raise ValueError("file expired or missing")
        out = predict_video(data) if kind == "video" else predict_image(data)
    except Exception as e:
        out = {"error": str(e)}
    finally:
        r.delete(f"file:{job_id}")

    r.setex(f"result:{job_id}", 600, json.dumps(out))
    ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    while True:
        try:
            conn = pika.BlockingConnection(
                pika.ConnectionParameters(RABBIT_HOST, heartbeat=600,
                                          blocked_connection_timeout=600))
            break
        except pika.exceptions.AMQPConnectionError:
            print("Waiting for RabbitMQ...", flush=True)
            time.sleep(3)

    ch = conn.channel()
    ch.queue_declare(queue="detect_jobs", durable=True)
    ch.basic_qos(prefetch_count=1)
    ch.basic_consume(queue="detect_jobs", on_message_callback=on_job)
    print("Worker ready.", flush=True)
    ch.start_consuming()


if __name__ == "__main__":
    main()