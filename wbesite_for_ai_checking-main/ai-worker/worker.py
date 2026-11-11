import io, json, os, time
import pika, redis
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"))
RABBIT_HOST = os.getenv("RABBIT_HOST", "rabbitmq")
MODEL_PATH = os.getenv("MODEL_PATH", "trained_model.pth")


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


# LOADING: happens once when the worker starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCNN(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("Model weights loaded successfully.", flush=True)

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
])


def on_job(ch, method, props, body):
    job_id = json.loads(body)["job_id"]
    try:
        data = r.get(f"file:{job_id}")
        if data is None:
            raise ValueError("file expired or missing")
        image = Image.open(io.BytesIO(data)).convert("RGB")
        tensor = inference_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():                      # INFERENCE
            probs = torch.softmax(model(tensor), dim=1)[0]
            confidence, idx = torch.max(probs, dim=0)

        out = {"class_idx": idx.item(), "confidence": confidence.item()}
    except Exception as e:
        out = {"error": str(e)}
    finally:
        r.delete(f"file:{job_id}")

    r.setex(f"result:{job_id}", 600, json.dumps(out))
    ch.basic_ack(delivery_tag=method.delivery_tag)   # ack only when finished


def main():
    while True:                                    # wait for RabbitMQ to be ready
        try:
            conn = pika.BlockingConnection(pika.ConnectionParameters(RABBIT_HOST))
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