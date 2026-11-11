import io
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, render_template_string
import torchvision.transforms as transforms

app = Flask(__name__)

# 1. Define the exact ImageCNN architecture
class ImageCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2. Instantiate and load the model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCNN(num_classes=2)
try:
    model.load_state_dict(torch.load("image_cnn_weights.pth", map_location=device))
    print("Model weights loaded successfully.")
except FileNotFoundError:
    print("Warning: image_cnn_weights.pth not found. Train and save your weights first.")

model.to(device)
model.eval()

# 3. Match the test_transforms
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.45, 0.45, 0.45),
        (0.225, 0.225, 0.225)
    )
])

CLASS_NAMES = {
    0: {"label": "AI-Generated", "color": "#b30000", "description": "This media appears to be created by artificial intelligence."},
    1: {"label": "Human-Made", "color": "#00802b", "description": "This media appears to be an authentic, real recording."}
}

# 4. Route for the submission
@app.route("/upload-photo", methods=["POST"])
@app.route("/upload-video", methods=["POST"])  # Added video route for future-proofing
def upload_media():
    if "file" not in request.files:
        return "No file uploaded", 400

    # Ensure only ONE file is processed, even if someone attempts a multi-file upload bypass
    file_list = request.files.getlist("file")
    if len(file_list) > 1:
        return "Please upload only one file at a time.", 400
        
    file = file_list[0]
    if file.filename == "":
        return "No selected file", 400

    try:
        # Read and prepare image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = inference_transforms(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        result = CLASS_NAMES[predicted_idx.item()]
        confidence_percent = f"{confidence.item() * 100:.1f}%"

        # Includes the new Confidence Score, Disclaimer, and Risk Warning
        return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Analysis Result</title>
                <link rel="stylesheet" href="/style.css">
                <style>
                    .result-box {
                        border: 3px solid {{ result.color }};
                        border-radius: 12px;
                        padding: 30px;
                        margin: 25px 0 15px;
                        background-color: #faf6ef;
                        text-align: center;
                    }
                    .result-tag {
                        font-size: 34px;
                        font-weight: bold;
                        color: {{ result.color }};
                        margin-bottom: 10px;
                    }
                    .confidence {
                        font-size: 24px;
                        color: #382f2d;
                        margin-top: 15px;
                    }
                    .disclaimer-box {
                        background-color: #fff3f3;
                        border-left: 5px solid #d9534f;
                        padding: 20px;
                        margin-bottom: 30px;
                        border-radius: 4px;
                        text-align: left;
                    }
                    .disclaimer-box h3 {
                        color: #d9534f;
                        margin-top: 0;
                        margin-bottom: 10px;
                        font-size: 22px;
                    }
                    .disclaimer-box p {
                        font-size: 18px;
                        color: #555;
                        margin: 0;
                        line-height: 1.5;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Analysis Result</h1>
                    <hr class="divider">
                    
                    <div class="result-box">
                        <div class="result-tag">{{ result.label }}</div>
                        <p style="font-size: 20px;">{{ result.description }}</p>
                        <p class="confidence">Confidence Score: <strong>{{ confidence_percent }}</strong></p>
                    </div>

                    <div class="disclaimer-box">
                        <h3>⚠️ Important Disclaimer</h3>
                        <p>
                            <strong>This tool is not 100% accurate.</strong> AI detection technology is experimental and frequently makes mistakes. 
                            <br><br>
                            <strong>Risks of use:</strong> This model may falsely flag a real, human-made photo as AI (a false positive), or it may completely miss a sophisticated AI-generated image (a false negative). <strong>Do not use this tool</strong> as definitive proof of authenticity for legal, academic, financial, or professional decisions.
                        </p>
                    </div>

                    <a href="/index.html" class="action-button" style="text-align:center; display:block; text-decoration:none; box-sizing:border-box;">Check Another File</a>
                </div>
            </body>
            </html>
        """, result=result, confidence_percent=confidence_percent)

    except Exception as e:
        return f"Error processing file: {str(e)}", 500

# 5. Serve static HTML/CSS files
@app.route("/")
@app.route("/<path:filename>")
def serve_static(filename="index.html"):
    from flask import send_from_directory
    return send_from_directory(".", filename)

if __name__ == "__main__":
    print(">>> Starting the web server...")
    app.run(debug=True, host="127.0.0.1", port=5000)