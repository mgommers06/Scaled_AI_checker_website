
import torch
from PIL import Image

from image_classifier import ImageCNN, test_transforms, dataset



model = ImageCNN(num_classes=2)

model.load_state_dict(
    torch.load("trained_model.pth", map_location="cpu", weights_only=True)
)

model.eval()

image_path = "data_images/fake_dataset/image-2.png"

def test_image_exists():
    import os
    assert os.path.exists(image_path)

def test_image_classification():
    image = Image.open(image_path).convert("RGB")
    image = test_transforms(image)

    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

   
    prediction = dataset.classes[predicted_class.item()]

    assert prediction in dataset.classes
    assert 0 <= confidence.item() <= 1
