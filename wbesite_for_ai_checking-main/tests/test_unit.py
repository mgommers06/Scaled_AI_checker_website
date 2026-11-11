import hashlib, io
from unittest.mock import patch
from PIL import Image
import pytest
from app import app   

@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()

def sha256(b): 
    return hashlib.sha256(b).hexdigest()

def test_route_receives_same_image_bytes(client):
    with open("tests/fixtures/dogs.jpg", "rb") as f:
        original = f.read()

    received = {}
    real_open = Image.open

    def spy_open(fp, *args, **kwargs):
        received["bytes"] = fp.getvalue()   
        return real_open(fp, *args, **kwargs)

    with patch("app.Image.open", side_effect=spy_open):
        resp = client.post(
            "/upload-photo",
            data={"file": (io.BytesIO(original), "cat.jpg")},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 200
    assert sha256(received["bytes"]) == sha256(original)
    