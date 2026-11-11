
import io
import unittest
from unittest.mock import patch
import torch
from PIL import Image
import app as app_module


class TestImageUpload(unittest.TestCase):

    def setUp(self):
        self.client = app_module.app.test_client()

    def create_image(self, image_format):
        image = Image.new("RGB", (100, 100), color="red")
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image_format)
        image_bytes.seek(0)
        return image_bytes

    def upload_valid_image(self, image_format, filename):
        # stoping the AI from processing the image
        with patch.object(
            app_module.model,
            "forward",
            return_value=torch.tensor([[2.0, 1.0]])
        ):
            return self.client.post(
                "/upload-photo",
                data={
                    "file": (
                        self.create_image(image_format),
                        filename
                    )
                },
                content_type="multipart/form-data"
            )

    #TEST 1 -JPEG
    def test_valid_jpg(self):
        response = self.upload_valid_image("JPEG", "photo.jpg")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analysis Result", response.data)

    #TEST 2 -PNG
    def test_valid_png(self):
        response = self.upload_valid_image("PNG", "photo.png")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analysis Result", response.data)

    # TEST 3 - PDF
    def test_invalid_pdf(self):
        response = self.client.post(
            "/upload-photo",
            data={
                "file": (io.BytesIO(b"%PDF-1.4"), "document.pdf")
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn(b"Error processing file", response.data)

    # TEST 4 - TXT
    def test_invalid_txt(self):
        response = self.client.post(
            "/upload-photo",
            data={
                "file": (io.BytesIO(b"Hello world"), "document.txt")
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn(b"Error processing file", response.data)

    # TEST 5 - fake JPG
    def test_fake_jpg(self):
        response = self.client.post(
            "/upload-photo",
            data={
                "file": (
                    io.BytesIO(b"This is not a real image"),
                    "fake.jpg"
                )
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn(b"Error processing file", response.data)

    # TEST 6 - corrupted image
    def test_corrupted_image(self):
        response = self.client.post(
            "/upload-photo",
            data={
                "file": (
                    io.BytesIO(b"\x89PNG corrupted data"),
                    "broken.png"
                )
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn(b"Error processing file", response.data)

    # TEST 7 - no file uploaded
    def test_no_file(self):
        response = self.client.post(
            "/upload-photo",
            data={},
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"No file uploaded", response.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
