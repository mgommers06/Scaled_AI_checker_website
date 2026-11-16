import json, os, time, uuid
import pika, redis
from flask import Flask, request, render_template_string, send_from_directory

app = Flask(__name__, static_folder=None)

r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"))
RABBIT_HOST = os.getenv("RABBIT_HOST", "rabbitmq")

LIMITS = {
    "image": {"types": {"image/jpeg", "image/png", "image/webp"},
              "max_bytes": 10 * 1024 * 1024, "timeout": 30},
    "video": {"types": {"video/mp4", "video/quicktime", "video/webm", "video/x-msvideo"},
              "max_bytes": 50 * 1024 * 1024, "timeout": 120},
}

CLASS_NAMES = {
    0: {"label": "AI-Generated", "color": "#b30000", "description": "This media appears to be created by artificial intelligence."},
    1: {"label": "Human-Made", "color": "#00802b", "description": "This media appears to be an authentic, real recording."},
}

RESULT_TEMPLATE = """
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
            font-size: 34px; font-weight: bold; color: {{ result.color }}; margin-bottom: 10px;
        }
        .confidence {
            font-size: 24px; color: #382f2d; margin-top: 15px;
        }
        .disclaimer-box {
            background-color: #fff3f3; border-left: 5px solid #d9534f;
            padding: 20px; margin-bottom: 30px; border-radius: 4px; text-align: left;
        }
        .disclaimer-box h3 {
            color: #d9534f; margin-top: 0; margin-bottom: 10px; font-size: 22px;
        }
        .disclaimer-box p {
            font-size: 18px; color: #555; margin: 0; line-height: 1.5;
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
                <strong>Risks of use:</strong> This model may falsely flag a real, human-made recording as AI, or it may completely miss a sophisticated AI-generated file. <strong>Do not use this tool</strong> for legal, academic, or professional decisions.
            </p>
        </div>
        <a href="{{ back_url }}" class="action-button" style="text-align:center; display:block; text-decoration:none; box-sizing:border-box;">Check Another File</a>
        <a href="/index.html" class="secondary-button" style="text-align:center;">Back to Main Menu</a>
    </div>
</body>
</html>
"""


def render_result(result, confidence_percent, back_url):
    return render_template_string(
        RESULT_TEMPLATE, result=result,
        confidence_percent=confidence_percent, back_url=back_url,
    )


def publish_job(job_id, kind):
    conn = pika.BlockingConnection(pika.ConnectionParameters(RABBIT_HOST))
    ch = conn.channel()
    ch.queue_declare(queue="detect_jobs", durable=True)
    ch.basic_publish(
        exchange="", routing_key="detect_jobs",
        body=json.dumps({"job_id": job_id, "type": kind}),
        properties=pika.BasicProperties(delivery_mode=2),
    )
    conn.close()


def handle_upload(kind, back_url):
    cfg = LIMITS[kind]
    file_list = request.files.getlist("file")
    if not file_list or file_list[0].filename == "":
        return "No selected file", 400
    if len(file_list) > 1:
        return "Please upload only one file at a time.", 400

    file = file_list[0]
    if file.mimetype not in cfg["types"]:
        return f"Unsupported {kind} file type ({file.mimetype}).", 400

    data = file.read()
    if len(data) > cfg["max_bytes"]:
        return f"File too large (max {cfg['max_bytes'] // (1024 * 1024)} MB).", 413

    job_id = str(uuid.uuid4())
    try:
        r.setex(f"file:{job_id}", 600, data)      # auto-expires in 10 min
        publish_job(job_id, kind)
    except Exception:
        return "Service temporarily unavailable, please try again.", 503

    deadline = time.time() + cfg["timeout"]
    while time.time() < deadline:
        raw = r.get(f"result:{job_id}")
        if raw:
            r.delete(f"result:{job_id}")           # keep nothing
            res = json.loads(raw)
            if "error" in res:
                return f"Error processing {kind}: {res['error']}", 500
            return render_result(
                CLASS_NAMES[res["class_idx"]],
                f"{res['confidence'] * 100:.1f}%",
                back_url,
            )
        time.sleep(0.2)

    return "Analysis is taking too long, please try again.", 504


@app.route("/upload-photo", methods=["POST"])
def upload_photo():
    return handle_upload("image", "/photo.html")


@app.route("/upload-video", methods=["POST"])
def upload_video():
    return handle_upload("video", "/video.html")


@app.route("/")
@app.route("/<path:filename>")
def serve_static(filename="index.html"):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)