import sys, time, mimetypes, concurrent.futures
import requests

# Usage: python load_test.py <file> [photo|video] [requests]
FILE = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
KIND = sys.argv[2] if len(sys.argv) > 2 else "photo"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 50

URL = f"http://localhost:5000/upload-{KIND}"
MIME = mimetypes.guess_type(FILE)[0] or "application/octet-stream"

def send(_):
    with open(FILE, "rb") as f:
        r = requests.post(URL, files={"file": (FILE, f, MIME)}, timeout=300)
        return r.status_code

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
    results = list(ex.map(send, range(N)))
elapsed = time.time() - start

print(f"{results.count(200)} ok out of {N} in {elapsed:.1f} s")
print("status codes:", set(results))
print(f"throughput: {N / elapsed:.1f} requests/s")