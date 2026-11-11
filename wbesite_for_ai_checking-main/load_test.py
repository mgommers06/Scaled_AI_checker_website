import requests, time, concurrent.futures

URL = "http://localhost:5000/upload-photo"
IMG = "image-2 copy.png"   # change to the name of your image

def send(_):
    with open(IMG, "rb") as f:
        files = {"file": ("test.png", f, "image/png")}
        r = requests.post(URL, files=files)
        return r.status_code

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
    results = list(ex.map(send, range(50)))
print(results.count(200), "ok out of 50 in", round(time.time() - start, 1), "s")