import os
import gc
import shutil
import random
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import v2
import cv2

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
VALID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

SEARCH_CANDIDATES = [
    HERE / "data_sample",
    HERE.parent / "data_sample",
    Path("./data_sample"),
    Path("../data_sample"),
    Path("/home/jupyter/S.E processing/data_sample"),
    Path("/home/jupyter/data_sample"),
]


# --- Helpers (no test_ prefix, so pytest ignores them) ---

def find_video_dir():
    """Return the first existing data_sample folder, or None."""
    for candidate in SEARCH_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def extract_keyframes(video_path, output_dir, frame_interval=15):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_count, saved_count = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(
                output_dir, f"{video_name}_frame{saved_count:04d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    return saved_count


def extract_all_frames(video_dir, output_dir, frame_interval=15):
    """Walk video_dir, extract frames into output_dir/fake and output_dir/real.
    Returns (videos, real_count, fake_count)."""
    real_count, fake_count = 0, 0
    videos = []

    for root, _, files in os.walk(video_dir):
        for file in files:
            if not file.lower().endswith(VALID_EXTENSIONS):
                continue
            full_path = os.path.join(root, file)
            videos.append(full_path)

            path_lower = full_path.lower()
            if "fake" in path_lower:
                dst = os.path.join(output_dir, "fake")
                fake_count += extract_keyframes(full_path, dst, frame_interval)
            elif "real" in path_lower or "photos" in path_lower:
                dst = os.path.join(output_dir, "real")
                real_count += extract_keyframes(full_path, dst, frame_interval)

    return videos, real_count, fake_count


def make_dummy_video(path, n_frames=31, size=(64, 64), fps=10):
    """Create a small synthetic video so tests don't need real data."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size
    )
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i * 8 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


# --- Tests ---

def test_extract_keyframes_invalid_file_returns_zero(tmp_path):
    saved = extract_keyframes(tmp_path / "does_not_exist.mp4", tmp_path / "out")
    assert saved == 0


def test_extract_keyframes_saves_every_nth_frame(tmp_path):
    video = tmp_path / "clip.avi"
    make_dummy_video(video, n_frames=31)          # frames 0, 15, 30 are kept

    out_dir = tmp_path / "frames"
    saved = extract_keyframes(video, out_dir, frame_interval=15)

    assert saved == 3
    assert len(list(out_dir.glob("*.jpg"))) == 3


def test_extract_all_frames_sorts_into_real_and_fake(tmp_path):
    (tmp_path / "fake").mkdir()
    (tmp_path / "real").mkdir()
    make_dummy_video(tmp_path / "fake" / "a.avi", n_frames=31)
    make_dummy_video(tmp_path / "real" / "b.avi", n_frames=16)   # frames 0, 15

    out_dir = tmp_path / "video_frames"
    videos, real_count, fake_count = extract_all_frames(tmp_path, out_dir)

    assert len(videos) == 2
    assert fake_count == 3
    assert real_count == 2
    assert len(list((out_dir / "fake").glob("*.jpg"))) == 3
    assert len(list((out_dir / "real").glob("*.jpg"))) == 2


def test_real_data_sample_produces_frames(tmp_path):
    video_dir = find_video_dir()
    if video_dir is None:
        pytest.skip("data_sample folder not found")

    videos, real_count, fake_count = extract_all_frames(video_dir, tmp_path / "frames")

    assert len(videos) > 0, "no videos found in data_sample"
    assert real_count + fake_count > 0

import os
import random

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# Helpers from the first test file (same tests/ folder). If you rename that
# file, update this import. Better long term: move the helpers into a normal
# module such as pipeline.py next to app.py and import from there.
from test_Etests import extract_all_frames, make_dummy_video  # noqa: F401

CLASS_TO_IDX = {"fake": 0, "real": 1}
ALLOWED_EXTENSIONS = {".mp4", ".avi"}
NORMALIZE = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# =====================================================================
# Pipeline code (no test_ prefix, nothing runs on import)
# =====================================================================

# --- Stratified group split -------------------------------------------------

def build_video_index(frames_dir):
    """Scan frames_dir/{fake,real} and group frame paths by video id.
    Returns (video_to_frames, real_video_ids, fake_video_ids)."""
    video_to_frames = {}
    real_video_ids, fake_video_ids = [], []

    for category in ["fake", "real"]:
        folder = os.path.join(frames_dir, category)
        if not os.path.exists(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            fpath = os.path.join(folder, fname)
            video_id = fname.split("_frame")[0]
            label = CLASS_TO_IDX[category]

            if video_id not in video_to_frames:
                video_to_frames[video_id] = []
                if category == "real":
                    real_video_ids.append(video_id)
                else:
                    fake_video_ids.append(video_id)
            video_to_frames[video_id].append((fpath, label))

    return video_to_frames, real_video_ids, fake_video_ids


def split_ids(id_list):
    n = len(id_list)
    tr = max(1, int(0.70 * n))
    va = tr + max(1, int(0.15 * n)) if n > 2 else tr
    return (
        id_list[:tr],
        (id_list[tr:va] if n > 1 else id_list[:tr]),
        (id_list[va:] if n > 2 else id_list[tr:va]),
    )


def split_videos(real_video_ids, fake_video_ids, seed=42):
    """Shuffle (deterministically) and split by video. Returns train/val/test video ids."""
    real_ids, fake_ids = list(real_video_ids), list(fake_video_ids)
    rng = random.Random(seed)          # local RNG: doesn't touch global random state
    rng.shuffle(real_ids)
    rng.shuffle(fake_ids)

    real_train, real_val, real_test = split_ids(real_ids)
    fake_train, fake_val, fake_test = split_ids(fake_ids)
    return (
        real_train + fake_train,
        real_val + fake_val,
        real_test + fake_test,
    )


def gather_dataset_lists(vid_list, video_to_frames):
    paths, labels = [], []
    for vid_id in vid_list:
        for fpath, lbl in video_to_frames[vid_id]:
            paths.append(fpath)
            labels.append(lbl)
    return paths, labels


# --- Dataset & transforms ---------------------------------------------------

class VideoFrameDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


train_transform = v2.Compose([
    v2.Resize((128, 128)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(**NORMALIZE),
])

val_test_transform = v2.Compose([
    v2.Resize((128, 128)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(**NORMALIZE),
])


def make_loaders(train_paths, train_labels, val_paths, val_labels):
    if len(train_paths) == 0:
        raise ValueError("train_paths is empty! Extract frames first.")

    train_ds = VideoFrameDataset(train_paths, train_labels, train_transform)
    val_ds = VideoFrameDataset(val_paths, val_labels, val_test_transform)

    train_loader = DataLoader(
        train_ds, batch_size=max(1, min(8, len(train_ds))), shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, min(8, len(val_ds))), shuffle=False, num_workers=0
    )
    return train_ds, val_ds, train_loader, val_loader


# --- Model ------------------------------------------------------------------

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


# --- Training ---------------------------------------------------------------

def train_model(model, train_loader, val_loader, train_ds, val_ds, device, epochs=15, lr=1e-4):
    """Train and return a list of (train_acc_percent, val_acc_percent), one per epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []

    for epoch in range(epochs):
        model.train()
        correct_train = 0
        for inputs, lbls in train_loader:
            inputs, lbls = inputs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == lbls).sum().item()

        train_acc = (correct_train / len(train_ds)) * 100 if len(train_ds) > 0 else 0

        model.eval()
        correct_val = 0
        with torch.no_grad():
            for inputs, lbls in val_loader:
                inputs, lbls = inputs.to(device), lbls.to(device)
                _, preds = torch.max(model(inputs), 1)
                correct_val += (preds == lbls).sum().item()

        val_acc = (correct_val / len(val_ds)) * 100 if len(val_ds) > 0 else 0
        history.append((train_acc, val_acc))

    return history


# --- File format validation -------------------------------------------------

def is_valid_video_format(filename):
    if not isinstance(filename, str) or not filename.strip():
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


# --- Video-level prediction & metrics --------------------------------------

def predict_video(model, video_path, device, frame_interval=15):
    """Average P(real) over every Nth frame. Returns None if the video can't be read
    or yields no frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_probs = []
    frame_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = val_test_transform(img).unsqueeze(0).to(device)
                frame_probs.append(F.softmax(model(tensor), dim=1)[0][1].item())
            frame_count += 1
    cap.release()

    return float(np.mean(frame_probs)) if frame_probs else None


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def frames_dir(tmp_path):
    """4 fake + 4 real videos, 2 tiny frames each, laid out like ./video_frames."""
    rng = np.random.default_rng(0)
    root = tmp_path / "video_frames"
    for category in ["fake", "real"]:
        (root / category).mkdir(parents=True)
        for i in range(4):
            for j in range(2):
                img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(root / category / f"{category}{i}_frame{j:04d}.jpg"), img)
    return root


@pytest.fixture
def index(frames_dir):
    return build_video_index(str(frames_dir))


# =====================================================================
# Tests: split
# =====================================================================

def test_build_video_index_groups_frames_by_video(index):
    video_to_frames, real_ids, fake_ids = index

    assert len(real_ids) == 4
    assert len(fake_ids) == 4
    assert all(len(frames) == 2 for frames in video_to_frames.values())
    assert all(lbl == 0 for _, lbl in video_to_frames["fake0"])
    assert all(lbl == 1 for _, lbl in video_to_frames["real0"])


def test_build_video_index_missing_folder_returns_empty(tmp_path):
    video_to_frames, real_ids, fake_ids = build_video_index(str(tmp_path / "nope"))
    assert video_to_frames == {} and real_ids == [] and fake_ids == []


def test_split_ids_is_disjoint_and_complete():
    ids = [f"v{i}" for i in range(10)]
    train, val, test = split_ids(ids)

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert sorted(train + val + test) == sorted(ids)


def test_split_videos_is_reproducible(index):
    _, real_ids, fake_ids = index
    assert split_videos(real_ids, fake_ids, seed=42) == split_videos(real_ids, fake_ids, seed=42)


def test_no_video_appears_in_two_splits(index):
    """The whole point of a group split: frames of one video never leak across sets."""
    _, real_ids, fake_ids = index
    train, val, test = split_videos(real_ids, fake_ids)

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)


def test_gather_dataset_lists_matches_video_frames(index):
    video_to_frames, _, _ = index
    paths, labels = gather_dataset_lists(["fake0", "real0"], video_to_frames)

    assert len(paths) == len(labels) == 4
    assert sorted(set(labels)) == [0, 1]


# =====================================================================
# Tests: dataset & model
# =====================================================================

def test_dataset_item_has_expected_shape_and_label(index):
    video_to_frames, _, _ = index
    paths, labels = gather_dataset_lists(["real0"], video_to_frames)
    ds = VideoFrameDataset(paths, labels, val_test_transform)

    img, label = ds[0]
    assert len(ds) == 2
    assert img.shape == (3, 128, 128)
    assert label == 1


def test_make_loaders_rejects_empty_training_set():
    with pytest.raises(ValueError):
        make_loaders([], [], [], [])


def test_cnn_output_shape():
    model = CustomCNN().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 2)


def test_training_runs_and_reports_valid_accuracies(index):
    video_to_frames, real_ids, fake_ids = index
    train_vids, val_vids, _ = split_videos(real_ids, fake_ids)
    train_paths, train_labels = gather_dataset_lists(train_vids, video_to_frames)
    val_paths, val_labels = gather_dataset_lists(val_vids, video_to_frames)

    train_ds, val_ds, train_loader, val_loader = make_loaders(
        train_paths, train_labels, val_paths, val_labels
    )
    torch.manual_seed(0)
    model = CustomCNN()
    history = train_model(
        model, train_loader, val_loader, train_ds, val_ds, torch.device("cpu"), epochs=1
    )

    assert len(history) == 1
    train_acc, val_acc = history[0]
    assert 0 <= train_acc <= 100
    assert 0 <= val_acc <= 100


# =====================================================================
# Tests: file format validation
# =====================================================================

@pytest.mark.parametrize("name", ["sample_video.mp4", "clip.avi", "UPPERCASE_TEST.MP4"])
def test_allowed_formats_pass(name):
    assert is_valid_video_format(name)


@pytest.mark.parametrize(
    "name", ["document.pdf", "script.py", "picture.png", "unsupported.mkv"]
)
def test_unsupported_formats_fail(name):
    assert not is_valid_video_format(name)


@pytest.mark.parametrize("name", ["no_extension", "", "   ", None, 123])
def test_edge_cases_fail(name):
    assert not is_valid_video_format(name)


# =====================================================================
# Tests: video prediction & metrics
# =====================================================================

def test_predict_video_returns_probability(tmp_path):
    video = tmp_path / "clip.avi"
    make_dummy_video(video, n_frames=31)

    torch.manual_seed(0)
    prob = predict_video(CustomCNN(), video, torch.device("cpu"), frame_interval=15)

    assert prob is not None
    assert 0.0 <= prob <= 1.0


def test_predict_video_unreadable_file_returns_none(tmp_path):
    assert predict_video(CustomCNN(), tmp_path / "missing.mp4", torch.device("cpu")) is None


def test_compute_metrics_known_values():
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 1]      # tp=2, fn=1, tn=2, fp=1
    m = compute_metrics(y_true, y_pred)

    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_compute_metrics_perfect_predictions():
    m = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1])
    assert m == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_compute_metrics_empty_input_does_not_crash():
    m = compute_metrics([], [])
    assert m == {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}