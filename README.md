# Webcam Face Recognition with ArcFace (InsightFace)

Webcam-based face recognition using **InsightFace** (ArcFace). Register a face via webcam, then get real-time identification with cosine similarity matching.

## Features

- **Face enrollment**: Register by pressing **S** (webcam) or put photos in the **`known_faces`** folder (filename = name, e.g. `John.jpg` → "John").
- **Real-time recognition**: Live webcam shows "Identified: &lt;name&gt;" or "Unknown (similarity)" above each face.
- **Persistent database**: Embeddings saved to `face_embeddings.npy` and `face_names.json` (loaded at startup).

## Setup

### 1. Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

**On Windows** (avoids "Microsoft Visual C++ 14.0 required" build error):

```bash
pip install -r requirements.txt
pip install -r requirements-windows.txt
```

`requirements-windows.txt` installs a pre-built InsightFace wheel from [Hugging Face](https://huggingface.co/ussoewwin/Insightface_for_windows). For Python 3.11 or 3.12, edit that file to use the matching wheel URL from the repo.

**On Linux/macOS** (or if you have Visual C++ Build Tools):

```bash
pip install -r requirements.txt
pip install insightface
```

First run will download the InsightFace `buffalo_l` model (~326MB) automatically.

## Run

```bash
python face_recognition_webcam.py
```

### Controls

| Key   | Action                          |
|-------|----------------------------------|
| **S** | Register (enroll) current face   |
| **ESC** / **Q** | Quit                    |

When you press **S**, type the person’s name in the **terminal** and press Enter.

**Enroll from saved photos:** Create a folder `known_faces` next to the script and add one photo per person. The **filename (without extension) = name** (e.g. `John.jpg` → "John"). On startup, the app loads these and adds them to the database. If recognition still shows "Unknown", the similarity score is shown (e.g. "Unknown (0.45)"); you can lower the threshold in code (`SIMILARITY_THRESHOLD` in `face_recognition_webcam.py`, default 0.5).

## How it works

- **ArcFace** (via InsightFace `buffalo_l`) gives a fixed-length embedding per face.
- **Recognition** = cosine similarity between live embedding and stored embeddings (threshold 0.8).
- **Face registration** here means enrollment (storing one embedding per identity), not model training.
- Only **embeddings** are stored; no raw images are saved. The model is pretrained; no training is done in this app.

## Expected output

1. Webcam window opens.
2. Press **S** to register; enter name in terminal.
3. Stored face is recognized in real time with "Identified: &lt;name&gt;" and similarity score.
4. Unknown faces show "Unknown".

## Next extensions (for later)

- Multiple users registration (already supported: press **S** for each person).
- Save/load face database (already implemented).
- Webcam → video file recognition.
- FAISS for large-scale matching.
- Missing person detection.
