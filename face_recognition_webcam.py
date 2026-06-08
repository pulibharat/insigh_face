"""
Webcam Face Recognition with ArcFace (InsightFace)

Ethical & design notes:
- This system uses a PRETRAINED model (ArcFace via InsightFace). No model training is performed.
- Only face EMBEDDINGS (numeric vectors) are stored—not raw biometric images.
- No biometric data is shared externally.
- Recognition is similarity matching (cosine similarity), not classification.
- Face registration here means "enrollment" (storing one embedding per identity), not model training.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths for face database (embeddings + names)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
EMBEDDINGS_FILE = SCRIPT_DIR / "face_embeddings.npy"
NAMES_FILE = SCRIPT_DIR / "face_names.json"
# Optional: put photos here to enroll from files. Name = filename without extension (e.g. John.jpg → "John").
# Or use subfolders: known_faces/John/photo1.jpg → name "John".
KNOWN_FACES_DIR = SCRIPT_DIR / "known_faces"

# Recognition threshold: similarity above this → "Identified" with stored name.
# 0.5 is lenient (same person with different angle/light often 0.5–0.85). Use 0.6–0.7 if too many false matches.
SIMILARITY_THRESHOLD = 0.5

# Detection size for ArcFace (larger = more accurate, slower)
DET_SIZE = (640, 640)


def load_face_database():
    """
    Load face database from disk if files exist.
    Returns: dict mapping name (str) -> embedding (np.ndarray).
    """
    db = {}
    if EMBEDDINGS_FILE.exists() and NAMES_FILE.exists():
        try:
            embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True)
            with open(NAMES_FILE, "r", encoding="utf-8") as f:
                names = json.load(f)
            # names is list of names in same order as rows of embeddings
            if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                for i, name in enumerate(names):
                    if i < len(embeddings):
                        db[name] = embeddings[i].astype(np.float32)
        except Exception as e:
            print(f"[Warning] Could not load face database: {e}")
    return db


def save_face_database(db):
    """Save face database to disk: embeddings as .npy, names as JSON."""
    if not db:
        return
    names = list(db.keys())
    embeddings = np.array([db[name] for name in names], dtype=np.float32)
    np.save(EMBEDDINGS_FILE, embeddings, allow_pickle=False)
    with open(NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2, ensure_ascii=False)


def enroll_from_known_faces_folder(app, face_db):
    """
    Load face photos from KNOWN_FACES_DIR and add them to face_db.
    - Flat: known_faces/John.jpg → name "John"
    - Subfolders: known_faces/John/photo1.jpg → name "John" (first face found)
    """
    if not KNOWN_FACES_DIR.exists():
        return 0
    added = 0
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    # Flat: Name.jpg (filename without extension = person name)
    for path in KNOWN_FACES_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        name = path.stem
        if name in face_db:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        faces = app.get(img)
        if not faces or faces[0].normed_embedding is None:
            print(f"  [Skip] No face in {path.name}")
            continue
        face_db[name] = faces[0].normed_embedding.astype(np.float32)
        added += 1
        print(f"  Enrolled from file: {name} ({path.name})")
    # Subfolders: Name/photo.jpg
    for sub in KNOWN_FACES_DIR.iterdir():
        if not sub.is_dir() or sub.name in face_db:
            continue
        for path in sub.iterdir():
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            faces = app.get(img)
            if not faces or faces[0].normed_embedding is None:
                continue
            face_db[sub.name] = faces[0].normed_embedding.astype(np.float32)
            added += 1
            print(f"  Enrolled from folder: {sub.name} ({path.name})")
            break
    if added:
        save_face_database(face_db)
    return added


def get_arcface_model():
    """
    Initialize ArcFace model once (InsightFace buffalo_l).
    Prepared for CPU with detection size (640, 640).
    """
    from insightface.app import FaceAnalysis

    # buffalo_l includes detection + recognition (ArcFace); run on CPU
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)
    return app


def identify_face(embedding, face_db, threshold=SIMILARITY_THRESHOLD):
    """
    Compare one embedding to all stored embeddings using cosine similarity.
    Recognition is similarity matching, not classification.
    Returns (label, similarity): "Identified" with name, or "Unknown"; and best similarity.
    """
    if not face_db:
        return "Unknown", 0.0

    names = list(face_db.keys())
    stored = np.array([face_db[n] for n in names], dtype=np.float32)
    # embedding: (1, dim), stored: (n, dim) -> similarity (1, n)
    sim = cosine_similarity(embedding.reshape(1, -1), stored)[0]
    best_idx = int(np.argmax(sim))
    best_sim = float(sim[best_idx])

    if best_sim >= threshold:
        return names[best_idx], best_sim
    return "Unknown", best_sim


def main():
    print("Loading ArcFace model (buffalo_l)...")
    app = get_arcface_model()
    face_db = load_face_database()
    if face_db:
        print(f"Loaded {len(face_db)} registered face(s): {list(face_db.keys())}")
    # Enroll from known_faces folder: put photos here (filename = name, e.g. John.jpg → "John")
    if KNOWN_FACES_DIR.exists():
        print("Loading faces from 'known_faces' folder (filename = name)...")
        n = enroll_from_known_faces_folder(app, face_db)
        if n:
            print(f"Enrolled {n} face(s) from known_faces. Total: {list(face_db.keys())}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    print("\nControls:")
    print("  S - Register (enroll) current face with a name (manual)")
    print("  ESC or Q - Quit")
    print("  (Photos in 'known_faces' folder are used for recognition; filename = name, e.g. John.jpg)")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces (ArcFace returns bbox + embedding per face)
        faces = app.get(frame)

        # If no face detected, skip recognition; just show frame
        if not faces:
            cv2.imshow("Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            continue

        # Process each detected face independently
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Get ArcFace embedding (fixed-length vector for this face)
            emb = face.normed_embedding
            if emb is None:
                label, sim = "Unknown", 0.0
            else:
                emb = emb.reshape(1, -1)
                label, sim = identify_face(emb, face_db)

            # Draw bounding box
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Display label: show "Identified: Name" or "Unknown" (with best similarity so you can tune threshold)
            if label != "Unknown":
                text = f"Identified: {label} ({sim:.2f})"
            else:
                text = f"Unknown ({sim:.2f})"
            cv2.putText(
                frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") or key == ord("S"):
            # Face enrollment: capture current face and store with name
            if not faces:
                print("No face in frame. Move into view and press S again.")
                continue
            # Use the first/largest face for registration
            face = faces[0]
            emb = face.normed_embedding
            if emb is None:
                print("Could not extract embedding for this face.")
                continue
            name = input("Enter name for this person: ").strip()
            if not name:
                print("Name empty, registration skipped.")
                continue
            face_db[name] = emb.astype(np.float32)
            save_face_database(face_db)
            print(f"Registered: {name}")

        if key in (27, ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
