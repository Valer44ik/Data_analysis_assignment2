"""Real-time emotion recognition demo — Group 2, Assignment 2 (Task 6).

Opens the webcam, detects faces with OpenCV's Haar cascade, converts every face to the FER2013
format (crop with margin -> gray -> 48x48), extracts exactly the features the model was trained
with (the feature spec is stored inside the .pkl bundle) and shows the predicted emotion.

Usage (from this folder, in the main environment):
    python demo_emotion.py                       # best model (models/final.pkl), camera 0
    python demo_emotion.py --model models/task1_SVM.pkl --camera 1
    python demo_emotion.py --image some_face.jpg # still image, no camera needed

Keys in the camera window:  q = quit,  s = toggle temporal smoothing,  + / - = crop margin.
"""
import argparse
import collections
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import features as F  # noqa: E402

DEFAULT_MODELS = ["models/final.pkl", "models/task1_SVM.pkl", "models/task1_LR.pkl", "models/task1_RF.pkl"]
COLORS = [(40, 40, 220), (40, 160, 40), (160, 40, 160), (0, 210, 255), (200, 120, 40), (255, 200, 0), (180, 180, 180)]


def load_bundle(path=None):
    candidates = [path] if path else DEFAULT_MODELS
    for c in candidates:
        p = ROOT / c
        if p.exists():
            bundle = joblib.load(p)
            print(f"loaded {c}: {type(bundle['model']).__name__}, features {bundle['spec'].get('blocks')}, "
                  f"variant {bundle['spec'].get('variant', 'gray')}")
            return bundle
    sys.exit(f"No model found (tried {candidates}). Run the notebooks first.")


def class_scores(model, X):
    """Probabilities if the model has them, otherwise a softmax of the decision function."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[0]
        except AttributeError:
            pass
    if hasattr(model, "decision_function"):
        d = np.atleast_2d(model.decision_function(X))[0]
        e = np.exp(d - d.max())
        return e / e.sum()
    p = np.zeros(7)
    p[int(model.predict(X)[0])] = 1.0
    return p


def crop_face(gray, box, margin):
    """Same procedure as the KDEF preprocessing: expand the Haar box, crop, resize to 48x48."""
    x, y, w, h = box
    dx, dy = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - dx), max(0, y - dy)
    x2, y2 = min(gray.shape[1], x + w + dx), min(gray.shape[0], y + h + dy)
    return cv2.resize(gray[y1:y2, x1:x2], (F.IMG_SIZE, F.IMG_SIZE), interpolation=cv2.INTER_AREA)


def predict_face(bundle, face48, margin):
    # the landmark detector must know where the face is inside the crop (see extract_landmarks)
    X = F.extract_feature_set(face48[None], {**bundle["spec"], "crop_margin": margin}, n_jobs=1)
    return class_scores(bundle["model"], X)


def detect_faces(gray):
    faces = F._cascade().detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    return sorted([tuple(int(v) for v in f) for f in faces], key=lambda f: -f[2] * f[3])


def draw(frame, box, probs, labels):
    x, y, w, h = box
    k = int(np.argmax(probs))
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS[k], 2)
    cv2.putText(frame, f"{labels[k]} {probs[k]:.2f}", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[k], 2)
    # small bar chart of all classes next to the face
    for i, (lab, p) in enumerate(zip(labels, probs)):
        yy = y + 14 * i
        cv2.rectangle(frame, (x + w + 8, yy), (x + w + 8 + int(80 * p), yy + 10), COLORS[i], -1)
        cv2.putText(frame, lab[:3], (x + w + 92, yy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def run_image(bundle, path, margin):
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"cannot read {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if not faces:  # e.g. an already cropped face: use the whole image
        faces = [(0, 0, gray.shape[1], gray.shape[0])]
    for box in faces:
        probs = predict_face(bundle, crop_face(gray, box, margin), margin)
        print(f"face {box}: " + ", ".join(f"{l} {p:.2f}" for l, p in zip(bundle["labels"], probs)))
        draw(img, box, probs, bundle["labels"])
    out = ROOT / "demo_output.jpg"
    cv2.imwrite(str(out), img)
    print(f"annotated image written to {out}")


def run_camera(bundle, camera, margin, smooth):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {camera}")
    history = collections.deque(maxlen=max(1, smooth))
    smoothing, t_prev, fps = smooth > 1, time.time(), 0.0
    labels = bundle["labels"]
    print("q = quit, s = toggle smoothing, +/- = crop margin")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        for i, box in enumerate(faces[:3]):
            probs = predict_face(bundle, crop_face(gray, box, margin), margin)
            if i == 0:  # temporal smoothing (moving average of the scores) for the main face
                history.append(probs)
                if smoothing:
                    probs = np.mean(history, axis=0)
            draw(frame, box, probs, labels)
        if not faces:
            history.clear()
            cv2.putText(frame, "no face", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(frame, f"{fps:4.1f} FPS | margin {margin:.2f} | smoothing {'on' if smoothing else 'off'}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Group 2 - emotion recognition (q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            smoothing = not smoothing
        if key in (ord("+"), ord("=")):
            margin = min(0.5, margin + 0.05)
        if key == ord("-"):
            margin = max(0.0, margin - 0.05)
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="model bundle (.pkl); default: models/final.pkl, else best Task 1 model")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--margin", type=float, default=None,
                    help="crop margin around the Haar box (default 0.1)")
    ap.add_argument("--smooth", type=int, default=10, help="frames in the moving average (1 = off)")
    ap.add_argument("--image", help="run on a still image instead of the camera")
    args = ap.parse_args()
    bundle = load_bundle(args.model)
    margin = args.margin if args.margin is not None else 0.1
    if args.image:
        run_image(bundle, args.image, margin)
    else:
        run_camera(bundle, args.camera, margin, args.smooth)


if __name__ == "__main__":
    main()
