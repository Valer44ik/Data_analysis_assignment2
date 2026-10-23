"""Image preprocessing and feature extraction for Group 2 - Assignment 2 (FER2013 vs KDEF).

This is the only helper module of the project (the assignment allows the feature
extraction to live outside the notebooks). Everything related to models, training and
evaluation is implemented in the notebooks.

Conventions
-----------
* Images are uint8 arrays of shape (48, 48) (grayscale) or (48, 48, 3) (RGB), values 0-255.
* Labels follow the FER2013 order: 0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral.
* Landmarks are dlib's 68 points, returned in relative image coordinates in [0, 1].
"""
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.feature import hog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
FACE_CASCADE_PATH = PROJECT_ROOT / "faceAndLandmarkModels" / "haarcascade_frontalface_default.xml"
PREDICTOR_PATH = PROJECT_ROOT / "faceAndLandmarkModels" / "shape_predictor_68_face_landmarks_GTX.dat"
KDEF_ROOT = DATA_DIR / "KDEF_and_AKDEF" / "KDEF"
KDEF_SPLIT_PATH = PROJECT_ROOT / "auxFiles" / "kdef_subject_split.pkl"

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
KDEF_EMOTION_CODES = {"AF": "afraid", "AN": "angry", "DI": "disgust", "HA": "happy",
                      "NE": "neutral", "SA": "sad", "SU": "surprise"}
KDEF_POSE_CODES = {"FL": "full_left", "HL": "half_left", "S": "straight",
                   "HR": "half_right", "FR": "full_right"}
# KDEF emotion names -> FER2013 integer labels ("afraid" is FER's "Fear")
KDEF_TO_FER = {"angry": 0, "disgust": 1, "afraid": 2, "happy": 3, "sad": 4, "surprise": 5, "neutral": 6}

IMG_SIZE = 48
N_LANDMARKS = 68


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------
def pixels_to_images(pixel_strings):
    """FER2013 'pixels' column (space separated strings) -> (N, 48, 48) uint8."""
    return np.stack([np.asarray(s.split(), dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
                     for s in pixel_strings])


def load_fer_split(split):
    """Load one of the CSVs written by z_load_split_fer2013.ipynb ('train', 'validation', 'test')."""
    df = pd.read_csv(DATA_DIR / f"fer2013_{split}.csv")
    return pixels_to_images(df["pixels"]), df["emotion"].to_numpy(dtype=np.int64)


def parse_kdef(root=KDEF_ROOT):
    """Index all KDEF images (same parsing as z_load_kdef_student.ipynb) + FER label column."""
    records = []
    for path in sorted(Path(root).rglob("*.JPG")):
        stem = path.stem
        acquisition, subject, rest = stem[0], stem[1:4], stem[4:]
        code = next((c for c in KDEF_EMOTION_CODES if rest.startswith(c)), None)
        pose_code = rest[len(code):] if code else None
        if code is None or pose_code not in KDEF_POSE_CODES:
            continue  # two malformed file names in the release (e.g. AF31V, AM31H)
        emotion = KDEF_EMOTION_CODES[code]
        records.append({"path": str(path), "acquisition": acquisition, "subject": subject,
                        "emotion": emotion, "pose": KDEF_POSE_CODES[pose_code],
                        "label": KDEF_TO_FER[emotion]})
    return pd.DataFrame(records)


def load_kdef_frontal_split(root=KDEF_ROOT, split_path=KDEF_SPLIT_PATH):
    """Frontal ('straight') KDEF images split by subject with the official split file."""
    df = parse_kdef(root)
    with open(split_path, "rb") as f:
        split = pickle.load(f)
    frontal = df[df.pose == "straight"]
    train_df = frontal[frontal.subject.isin(split["train_subjects"])].reset_index(drop=True)
    test_df = frontal[frontal.subject.isin(split["test_subjects"])].reset_index(drop=True)
    assert not set(train_df.subject) & set(test_df.subject), "subject leakage between splits"
    return train_df, test_df


# ----------------------------------------------------------------------------------------
# dlib landmark predictor (lazily loaded once per process, also inside joblib workers)
# ----------------------------------------------------------------------------------------
_PREDICTOR = None
_CASCADE = None


def _predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        import dlib
        _PREDICTOR = dlib.shape_predictor(str(PREDICTOR_PATH))
    return _PREDICTOR


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    return _CASCADE


def detect_face(gray, scale_factor=1.1, min_neighbors=5, min_size=(50, 50)):
    """Largest Haar-cascade face (x, y, w, h) or None."""
    faces = _cascade().detectMultiScale(gray, scaleFactor=scale_factor,
                                        minNeighbors=min_neighbors, minSize=min_size)
    if len(faces) == 0:
        return None
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def landmarks_in_box(gray, box):
    """68 dlib landmarks (absolute pixel coordinates) inside a face box (x, y, w, h)."""
    import dlib
    x, y, w, h = box
    shape = _predictor()(gray, dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)


def eye_centers(pts):
    """Centers of the subject's right (points 36-41) and left (42-47) eye."""
    return pts[36:42].mean(axis=0), pts[42:48].mean(axis=0)


def rotate_to_horizontal_eyes(img, pts):
    """Rotate an image around the eye midpoint so the eye line becomes horizontal."""
    right, left = eye_centers(pts)
    angle = np.degrees(np.arctan2(left[1] - right[1], left[0] - right[0]))
    center = (float((right[0] + left[0]) / 2), float((right[1] + left[1]) / 2))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ----------------------------------------------------------------------------------------
# KDEF -> FER format
# ----------------------------------------------------------------------------------------
def preprocess_kdef_image(path, margin=0.20, size=IMG_SIZE, color="gray", align=False):
    """Extended version of the kit's preprocess_kdef_image (src/preprocess.py).

    1. Haar face detection (largest face)  2. optional eye-line alignment (dlib)
    3. expand the box by `margin` on every side  4. crop  5. gray (or keep RGB)
    6. resize to 48x48 with INTER_AREA.  Returns uint8 (not /255: normalization is done
    identically for FER and KDEF inside the feature extractors).  None if no face found.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Could not read {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    box = detect_face(gray)
    if box is None:
        return None
    if align:
        pts = landmarks_in_box(gray, box)
        gray = rotate_to_horizontal_eyes(gray, pts)
        bgr = rotate_to_horizontal_eyes(bgr, pts)
    x, y, w, h = box
    dx, dy = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - dx), max(0, y - dy)
    x2, y2 = min(gray.shape[1], x + w + dx), min(gray.shape[0], y + h + dy)
    if color == "rgb":
        face = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    else:
        face = gray[y1:y2, x1:x2]
    return cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_kdef_batch(paths, n_jobs=-1, **kwargs):
    """Parallel preprocess_kdef_image. Returns (images, ok_mask)."""
    out = Parallel(n_jobs=n_jobs)(delayed(preprocess_kdef_image)(p, **kwargs) for p in paths)
    ok = np.array([o is not None for o in out])
    shape = next(o.shape for o in out if o is not None)
    imgs = np.stack([o if o is not None else np.zeros(shape, np.uint8) for o in out])
    return imgs, ok


# ----------------------------------------------------------------------------------------
# Image-level variants (input: uint8 gray 48x48)
# ----------------------------------------------------------------------------------------
def to_clahe(img, clip_limit=2.0, tile=(4, 4)):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(img)


def to_binary_otsu(img):
    """Pure black/white face: global Otsu threshold after a light blur (no grey levels)."""
    _, out = cv2.threshold(cv2.GaussianBlur(img, (3, 3), 0), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def to_binary_adaptive(img, block_size=11, C=2):
    """Pure black/white face with a local (adaptive Gaussian) threshold: robust to shadows."""
    return cv2.adaptiveThreshold(cv2.GaussianBlur(img, (3, 3), 0), 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                 block_size, C)


def to_edges(img, method="sobel"):
    """Edge map: Sobel gradient magnitude (soft) or Canny (binary)."""
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    if method == "canny":
        return cv2.Canny(blur, 50, 150)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


VARIANTS = {
    "gray": lambda im: im,
    "clahe": to_clahe,
    "otsu": to_binary_otsu,
    "adaptive": to_binary_adaptive,
    "edges": to_edges,
}


def apply_variant(imgs, variant="gray"):
    """Apply one of VARIANTS to a stack of gray images. RGB stacks are only allowed for 'gray'."""
    if variant == "gray":
        return imgs
    fn = VARIANTS[variant]
    return np.stack([fn(im) for im in imgs])


def degrade_like_fer(img, rng):
    """Make a clean KDEF crop look like an in-the-wild FER image.

    Random contrast/brightness, Gaussian blur, sensor noise and JPEG compression.
    """
    out = img.astype(np.float32)
    out = out * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)
    sigma = rng.uniform(0.0, 1.0)
    if sigma > 0.2:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    out = out + rng.normal(0, rng.uniform(0, 6), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(20, 61))])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def degrade_batch(imgs, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([degrade_like_fer(im, rng) for im in imgs])


def zscore_images(imgs):
    """Per-image intensity standardization, rescaled back to uint8 (mean 128, std 40)."""
    f = imgs.astype(np.float32)
    axes = tuple(range(1, f.ndim))
    f = (f - f.mean(axis=axes, keepdims=True)) / (f.std(axis=axes, keepdims=True) + 1e-6)
    return np.clip(f * 40 + 128, 0, 255).astype(np.uint8)


def augment_images(imgs, labels, mode="flip", seed=0, max_angle=10):
    """Offline augmentation for the training set only.

    mode='flip': original + horizontal flip; 'flip_rot': + one random rotation in
    [-max_angle, max_angle] of every image. Returns (images, labels, origin_index).
    """
    rng = np.random.default_rng(seed)
    out, ys, origin = [imgs], [labels], [np.arange(len(imgs))]
    out.append(imgs[:, :, ::-1].copy()); ys.append(labels); origin.append(np.arange(len(imgs)))
    if mode == "flip_rot":
        rot = []
        for im in imgs:
            M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), rng.uniform(-max_angle, max_angle), 1.0)
            rot.append(cv2.warpAffine(im, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REPLICATE))
        out.append(np.stack(rot)); ys.append(labels); origin.append(np.arange(len(imgs)))
    return np.concatenate(out), np.concatenate(ys), np.concatenate(origin)


# ----------------------------------------------------------------------------------------
# HOG
# ----------------------------------------------------------------------------------------
def _hog_one(img, orientations, ppc, cpb, per_channel):
    if img.ndim == 3 and per_channel:
        return np.concatenate([hog(img[..., c], orientations=orientations, pixels_per_cell=ppc,
                                   cells_per_block=cpb, block_norm="L2-Hys") for c in range(img.shape[2])])
    return hog(img, orientations=orientations, pixels_per_cell=ppc, cells_per_block=cpb,
               block_norm="L2-Hys", channel_axis=-1 if img.ndim == 3 else None)


def _hog_chunk(chunk, orientations, ppc, cpb, per_channel):
    return np.stack([_hog_one(im, orientations, ppc, cpb, per_channel) for im in chunk]).astype(np.float32)


def extract_hog(imgs, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                per_channel=True, n_jobs=-1, chunk=2000):
    """HOG descriptors (reused from last year's extract_hog_features, chunked for speed).

    9 orientations, 8x8 cells, 2x2 blocks on 48x48 -> 900 dims. For RGB images
    per_channel=True concatenates the three channel descriptors (2700 dims).
    """
    chunks = [imgs[i:i + chunk] for i in range(0, len(imgs), chunk)]
    out = Parallel(n_jobs=n_jobs)(delayed(_hog_chunk)(c, orientations, tuple(pixels_per_cell),
                                                      tuple(cells_per_block), per_channel) for c in chunks)
    return np.concatenate(out)


# ----------------------------------------------------------------------------------------
# LBP texture histograms
# ----------------------------------------------------------------------------------------
def _lbp_chunk(chunk, P, R, grid):
    from skimage.feature import local_binary_pattern
    n_bins = P + 2                      # 'uniform' LBP codes: 0..P+1
    out = np.empty((len(chunk), grid * grid * n_bins), np.float32)
    for i, im in enumerate(chunk):
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        codes = local_binary_pattern(im, P, R, method="uniform")
        cells = []
        for rows in np.array_split(codes, grid, axis=0):
            for cell in np.array_split(rows, grid, axis=1):
                h = np.bincount(cell.astype(int).ravel(), minlength=n_bins)[:n_bins]
                cells.append(h / max(h.sum(), 1))
        out[i] = np.concatenate(cells)
    return out


def extract_lbp(imgs, P=8, R=1, grid=6, n_jobs=-1, chunk=2000):
    """Uniform LBP (Ojala et al. 2002) histograms on a grid x grid layout of cells.

    Classic texture descriptor for facial expressions (Shan et al. 2009): it captures wrinkles,
    furrows and teeth that geometry misses. 48x48 face, 6x6 cells of 8x8 px, 10 bins -> 360 dims.
    """
    chunks = [imgs[i:i + chunk] for i in range(0, len(imgs), chunk)]
    out = Parallel(n_jobs=n_jobs)(delayed(_lbp_chunk)(c, P, R, grid) for c in chunks)
    return np.concatenate(out)


def flip_images(imgs):
    """Horizontal mirror of a stack of images (test-time augmentation)."""
    return imgs[:, :, ::-1].copy()


# ----------------------------------------------------------------------------------------
# Landmarks (dlib 68 points)
# ----------------------------------------------------------------------------------------
def face_rect_in_crop(upscale, margin):
    """dlib rectangle of the detected face inside a crop made with `margin` (see
    preprocess_kdef_image): the face box is the central 1/(1+2*margin) part of the crop."""
    import dlib
    inset = int(round(upscale * margin / (1 + 2 * margin)))
    if inset == 0:  # tight crop (FER): whole image, as in the kit's baseline
        return dlib.rectangle(1, 1, upscale - 1, upscale - 1)
    return dlib.rectangle(inset, inset, upscale - 1 - inset, upscale - 1 - inset)


def _landmarks_chunk(chunk, upscale, margin):
    pred = _predictor()
    rect = face_rect_in_crop(upscale, margin)
    out = np.empty((len(chunk), N_LANDMARKS, 2), np.float32)
    for i, im in enumerate(chunk):
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        big = cv2.resize(im, (upscale, upscale), interpolation=cv2.INTER_CUBIC)
        shape = pred(big, rect)
        out[i] = [[p.x / upscale, p.y / upscale] for p in shape.parts()]
    return out


def extract_landmarks(imgs, upscale=96, margin=0.0, n_jobs=-1, chunk=2000):
    """dlib 68 landmarks for 48x48 face crops -> (N, 68, 2) in relative [0, 1] coordinates.

    The shape predictor needs the face rectangle. FER images are tight face crops, so (as in
    the kit's baseline) the whole image is used (margin=0). KDEF/webcam crops were expanded by
    `margin` around the Haar box, so the predictor gets that inner box instead; using the
    whole image there misplaces the landmarks (on KDEF: geometric-feature macro-F1 0.28 vs
    0.67). Images are upscaled first because the predictor works better on >48 px faces.
    """
    chunks = [imgs[i:i + chunk] for i in range(0, len(imgs), chunk)]
    out = Parallel(n_jobs=n_jobs)(delayed(_landmarks_chunk)(c, upscale, margin) for c in chunks)
    return np.concatenate(out)


def landmark_vector(pts):
    """Kit baseline normalization (extractLandmarksFER): center each face on its mean point
    and divide every axis by its max absolute value -> 136-dim translation/scale invariant."""
    p = pts - pts.mean(axis=1, keepdims=True)
    p = p / (np.abs(p).max(axis=1, keepdims=True) + 1e-8)
    return p.reshape(len(pts), -1).astype(np.float32)


def align_images_by_eyes(imgs, pts):
    """Eye-line alignment of 48x48 crops using their (relative) landmarks."""
    return np.stack([rotate_to_horizontal_eyes(im, p * IMG_SIZE) for im, p in zip(imgs, pts)])


# ----------------------------------------------------------------------------------------
# Geometric (human-understandable) features
# ----------------------------------------------------------------------------------------
def _d(pts, a, b):
    return np.linalg.norm(pts[:, a] - pts[:, b], axis=-1)


def _ear(pts, idx):
    """Eye aspect ratio (Soukupova & Cech 2016) for the 6 points of one eye."""
    p = [pts[:, i] for i in idx]
    return (np.linalg.norm(p[1] - p[5], axis=-1) + np.linalg.norm(p[2] - p[4], axis=-1)) / \
        (2 * np.linalg.norm(p[0] - p[3], axis=-1) + 1e-8)


GEO_NAMES = [
    "mouth_open", "mouth_open_outer", "mouth_width", "mouth_aspect", "smile_ratio",
    "eye_open", "brow_height", "inner_brow_height", "outer_brow_height", "brow_distance",
    "brow_slope", "nose_to_mouth", "jaw_drop", "upper_lip_thickness", "lower_lip_thickness",
    "nostril_width", "mouth_corner_to_eye", "face_height", "lip_corner_angle", "chin_to_mouth",
]


def geometric_features(pts):
    """~20 scale-invariant geometric descriptors from (N, 68, 2) landmarks.

    All distances are divided by the inter-ocular distance (iod). Image y grows downwards,
    so 'heights' are computed as (lower point y - upper point y) to be positive.
    Index reference (iBUG 68): jaw 0-16, brows 17-21 / 22-26, nose 27-35,
    eyes 36-41 / 42-47, outer lips 48-59, inner lips 60-67.
    """
    pts = np.asarray(pts, dtype=np.float32)
    right_eye, left_eye = pts[:, 36:42].mean(1), pts[:, 42:48].mean(1)
    iod = np.linalg.norm(left_eye - right_eye, axis=-1) + 1e-8
    eye_y = (right_eye[:, 1] + left_eye[:, 1]) / 2
    brow_y = (pts[:, 17:22, 1].mean(1) + pts[:, 22:27, 1].mean(1)) / 2
    corners_y = (pts[:, 48, 1] + pts[:, 54, 1]) / 2
    lip_center_y = (pts[:, 51, 1] + pts[:, 57, 1]) / 2
    mouth_open = _d(pts, 62, 66) / iod
    mouth_open_outer = _d(pts, 51, 57) / iod
    mouth_width = _d(pts, 48, 54) / iod
    # angle of the lip corners w.r.t. the upper-lip center: positive when corners are raised
    v_r, v_l = pts[:, 48] - pts[:, 51], pts[:, 54] - pts[:, 51]
    corner_angle = -(np.arctan2(v_r[:, 1], -v_r[:, 0]) + np.arctan2(v_l[:, 1], v_l[:, 0])) / 2
    feats = np.column_stack([
        mouth_open,
        mouth_open_outer,
        mouth_width,
        mouth_open_outer / (mouth_width + 1e-8),
        (lip_center_y - corners_y) / iod,                                   # smile_ratio
        (_ear(pts, range(36, 42)) + _ear(pts, range(42, 48))) / 2,          # eye_open
        (eye_y - brow_y) / iod,                                             # brow_height
        ((pts[:, 39, 1] - pts[:, 21, 1]) + (pts[:, 42, 1] - pts[:, 22, 1])) / 2 / iod,
        ((pts[:, 36, 1] - pts[:, 17, 1]) + (pts[:, 45, 1] - pts[:, 26, 1])) / 2 / iod,
        _d(pts, 21, 22) / iod,                                              # brow_distance (frown)
        ((pts[:, 17, 1] - pts[:, 21, 1]) + (pts[:, 26, 1] - pts[:, 22, 1])) / 2 / iod,  # brow_slope
        _d(pts, 33, 51) / iod,                                              # nose_to_mouth (disgust)
        _d(pts, 27, 8) / iod,                                               # jaw_drop
        _d(pts, 51, 62) / iod,
        _d(pts, 66, 57) / iod,
        _d(pts, 31, 35) / iod,
        (_d(pts, 48, 36) + _d(pts, 54, 45)) / 2 / iod,
        _d(pts, 27, 8) / iod / (mouth_width + 1e-8),                        # face_height (relative)
        corner_angle,
        _d(pts, 57, 8) / iod,
    ]).astype(np.float32)
    return feats


# ----------------------------------------------------------------------------------------
# Emotion prototypes (Procrustes) learned on KDEF
# ----------------------------------------------------------------------------------------
def procrustes_normalize(pts):
    """Remove translation and scale (unit Frobenius norm) of each (68, 2) shape."""
    p = pts - pts.mean(axis=1, keepdims=True)
    return p / (np.linalg.norm(p, axis=(1, 2), keepdims=True) + 1e-8)


def align_shapes(pts, ref):
    """Similarity-align every normalized shape to `ref` (orthogonal Procrustes, rotation only)."""
    p = procrustes_normalize(pts)
    out = np.empty_like(p)
    for i, s in enumerate(p):
        u, _, vt = np.linalg.svd(s.T @ ref)
        out[i] = s @ (u @ vt)
    return out


def build_prototypes(pts, labels, n_classes=7, n_iter=5):
    """Generalized Procrustes mean shape per emotion. Returns (mean_shape, prototypes[n_classes])."""
    ref = procrustes_normalize(pts).mean(0)
    for _ in range(n_iter):
        aligned = align_shapes(pts, ref)
        ref = aligned.mean(0)
        ref /= np.linalg.norm(ref)
    aligned = align_shapes(pts, ref)
    protos = np.stack([aligned[labels == c].mean(0) for c in range(n_classes)])
    return ref, protos


def prototype_distances(pts, mean_shape, prototypes):
    """Procrustes distance of every face to each emotion prototype -> (N, n_classes)."""
    aligned = align_shapes(pts, mean_shape)
    return np.stack([np.linalg.norm(aligned - p, axis=(1, 2)) for p in prototypes], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------------------
# Near-duplicate hashing (data curation)
# ----------------------------------------------------------------------------------------
def dhash(imgs, hash_size=8):
    """Difference hash per image as a uint64 (same idea as imagehash.dhash, numpy only)."""
    small = np.stack([cv2.resize(im, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
                      for im in imgs]).astype(np.int16)
    bits = (small[:, :, 1:] > small[:, :, :-1]).reshape(len(imgs), -1)
    return np.packbits(bits, axis=1).view(">u8").ravel().astype(np.uint64)


# ----------------------------------------------------------------------------------------
# Feature sets used by the notebooks and by demo_emotion.py
# ----------------------------------------------------------------------------------------
def extract_feature_set(imgs, spec, pts=None, n_jobs=-1):
    """Build the feature matrix described by `spec` (a dict saved inside every model bundle).

    spec keys:
      variant  : image variant applied first ('gray', 'clahe', 'otsu', 'adaptive', 'edges')
      blocks   : list of blocks to concatenate, from
                 'hog' | 'lm' | 'pix' | 'lbp' | 'geo' | 'proto'
      hog      : dict(orientations, pixels_per_cell, cells_per_block) (optional)
      geo_idx  : indices into GEO_NAMES to keep (optional, default all)
      proto    : dict(mean_shape, prototypes) for the 'proto' block
      crop_margin : margin of the face crop (0 for FER images), tells dlib where the face is
    `pts` can be passed to reuse precomputed landmarks.
    """
    imgs = apply_variant(imgs, spec.get("variant", "gray"))
    blocks = spec["blocks"]
    need_pts = any(b in blocks for b in ("lm", "geo", "proto"))
    if need_pts and pts is None:
        pts = extract_landmarks(imgs, margin=spec.get("crop_margin", 0.0), n_jobs=n_jobs)
    parts = []
    for b in blocks:
        if b == "hog":
            parts.append(extract_hog(imgs, n_jobs=n_jobs, **spec.get("hog", {})))
        elif b == "lm":
            parts.append(landmark_vector(pts))
        elif b == "pix":
            parts.append(imgs.reshape(len(imgs), -1).astype(np.float32) / 255.0)
        elif b == "lbp":
            parts.append(extract_lbp(imgs, n_jobs=n_jobs))
        elif b == "geo":
            g = geometric_features(pts)
            parts.append(g[:, spec["geo_idx"]] if spec.get("geo_idx") is not None else g)
        elif b == "proto":
            parts.append(prototype_distances(pts, spec["proto"]["mean_shape"], spec["proto"]["prototypes"]))
        else:
            raise ValueError(f"unknown block {b}")
    return np.hstack(parts).astype(np.float32)


# ----------------------------------------------------------------------------------------
# Small caching helper
# ----------------------------------------------------------------------------------------
def cached(name, fn, force=False):
    """Compute fn() once and store the result in data/cache/<name>.npy (or .npz for dicts)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npy, npz = CACHE_DIR / f"{name}.npy", CACHE_DIR / f"{name}.npz"
    if not force and npy.exists():
        return np.load(npy)
    if not force and npz.exists():
        return dict(np.load(npz, allow_pickle=True))
    out = fn()
    if isinstance(out, dict):
        np.savez_compressed(npz, **out)
    else:
        np.save(npy, out)
    return out
