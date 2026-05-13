"""
Pexels Video Scraper — Cheating Detection Dataset
Per-class folder structure | Multi-query per class
===================================================
كل action له فولدر لوحده تحت train / val / test
اسم الفولدر = label تلقائي للـ PyTorch ImageFolder
"""

import os, time, csv, shutil, random
import requests
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIG — عدّل هنا بس
# ============================================================
API_KEY = "YOUR_PEXELS_API_KEY"   # ← من pexels.com/api

# كل class وأفضل queries للبحث عنه
CLASSES = {
    "phone_use": [
        "student using phone exam",
        "person hiding phone under desk",
        "student texting classroom secretly",
    ],
    "looking_sideways": [
        "student looking sideways classroom",
        "person glancing neighbor desk",
        "student turning head exam hall",
    ],
    "whispering": [
        "students whispering classroom",
        "person whispering secretly",
        "students talking quietly desk",
    ],
    "passing_paper": [
        "student passing note paper",
        "person handing paper secretly",
        "student sharing paper exam",
    ],
    "hiding_notes": [
        "student hiding cheat sheet",
        "person reading hidden notes",
        "student concealing paper desk",
    ],
    "normal_behavior": [
        "student writing exam paper",
        "person sitting exam concentration",
        "student reading exam questions",
        "student studying desk focused",
    ],
}

# ── كميات ──
VIDEOS_PER_QUERY  = 5     # فيديوهات لكل query (max 80 للـ free plan)
FRAME_INTERVAL_S  = 1.0   # استخرج frame كل ثانية
FRAME_SIZE        = (224, 224)

# ── نسب التقسيم ──
TRAIN_R, VAL_R, TEST_R = 0.70, 0.15, 0.15

# ── مجلدات ──
BASE       = Path("cheating_dataset")
RAW_DIR    = BASE / "raw_videos"
FRAMES_DIR = BASE / "frames"
SPLIT_DIR  = BASE / "split"
META_PATH  = BASE / "metadata.csv"

random.seed(42)
HEADERS = {"Authorization": API_KEY}

# ============================================================
# SETUP
# ============================================================
def setup():
    for cls in CLASSES:
        (RAW_DIR  / cls).mkdir(parents=True, exist_ok=True)
        (FRAMES_DIR / cls).mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (SPLIT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    print("[OK] Folder structure created\n")
    print("cheating_dataset/")
    print("├── raw_videos/")
    for cls in CLASSES:
        print(f"│   ├── {cls}/")
    print("├── frames/")
    for cls in CLASSES:
        print(f"│   ├── {cls}/")
    print("└── split/")
    print("    ├── train/  ├── val/  └── test/")
    for cls in CLASSES:
        print(f"        └── {cls}/")
    print()

# ============================================================
# PEXELS API
# ============================================================
def search_videos(query: str, per_page: int) -> list:
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers=HEADERS,
            params={"query": query, "per_page": per_page, "orientation": "landscape"},
            timeout=15
        )
        r.raise_for_status()
        return r.json().get("videos", [])
    except Exception as e:
        print(f"    [WARN] Search failed '{query}': {e}")
        return []

def best_url(video: dict) -> str | None:
    files = sorted(
        video.get("video_files", []),
        key=lambda f: {"hd": 0, "sd": 1}.get(f.get("quality", ""), 2)
    )
    for f in files:
        if f.get("file_type", "").startswith("video/mp4"):
            return f.get("link")
    return None

def download_video(video: dict, cls: str, query: str) -> dict | None:
    vid_id = video["id"]
    url    = best_url(video)
    if not url:
        return None

    fname    = f"{cls}_{vid_id}.mp4"
    filepath = RAW_DIR / cls / fname

    if not filepath.exists():
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            print(f"    [DL] {fname}")
        except Exception as e:
            print(f"    [ERR] {vid_id}: {e}")
            return None
    else:
        print(f"    [SKIP] {fname} exists")

    return {
        "video_id": vid_id, "class": cls, "query": query,
        "filename": fname, "duration": video.get("duration", 0),
        "scraped_at": datetime.utcnow().isoformat(),
    }

# ============================================================
# FRAME EXTRACTION
# ============================================================
def extract_frames(meta: dict) -> int:
    try:
        import cv2
    except ImportError:
        print("    [ERR] opencv-python not installed — run: pip install opencv-python")
        return 0

    cls      = meta["class"]
    filepath = RAW_DIR / cls / meta["filename"]
    vid_id   = meta["video_id"]

    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return 0

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    step       = max(1, int(fps * FRAME_INTERVAL_S))
    saved      = 0
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            resized  = cv2.resize(frame, FRAME_SIZE)
            out_path = FRAMES_DIR / cls / f"{vid_id}_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
        frame_idx += 1

    cap.release()
    return saved

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================
def split_dataset():
    print("\n[SPLIT] Distributing frames to train / val / test ...")
    summary = {}
    for cls in CLASSES:
        frames = list((FRAMES_DIR / cls).glob("*.jpg"))
        random.shuffle(frames)
        n      = len(frames)
        n_tr   = int(n * TRAIN_R)
        n_val  = int(n * VAL_R)

        splits = {
            "train": frames[:n_tr],
            "val":   frames[n_tr:n_tr + n_val],
            "test":  frames[n_tr + n_val:],
        }
        counts = {}
        for split, flist in splits.items():
            for src in flist:
                dst = SPLIT_DIR / split / cls / src.name
                if dst.exists():
                    dst = dst.with_stem(dst.stem + "_dup")
                shutil.copy2(src, dst)
            counts[split] = len(flist)
        summary[cls] = counts
        print(f"  {cls:20s}: train={counts['train']:4d}  val={counts['val']:4d}  test={counts['test']:4d}")
    return summary

# ============================================================
# METADATA CSV
# ============================================================
def save_meta(records: list):
    if not records:
        return
    with open(META_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()) + ["frames_extracted"])
        w.writeheader()
        w.writerows(records)
    print(f"\n[SAVED] metadata → {META_PATH}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  Cheating Detection — Pexels Scraper v2")
    print("  Per-class folder structure")
    print("=" * 60 + "\n")

    if API_KEY == "YOUR_PEXELS_API_KEY":
        print("[STOP] حط الـ API key الأول!")
        print("       اعمل account مجاني على https://www.pexels.com/api/")
        return

    setup()

    all_meta = []

    for cls, queries in CLASSES.items():
        print(f"\n── Class: {cls} ──")
        for query in queries:
            print(f"  Query: \"{query}\"")
            videos = search_videos(query, VIDEOS_PER_QUERY)
            for video in videos:
                meta = download_video(video, cls, query)
                if meta:
                    n_frames = extract_frames(meta)
                    meta["frames_extracted"] = n_frames
                    all_meta.append(meta)
                time.sleep(0.4)   # rate limit

    save_meta(all_meta)
    split_dataset()

    # ── Final summary ──
    total_videos = len(all_meta)
    total_frames = sum(m.get("frames_extracted", 0) for m in all_meta)
    print("\n" + "=" * 60)
    print(f"  DONE!")
    print(f"  Videos downloaded : {total_videos}")
    print(f"  Frames extracted  : {total_frames}")
    print(f"  Dataset ready in  : {SPLIT_DIR.resolve()}")
    print("=" * 60)
    print("\nNext step:")
    print("  from torchvision import datasets")
    print("  train_data = datasets.ImageFolder('cheating_dataset/split/train')")
    print("  # classes =", list(CLASSES.keys()))

if __name__ == "__main__":
    main()
