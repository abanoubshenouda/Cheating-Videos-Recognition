import os
import time
import requests
from pathlib import Path

# ============================================================
API_KEY = "ii2SyMi1SEJSTUhhz6QkfraFUDBFSizNs8tnYPyEeM0w2zUQQXprzdSR"   # ← حط مفتاحك هنا
# ============================================================

CLASSES = {
    "phone_use": [
        "student using phone exam",
        "person hiding phone under desk",
        "student texting classroom secretly",
    ],
    "looking_sideways": [
        "student looking sideways classroom",
        "person glancing neighbor desk",
        "student turning head exam",
    ],
    "whispering": [
        "students whispering classroom",
        "person whispering secretly",
    ],
    "passing_paper": [
        "student passing note paper",
        "person handing paper secretly",
    ],
    "hiding_notes": [
        "student hiding cheat sheet",
        "person reading hidden notes",
    ],
    "normal_behavior": [
        "student writing exam paper",
        "student studying desk focused",
        "person sitting exam concentration",
    ],
}

VIDEOS_PER_QUERY = 40
OUTPUT_DIR = Path("dataset")
HEADERS = {"Authorization": API_KEY}


def search_videos(query, per_page):
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers=HEADERS,
            params={"query": query, "per_page": per_page, "orientation": "landscape"},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("videos", [])
    except Exception as e:
        print(f"    [WARN] Search failed: {e}")
        return []


def get_download_url(video):
    files = video.get("video_files", [])
    files_sorted = sorted(
        files,
        key=lambda f: {"hd": 0, "sd": 1}.get(f.get("quality", ""), 2),
    )
    for f in files_sorted:
        if f.get("file_type", "").startswith("video/mp4"):
            return f.get("link")
    return None


def download_video(url, filepath):
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    [ERR] Download failed: {e}")
        return False


def main():
    if API_KEY == "YOUR_PEXELS_API_KEY":
        print("حط الـ API key الأول!")
        print("اعمل account مجاني على: https://www.pexels.com/api/")
        return

    # إنشاء الفولدرات
    for cls in CLASSES:
        (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    print("dataset/")
    for cls in CLASSES:
        print(f"  └── {cls}/")
    print()

    total = 0

    for cls, queries in CLASSES.items():
        print(f"\n── {cls} ──")
        downloaded_ids = set()

        for query in queries:
            print(f'  Query: "{query}"')
            videos = search_videos(query, VIDEOS_PER_QUERY)

            for video in videos:
                vid_id = video["id"]

                if vid_id in downloaded_ids:
                    continue

                url = get_download_url(video)
                if not url:
                    continue

                filepath = OUTPUT_DIR / cls / f"{vid_id}.mp4"

                if filepath.exists():
                    print(f"    [SKIP] {vid_id}.mp4")
                    downloaded_ids.add(vid_id)
                    continue

                print(f"    [DL]   {vid_id}.mp4 ...", end=" ", flush=True)
                ok = download_video(url, filepath)
                if ok:
                    size_mb = filepath.stat().st_size / 1_000_000
                    print(f"{size_mb:.1f} MB")
                    downloaded_ids.add(vid_id)
                    total += 1

                time.sleep(0.3)

    print(f"\n{'='*40}")
    print(f"Done! {total} videos downloaded")
    print(f"Folder: {OUTPUT_DIR.resolve()}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
