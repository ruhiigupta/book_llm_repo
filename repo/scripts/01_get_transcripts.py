import os
import json
import subprocess
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu"


def get_video_ids(playlist_url):
    print("Fetching playlist using yt-dlp...")
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--yes-playlist", "-J", playlist_url],
        capture_output=True,
        text=True
    )


    print("Return code:", result.returncode)
    if result.stderr:
        print("STDERR:", result.stderr[:500]) 
    if not result.stdout.strip():
        print(" yt-dlp returned empty output")
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f" JSON parse failed: {e}")
        print("Raw output preview:", result.stdout[:300])
        return []

    if data is None:
        print(" Parsed JSON is None")
        return []

    entries = data.get("entries", [])
    print(f"Raw entries found: {len(entries)}")

    return [
        (entry["id"], entry.get("title", f"video_{i+1}"))
        for i, entry in enumerate(entries)
    ]


def save_transcripts(video_entries):
    os.makedirs(RAW_DIR, exist_ok=True)
    success = 0
    failed = []

    for i, (vid, title) in enumerate(tqdm(video_entries, desc="Downloading transcripts")):
        try:
            transcript = YouTubeTranscriptApi().fetch(vid)
            text = " ".join([t.text for t in transcript])

            payload = {
                "video_index": i + 1,
                "video_id": vid,
                "title": title,
                "url": f"https://youtube.com/watch?v={vid}",
                "transcript": text,
                "chunks": [{"text": t.text, "start": t.start, "duration": t.duration} for t in transcript]
            }

            filepath = os.path.join(RAW_DIR, f"video_{i+1:02d}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            success += 1

        except Exception as e:
            print(f"\n Skipped [{title}] ({vid}): {e}")
            failed.append({
                "video_id": vid,
                "title": title,
                "error": str(e)
            })


    log_path = os.path.join(RAW_DIR, "run_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "total_videos": len(video_entries),
            "success": success,
            "failed": failed
        }, f, indent=2)

    print(f"\n Done: {success}/{len(video_entries)} transcripts saved")
    if failed:
        print(f"  {len(failed)} videos skipped — check run_log.json")


if __name__ == "__main__":
    video_entries = get_video_ids(PLAYLIST_URL)
    print(f"Found {len(video_entries)} videos")
    save_transcripts(video_entries)