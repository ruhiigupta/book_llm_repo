import os
import json
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTLINE_PATH = os.path.join(BASE_DIR, "output", "book_outline.json")

TOTAL_VIDEOS = 43


SPLIT_CHAPTERS = [
    {
        "chapter_number": 8,
        "chapter_title": "Text Generation and Model Evaluation",
        "video_indices": [26, 27, 28, 29, 30],
        "sections": [
            "Generating Text with LLMs",
            "Measuring LLM Loss Function",
            "Temperature and Top-K Sampling",
            "Evaluating LLM Performance"
        ]
    },
    {
        "chapter_number": 9,
        "chapter_title": "Loading Pretrained Weights and Classification Fine-Tuning",
        "video_indices": [31, 32, 33, 34, 35, 36, 37],
        "sections": [
            "Saving and Loading Model Weights",
            "Loading OpenAI Pretrained Weights",
            "Fine-Tuning for Text Classification",
            "Building a Spam Classifier",
            "Evaluating the Classification Model"
        ]
    },
    {
        "chapter_number": 10,
        "chapter_title": "Instruction Fine-Tuning and Advanced Directions",
        "video_indices": [38, 39, 40, 41, 42, 43],
        "sections": [
            "Introduction to Instruction Fine-Tuning",
            "Data Batching and Dataloaders",
            "Instruction Fine-Tuning Training Loop",
            "Evaluating the Fine-Tuned Model",
            "Future Directions and Next Steps"
        ]
    }
]


SPLIT_VIDEO_INDICES = set()
for ch in SPLIT_CHAPTERS:
    SPLIT_VIDEO_INDICES.update(ch["video_indices"])


def auto_repair_unmapped(chapters):
    """
    Finds any unmapped videos and assigns them
    to the nearest chapter automatically.
    Never fails silently.
    """
    all_mapped = []
    for ch in chapters:
        all_mapped.extend(ch["video_indices"])

    mapped_set = set(all_mapped)
    all_videos = set(range(1, TOTAL_VIDEOS + 1))
    unmapped = sorted(all_videos - mapped_set)

    if not unmapped:
        return chapters

    print(f"\n Auto-repairing {len(unmapped)} unmapped videos: {unmapped}")

    for vid in unmapped:
        best = min(
            chapters,
            key=lambda c: min(abs(vid - i) for i in c["video_indices"])
        )
        best["video_indices"].append(vid)
        best["video_indices"].sort()
        print(f"    Video {vid} → Ch{best['chapter_number']}: {best['chapter_title']}")

    return chapters


def fix_outline():

    backup_path = OUTLINE_PATH.replace(".json", "_backup.json")
    shutil.copy(OUTLINE_PATH, backup_path)
    print(f" Backup created → {backup_path}")

    with open(OUTLINE_PATH, "r", encoding="utf-8") as f:
        outline = json.load(f)

    chapters = outline["chapters"]


    new_chapters = []
    for ch in chapters:
        overlap = set(ch["video_indices"]) & SPLIT_VIDEO_INDICES
        if overlap:
            print(f"     Removing Ch{ch['chapter_number']} "
                f"(overlaps with split: {sorted(overlap)})")
        else:
            new_chapters.append(ch)


    new_chapters.extend(SPLIT_CHAPTERS)


    new_chapters.sort(key=lambda x: x["chapter_number"])

    
    for i, ch in enumerate(new_chapters):
        ch["chapter_number"] = i + 1

    
    new_chapters = auto_repair_unmapped(new_chapters)

    outline["chapters"] = new_chapters

    
    all_mapped = []
    for ch in new_chapters:
        all_mapped.extend(ch["video_indices"])

    mapped_set = set(all_mapped)
    all_videos = set(range(1, TOTAL_VIDEOS + 1))
    unmapped = all_videos - mapped_set
    duplicates = [v for v in all_mapped if all_mapped.count(v) > 1]

    print(f"\n Validation:")
    print(f"   Total chapters: {len(new_chapters)}")
    print(f"   Videos mapped: {len(mapped_set)}/{TOTAL_VIDEOS}")
    print(f"   Chapter numbers: {[ch['chapter_number'] for ch in new_chapters]}")


    if unmapped:
        print(f"    STILL unmapped: {sorted(unmapped)}")
        raise ValueError(f"Auto-repair failed for videos: {sorted(unmapped)}")
    if duplicates:
        print(f"    Duplicates found: {list(set(duplicates))}")
        raise ValueError(f"Duplicate videos detected: {list(set(duplicates))}")

    print(f"    All {TOTAL_VIDEOS} videos mapped correctly")
    print(f"    No duplicates")
    print(f"   Sequential numbering: 1-{len(new_chapters)}")


    print(f"\n Final chapter structure:")
    for ch in new_chapters:
        print(f"   Ch{ch['chapter_number']}: {ch['chapter_title']}")
        print(f"      Videos: {ch['video_indices']}")


    with open(OUTLINE_PATH, "w", encoding="utf-8") as f:
        json.dump(outline, f, indent=2)

    print(f"\n Outline updated → output/book_outline.json")
    print(f" Next: python 06_generate_chapters.py")


if __name__ == "__main__":
    fix_outline()