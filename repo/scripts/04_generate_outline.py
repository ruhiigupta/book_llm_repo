import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTLINE_PATH = os.path.join(OUTPUT_DIR, "book_outline.json")
RAW_PATH = os.path.join(OUTPUT_DIR, "outline_raw.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


def compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clip_text(text: str, limit: int) -> str:
    text = compact_spaces(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def get_representative_text(chunks, preview_chars=120):
    if not chunks or preview_chars <= 0:
        return ""

    texts = [c.get("text", "") if isinstance(c, dict) else c for c in chunks]
    texts = [compact_spaces(t) for t in texts if t and compact_spaces(t)]

    if not texts:
        return ""

    selected = [texts[0]]
    if len(texts) > 2:
        selected.append(texts[len(texts) // 2])
    if len(texts) > 1:
        selected.append(texts[-1])

    joined = " ".join(selected)
    return clip_text(joined, preview_chars)


def load_video_summaries(preview_chars=120):
    if not os.path.exists(CLEAN_DIR):
        raise FileNotFoundError(f"Clean directory not found: {CLEAN_DIR}")

    files = sorted(
        [f for f in os.listdir(CLEAN_DIR) if f.startswith("video_") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )

    summaries = []
    for filename in files:
        path = os.path.join(CLEAN_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = data.get("chunks", [])
        preview = get_representative_text(chunks, preview_chars=preview_chars)
        title = compact_spaces(data.get("title", f"Video {data.get('video_index', '?')}"))
        video_index = data.get("video_index")

        if preview:
            line = f"Video {video_index}: {title} | Preview: {preview}"
        else:
            line = f"Video {video_index}: {title}"

        summaries.append(line)

    print(f"Loaded {len(summaries)} video summaries")
    return summaries


def build_prompt(combined: str) -> str:
    return f"""
Design a professional textbook outline for a course titled "Building LLMs from Scratch".

Rules:
- Follow the original video order exactly
- Group only adjacent videos into chapters
- Each video must appear exactly once
- No duplicates
- No skipped videos
- Aim for 6 to 8 chapters
- Each chapter must cover one clear concept
- Chapter titles must be specific and non-generic
- Each chapter must contain 3 to 6 sections
- Section titles must be specific, meaningful, and non-overlapping
- Do not invent topics not present in the videos
- Prefer clarity over compression

Return only valid JSON with this shape:

{{
  "book_title": "Building Large Language Models from Scratch",
  "chapters": [
    {{
      "chapter_number": 1,
      "chapter_title": "Specific Title",
      "video_indices": [1, 2, 3],
      "sections": [
        "Specific Section 1",
        "Specific Section 2",
        "Specific Section 3"
      ]
    }}
  ]
}}

Video summaries:
{combined}
""".strip()


def extract_json_block(raw: str) -> dict:
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1

    if start == -1 or end <= 0:
        raise ValueError("LLM did not return JSON")

    raw = raw[start:end]
    return json.loads(raw)


def call_outline_model(prompt: str):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2500,
    )
    return response.choices[0].message.content.strip()


def generate_outline(preview_chars=120):
    summaries = load_video_summaries(preview_chars=preview_chars)
    combined = "\n".join(summaries)
    prompt = build_prompt(combined)

    raw = call_outline_model(prompt)

    with open(RAW_PATH, "w", encoding="utf-8") as f:
        f.write(raw)

    outline = extract_json_block(raw)
    return outline


def normalize_outline(outline):
    chapters = outline.get("chapters", [])
    normalized = []
    for i, ch in enumerate(chapters, start=1):
        video_indices = ch.get("video_indices", [])
        video_indices = sorted({int(v) for v in video_indices if isinstance(v, int) or str(v).isdigit()})
        sections = ch.get("sections", [])
        sections = [str(s).strip() for s in sections if str(s).strip()]

        normalized.append(
            {
                "chapter_number": i,
                "chapter_title": str(ch.get("chapter_title", f"Chapter {i}")).strip(),
                "video_indices": video_indices,
                "sections": sections,
            }
        )

    outline["book_title"] = str(outline.get("book_title", "Building Large Language Models from Scratch")).strip()
    outline["chapters"] = normalized
    return outline


def auto_repair_outline(outline, total_videos):
    chapters = outline.get("chapters", [])
    if not chapters:
        return outline

    all_mapped = []
    for ch in chapters:
        all_mapped.extend(ch.get("video_indices", []))

    mapped_set = set(all_mapped)
    all_videos = set(range(1, total_videos + 1))
    unmapped = sorted(all_videos - mapped_set)

    if not unmapped:
        return outline

    for vid in unmapped:
        best_chapter = None
        best_distance = float("inf")

        for ch in chapters:
            indices = ch.get("video_indices", [])
            if not indices:
                continue
            distance = min(abs(vid - i) for i in indices)
            if distance < best_distance:
                best_distance = distance
                best_chapter = ch

        if best_chapter is not None:
            best_chapter["video_indices"].append(vid)
            best_chapter["video_indices"] = sorted(set(best_chapter["video_indices"]))

    return outline


def validate_outline(outline, total_videos):
    chapters = outline.get("chapters", [])
    all_mapped = []

    for ch in chapters:
        all_mapped.extend(ch.get("video_indices", []))

    mapped_set = set(all_mapped)
    all_videos = set(range(1, total_videos + 1))
    unmapped = sorted(all_videos - mapped_set)

    duplicates = []
    seen = set()
    for v in all_mapped:
        if v in seen:
            duplicates.append(v)
        seen.add(v)

    print("\nOutline Validation:")
    print(f"  Total chapters: {len(chapters)}")
    print(f"  Videos mapped: {len(mapped_set)}/{total_videos}")

    if unmapped:
        print(f"  Unmapped videos: {unmapped}")
    if duplicates:
        print(f"  Duplicates: {sorted(set(duplicates))}")

    print("\nChapter breakdown:")
    for ch in chapters:
        print(f"  Ch{ch['chapter_number']}: {ch['chapter_title']}")
        print(f"    Videos: {ch['video_indices']}")
        for s in ch.get("sections", []):
            print(f"    - {s}")

    return len(unmapped) == 0 and len(duplicates) == 0


def save_outline(outline):
    out_path = OUTLINE_PATH
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outline, f, indent=2)
    print(f"\nOutline saved → {out_path}")
    print(f"Book title: {outline.get('book_title', '')}")
    print(f"Total chapters: {len(outline.get('chapters', []))}")


if __name__ == "__main__":
    preview_sizes = [120, 80, 40, 0]
    outline = None
    last_error = None

    for preview_chars in preview_sizes:
        try:
            print(f"\nTrying outline generation with preview size: {preview_chars}")
            outline = generate_outline(preview_chars=preview_chars)
            break
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            print(f"Attempt failed: {e}")
            if "413" in msg or "request entity too large" in msg:
                continue
            if "json" in msg or "parse" in msg or "llm did not return json" in msg:
                continue
            continue

    if outline is None:
        raise RuntimeError(f"Outline generation failed: {last_error}")

    outline = normalize_outline(outline)
    outline = auto_repair_outline(outline, total_videos=len(load_video_summaries(preview_chars=0)))
    is_valid = validate_outline(outline, total_videos=len(load_video_summaries(preview_chars=0)))

    save_outline(outline)

    if is_valid:
        print("\nOutline VALID — ready for Step 5")
    else:
        print("\nFix issues before continuing")