"""
06_retry_failed_sections.py
────────────────────────────
Reads retry_queue.txt, regenerates each failed section with a
longer inter-call delay, and patches the placeholder back into
the correct chapter .md file.

Run this AFTER 05_generate_chapters.py completes.
Run it as many times as needed — processed entries are removed
from the queue automatically.
"""

import os
import re
import time
import chromadb
from groq import Groq
from config import get_groq_client

groq_client = get_groq_client()
from sentence_transformers import SentenceTransformer


BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")
BOOK_DIR     = os.path.join(OUTPUT_DIR, "book")
DB_DIR       = os.path.join(BASE_DIR, "data", "vectordb")
OUTLINE_PATH = os.path.join(OUTPUT_DIR, "book_outline.json")
RETRY_LOG    = os.path.join(OUTPUT_DIR, "retry_queue.txt")




print(" Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(" Embedding model loaded")


def load_vector_store():
    chroma = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma.get_collection("transcripts")
    print(f" Vector store loaded — {collection.count()} vectors")
    return collection


def build_query(chapter_title: str, section_title: str) -> str:
    expansion = f"{section_title}; definition, intuition, mechanism, example"
    base = (
        f"{section_title}. Key ideas, definitions, mechanisms, and an example. "
        f"Context: {chapter_title}."
    )
    return base + " " + expansion


def retrieve_chunks(collection, query: str, n: int = 10, min_score: float = 0.25) -> list[str]:
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )
    scored = []
    seen = set()
    for doc, _, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        doc = doc.strip()
        if not doc or doc in seen:
            continue
        seen.add(doc)
        score = round(1 - dist, 3)
        scored.append((score, doc))

    if not scored:
        return []

    scored.sort(reverse=True, key=lambda item: item[0])
    chunks = [doc for score, doc in scored if score >= min_score]
    if not chunks:
        chunks = [doc for _, doc in scored[:n]]
    return chunks


def format_context(chunks: list[str], max_chars: int = 2000) -> str:
    buf, total = [], 0
    for chunk in chunks:
        if total + len(chunk) > max_chars:
            break
        buf.append(chunk)
        total += len(chunk)
    return "\n\n".join(buf)


def clean_text(text: str) -> str:
    SKIP = {"source:", "score:", "instruction:", "context:"}
    return "\n".join(
        line.strip() for line in text.split("\n")
        if not any(kw in line.lower() for kw in SKIP)
    ).strip()

def call_groq(prompt: str, retries: int = 5, base_delay: int = 20) -> str:
    """
    More conservative than the main script:
    - 5 retries instead of 4
    - base_delay=20s instead of 10s
    - Pre-call sleep of 15s to avoid hitting limits immediately
    """
    time.sleep(15)   
    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = min(120, base_delay * (2 ** attempt))
            print(f"    Rate limit — waiting {wait}s (attempt {attempt + 1}/{retries})...")
            time.sleep(wait)
    raise RuntimeError("Groq failed after all retries")


def is_valid(text: str) -> bool:
    t = text.lower()
    return (
        len(text.split()) >= 90 and
        not any(x in t for x in ["source:", "score:", "context:", "instructions:"]) and
        text.count(".") >= 3
    )


def build_prompt(chapter_title: str, section_title: str, context: str) -> str:
    return f"""You are writing a professional, high-quality textbook.

Write a clear, self-contained section for the following:

Chapter : {chapter_title}
Section : {section_title}

Requirements:
- Start with intuition (why this matters / what the core idea is)
- Then give a precise technical explanation
- Include one concrete, specific example
- Use exact, non-generic language — avoid filler phrases
- Length: 150–200 words
- Do NOT mention sources, context, scores, or instructions

Reference material (use as background — do not copy directly):
{context}
"""


def generate_section(chapter_title: str, section_title: str, collection) -> str:
    query  = build_query(chapter_title, section_title)
    chunks = retrieve_chunks(collection, query)
    if not chunks:
        chunks = retrieve_chunks(collection, chapter_title, n=3)

    context = format_context(chunks)
    if len(context.split()) < 80:
        print("    Weak context detected — expanding fallback search")
        fallback = retrieve_chunks(collection, chapter_title, n=6, min_score=0.0)
        context = format_context(fallback)

    prompt  = build_prompt(chapter_title, section_title, context)
    text = call_groq(prompt)
    text = clean_text(text)

    if not is_valid(text):
        print("    Validation failed — retrying with stricter prompt...")
        stricter = prompt + (
            "\n\nIMPORTANT: Be specific. Avoid vague language. "
            "Ensure the section is complete and well-structured."
        )
        text = clean_text(call_groq(stricter))

    return text

def patch_chapter(chapter_title: str, section_title: str, new_text: str) -> bool:
    """
    Finds the chapter file containing this section and replaces the
    whole section body under the matching heading.
    Returns True if patch was successful.
    """
    section_header = re.escape(f"## {section_title}")
    section_pattern = re.compile(
        rf"({section_header}\s*\n)(.*?)(?=\n##\s+|\Z)",
        re.S,
    )

    for fname in sorted(os.listdir(BOOK_DIR)):
        if not fname.endswith(".md"):
            continue

        fpath = os.path.join(BOOK_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        if not re.search(section_header, content):
            continue

        if new_text.strip() in content:
            print(f"    Already patched — skipping")
            return True

        match = section_pattern.search(content)
        if not match:
            continue

        updated = content[: match.start(2)] + new_text.strip() + "\n\n" + content[match.end(2) :]
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(updated)
        print(f"    Patched into {fname}")
        return True

    print(f"    Could not find section '{section_title}' to patch — manual check needed")
    return False

def read_queue() -> list[tuple[str, str]]:
    if not os.path.exists(RETRY_LOG):
        return []
    entries = []
    with open(RETRY_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if " | " in line:
                chapter, section = line.split(" | ", 1)
                entries.append((chapter.strip(), section.strip()))
    return entries


def remove_from_queue(done: set[tuple[str, str]]) -> None:
    entries = read_queue()
    remaining = [
        f"{ch} | {sec}" for ch, sec in entries
        if (ch, sec) not in done
    ]
    with open(RETRY_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(remaining) + ("\n" if remaining else ""))


if __name__ == "__main__":
    queue = read_queue()

    if not queue:
        print(" retry_queue.txt is empty — nothing to do!")
        exit(0)

    print(f"\n Retrying {len(queue)} failed section(s)...\n")
    print("  Using slower pacing (15s pre-call + longer backoff) to avoid rate limits\n")

    collection = load_vector_store()

    succeeded: set[tuple[str, str]] = set()
    still_failed: list[tuple[str, str]] = []

    for i, (chapter_title, section_title) in enumerate(queue, 1):
        print(f"[{i}/{len(queue)}] {chapter_title} → {section_title}")

        try:
            text = generate_section(chapter_title, section_title, collection)
            patched = patch_chapter(chapter_title, section_title, text)

            if patched:
                succeeded.add((chapter_title, section_title))
                print(f"    Done\n")
            else:
                still_failed.append((chapter_title, section_title))
                print(f"    Generated but could not patch — saved to manual_fixes.txt\n")
                with open(os.path.join(OUTPUT_DIR, "manual_fixes.txt"), "a") as mf:
                    mf.write(f"=== {chapter_title} | {section_title} ===\n{text}\n\n")

        except Exception as e:
            print(f"    Failed again: {e}\n")
            still_failed.append((chapter_title, section_title))


    remove_from_queue(succeeded)


    print("─" * 50)
    print(f" Succeeded : {len(succeeded)}")
    print(f"Still failed: {len(still_failed)}")

    if still_failed:
        print(f"\n  {len(still_failed)} section(s) remain in retry_queue.txt")
        print("   → Run this script again after waiting a few minutes")
        print("   → Or check manual_fixes.txt if patch failed")
    else:
        print("\n All sections recovered — book is complete!")
        print(" Next: Step 8")