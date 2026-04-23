import os
import json
import re
import time
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

import chromadb
from config import get_groq_client
from sentence_transformers import SentenceTransformer

groq_client = get_groq_client()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
BOOK_DIR = os.path.join(OUTPUT_DIR, "book")
DB_DIR = os.path.join(BASE_DIR, "data", "vectordb")
OUTLINE_PATH = os.path.join(OUTPUT_DIR, "book_outline.json")
RETRY_LOG = os.path.join(OUTPUT_DIR, "retry_queue.txt")

MODEL = "llama-3.3-70b-versatile"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BOOK_DIR, exist_ok=True)

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded")


def load_vector_store():
    chroma = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma.get_collection("transcripts")
    print(f"Vector store loaded — {collection.count()} vectors")
    return collection


def build_query(chapter_title: str, section_title: str) -> str:
    return (
        f"{section_title}. {chapter_title}. "
        "Retrieve definitions, intuition, mechanisms, a concrete example, and implementation details."
    )


def retrieve_chunks(collection, query: str, n: int = 10, min_score: float = 0.25) -> list[str]:
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    scored = []
    seen = set()
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
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


def format_context(chunks: list[str], max_chars: int = 2400) -> str:
    buf = []
    total = 0
    for chunk in chunks:
        if total + len(chunk) > max_chars:
            break
        buf.append(chunk)
        total += len(chunk)
    return "\n\n".join(buf).strip()


def normalize_for_compare(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def paragraph_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_for_compare(a), normalize_for_compare(b)).ratio()


def sentence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_for_compare(a), normalize_for_compare(b)).ratio()


def is_meta_line(line: str) -> bool:
    low = line.lower().strip()
    return low.startswith(("source:", "score:", "context:", "instruction:", "instructions:"))


def clean_text(text: str, section_title: str = "") -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")
    text = re.sub(r"^\s*```(?:json)?\s*$", "", text, flags=re.M | re.I)
    text = re.sub(r"^\s*```\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, flags=re.M)
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.M)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()

    title_norm = normalize_for_compare(section_title) if section_title else ""
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    cleaned_paragraphs = []
    seen_paragraphs = []
    seen_sentences = []

    for para in raw_paragraphs:
        para = re.sub(r"^#+\s*", "", para).strip()
        para = re.sub(r"\s+([.,;:!?])", r"\1", para)
        para = re.sub(r"\s{2,}", " ", para).strip()

        if not para:
            continue

        if is_meta_line(para):
            continue

        para_norm = normalize_for_compare(para)
        if not para_norm:
            continue

        if title_norm and (para_norm == title_norm or title_norm in para_norm) and len(para_norm) < 120:
            continue

        if any(paragraph_similarity(para, prev) > 0.90 for prev in cleaned_paragraphs):
            continue

        sentences = split_sentences(para)
        local_seen = set()
        deduped_sentences = []

        for sent in sentences:
            sent = sent.strip()
            sent = re.sub(r"\s+([.,;:!?])", r"\1", sent)
            sent = re.sub(r"\s{2,}", " ", sent).strip()
            if not sent:
                continue

            sent_norm = normalize_for_compare(sent)
            if not sent_norm:
                continue

            if title_norm and (sent_norm == title_norm or title_norm in sent_norm) and len(sent_norm) < 120:
                continue

            if sent_norm in local_seen:
                continue

            if any(sentence_similarity(sent, prev) > 0.92 for prev in seen_sentences):
                continue

            local_seen.add(sent_norm)
            seen_sentences.append(sent)
            deduped_sentences.append(sent)

        if not deduped_sentences:
            continue

        para_out = " ".join(deduped_sentences)
        para_out = re.sub(r"\s+([.,;:!?])", r"\1", para_out)
        para_out = re.sub(r"\s{2,}", " ", para_out).strip()

        para_out_norm = normalize_for_compare(para_out)
        if any(paragraph_similarity(para_out, prev) > 0.90 for prev in cleaned_paragraphs):
            continue

        if para_out_norm in seen_paragraphs:
            continue

        seen_paragraphs.append(para_out_norm)
        cleaned_paragraphs.append(para_out)

    return "\n\n".join(cleaned_paragraphs).strip()


def parse_groq_wait(err: str) -> float:
    try:
        hours = re.search(r"(\d+)h", err)
        minutes = re.search(r"(\d+)m", err)
        seconds = re.search(r"(\d+(?:\.\d+)?)s", err)
        parsed = 0.0
        if hours:
            parsed += int(hours.group(1)) * 3600
        if minutes:
            parsed += int(minutes.group(1)) * 60
        if seconds:
            parsed += float(seconds.group(1))
        if parsed > 0:
            return parsed + 5
    except Exception:
        pass
    return 0.0


def call_groq(prompt: str, model: str = MODEL, retries: int = 5, delay: int = 20, max_tokens: int = 300) -> str:
    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            time.sleep(2)
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            wait = parse_groq_wait(err)
            if wait == 0.0:
                wait = delay * (2 ** attempt)
            wait = min(wait, 900)
            print(f"Groq rate limit - waiting {wait:.0f}s (attempt {attempt + 1}/{retries})...")
            time.sleep(wait)

    raise RuntimeError("Groq failed after all retries")


def load_existing_retry_entries() -> set[str]:
    if not os.path.exists(RETRY_LOG):
        return set()
    with open(RETRY_LOG, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def log_retry(chapter_title: str, section_title: str) -> None:
    entry = f"{chapter_title} | {section_title}"
    existing = load_existing_retry_entries()
    if entry not in existing:
        with open(RETRY_LOG, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    print(f"Queued for retry: {section_title}")


def has_repetition_issue(text: str) -> bool:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    if len(paragraphs) < 2:
        return True

    norm_paras = [normalize_for_compare(p) for p in paragraphs if normalize_for_compare(p)]
    if len(norm_paras) != len(set(norm_paras)):
        return True

    for i in range(len(paragraphs)):
        for j in range(i + 1, len(paragraphs)):
            if paragraph_similarity(paragraphs[i], paragraphs[j]) > 0.90:
                return True

    sentences = split_sentences(text)
    norm_sentences = [normalize_for_compare(s) for s in sentences if normalize_for_compare(s)]
    counts = Counter(norm_sentences)
    if any(c > 1 for c in counts.values()):
        return True

    if len(norm_sentences) >= 4 and len(set(norm_sentences)) / len(norm_sentences) < 0.85:
        return True

    return False


def is_valid(text: str, section_title: str = "") -> bool:
    t = text.lower()
    words = len(text.split())
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    sentences = split_sentences(text)

    if not (70 <= words <= 260):
        return False

    if len(paragraphs) < 2 or len(paragraphs) > 5:
        return False

    if len(sentences) < 2:
        return False

    if any(x in t for x in ["source:", "score:", "context:", "instructions:", "instruction:"]):
        return False

    if has_repetition_issue(text):
        return False

    if section_title:
        title_norm = normalize_for_compare(section_title)
        body_norm = normalize_for_compare(text[:250])
        if title_norm and title_norm in body_norm and len(body_norm) < 150:
            return False

    return True


def build_prompt(chapter_title: str, section_title: str, context: str) -> str:
    return f"""You are writing one section of a professional, textbook-quality technical book titled "Building LLMs from Scratch".

Audience:
Students learning how to build large language models from the ground up using open-source tools and code.

Goal:
Write a clear, accurate, self-contained textbook section for the following chapter and section.

Chapter: {chapter_title}
Section: {section_title}

Strict rules:
- Use only the provided context
- Do not add outside knowledge
- Do not invent numbers, model names, formulas, methods, or examples unless supported by the context
- Do not mention the words source, context, retrieval, score, or instructions
- Keep the writing natural, professional, and textbook-like
- Avoid generic filler phrases like "in the realm of", "it is crucial", "this approach", "to illustrate this concept"
- If the context is weak or incomplete, stay conservative and explain only what is clearly supported
- Do not repeat sentences or phrases
- Do not restate the same idea in slightly different wording
- Do not include headings or titles in the output
- Do not repeat the section title in the body
- Each paragraph must introduce new information

Structure requirements:
- Start with intuition: why this concept matters
- Then give a precise technical explanation
- Then include one concrete, specific example only if it is supported by the context
- End with a short takeaway or implication
- Length: 160-220 words
- Use 2-4 short paragraphs
- Do not use bullet points unless absolutely necessary

Quality bar:
The section should read like a polished textbook paragraph, not an AI-generated summary.

If a concept is not clearly supported by the context:
- Do not introduce it
- Do not guess
- Do not expand beyond the given material
- Omit the detail rather than hallucinating

Every technical claim must be traceable to the context.

Reference material:
{context}
"""


def generate_section(chapter_title: str, section_title: str, collection) -> str:
    query = build_query(chapter_title, section_title)
    chunks = retrieve_chunks(collection, query)

    if not chunks:
        print("No chunks found - falling back to chapter-level query")
        chunks = retrieve_chunks(collection, chapter_title, n=3)

    context = format_context(chunks)

    if len(context.split()) < 80:
        print("Weak context detected - expanding fallback search")
        fallback = retrieve_chunks(collection, chapter_title, n=6, min_score=0.0)
        context = format_context(fallback)

    if len(context.split()) < 80:
        print("Weak context still detected - skipping")
        log_retry(chapter_title, section_title)
        raise RuntimeError(f"Insufficient context for: {section_title}")

    prompt = build_prompt(chapter_title, section_title, context)

    try:
        text = call_groq(prompt)
    except Exception:
        log_retry(chapter_title, section_title)
        raise RuntimeError(f"Primary generation failed for: {section_title}")

    text = clean_text(text, section_title)

    if not is_valid(text, section_title):
        print("Validation failed - retrying with stricter prompt")
        stricter = prompt + "\n\nWrite only grounded, specific, well-structured prose. Do not be vague. Do not invent details. Keep every claim supported by the context. Do not repeat ideas."
        try:
            text = clean_text(call_groq(stricter), section_title)
        except Exception:
            log_retry(chapter_title, section_title)
            raise RuntimeError(f"Validation retry failed for: {section_title}")

        if not is_valid(text, section_title):
            log_retry(chapter_title, section_title)
            raise RuntimeError(f"Output still invalid after retry: {section_title}")

    return text


def generate_chapter(collection, chapter: dict) -> str:
    num = chapter["chapter_number"]
    title = chapter["chapter_title"]
    sections = chapter["sections"]

    print(f"\nChapter {num}: {title}")

    chapter_text = f"# Chapter {num}: {title}\n\n"
    failed_sections = []

    for sec in sections:
        print(sec)
        try:
            sec_text = generate_section(title, sec, collection)
            chapter_text += f"## {sec}\n\n{sec_text}\n\n"
            print("Done")
        except RuntimeError as e:
            print(f"Skipped: {e}")
            failed_sections.append(sec)
            chapter_text += f"## {sec}\n\nThis section could not be fully generated from the available source context.\n\n"

        time.sleep(10)

    if failed_sections:
        print(f"\n{len(failed_sections)} section(s) failed in Chapter {num}. Check retry_queue.txt")

    return chapter_text


def save_chapter(num: int, text: str) -> None:
    path = os.path.join(BOOK_DIR, f"chapter_{num:02d}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: chapter_{num:02d}.md")


def chapter_exists(num: int) -> bool:
    return os.path.exists(os.path.join(BOOK_DIR, f"chapter_{num:02d}.md"))


if __name__ == "__main__":
    with open(OUTLINE_PATH, "r", encoding="utf-8") as f:
        outline = json.load(f)

    collection = load_vector_store()
    chapters = outline["chapters"]

    print(f"\nGenerating {len(chapters)} chapters...\n")

    for ch in chapters:
        num = ch["chapter_number"]

        if chapter_exists(num):
            print(f"Skipping Chapter {num} (already exists)")
            continue

        try:
            text = generate_chapter(collection, ch)
            save_chapter(num, text)
        except Exception as e:
            print(f"\nChapter {num} failed entirely: {e}")
            print("Waiting 20s before next chapter...")
            time.sleep(20)

    print("\nStep 5 COMPLETE - Book Generated")

    if os.path.exists(RETRY_LOG):
        with open(RETRY_LOG, "r", encoding="utf-8") as f:
            retries = f.readlines()
        if retries:
            print(f"{len(retries)} section(s) need retry - see: {RETRY_LOG}")
    else:
        print("No failed sections - clean run!")

    print("Next: Step 7")