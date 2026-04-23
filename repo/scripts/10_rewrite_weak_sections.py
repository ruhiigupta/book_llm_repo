import os
import re
import json
import shutil
from pathlib import Path
from config import get_groq_client

BASE_DIR = Path(__file__).resolve().parent.parent
BOOK_DIR = BASE_DIR / "output" / "book"
BACKUP_DIR = BASE_DIR / "output" / "book_backup"
OUTPUT_DIR = BASE_DIR / "output"

client = get_groq_client()
MODEL = "llama-3.3-70b-versatile"


BAD_PATTERNS = [
    r"\badded transitional\b",
    r"\bkept the meaning\b",
    r"\bimproved clarity\b",
    r"\bremoved repetition\b",
    r"\bas an ai\b",
    r"\bi cannot\b",
    r"\bi don't have\b",
    r"•",
    r"￾",
    r"\[source:",
    r"\[from:",
]

MIN_WORDS = 300


def needs_rewrite(text):
    """Returns True if text has quality issues."""
    issues = 0
    for pattern in BAD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            issues += 1
    if len(text.split()) < MIN_WORDS:
        issues += 1
    return issues >= 1


def rewrite_section(section_title, text):
    """
    Rewrites one section at a time with strict grounding constraints.
    Smaller input = better output + no token overflow.
    """
    prompt = f"""You are a technical editor refining a section of a textbook on Large Language Models.

<section_title>{section_title}</section_title>

<original_text>
{text}
</original_text>

<task>
Rewrite the original text above into clean, professional, textbook-quality prose.
</task>

<strict_constraints>
1. GROUNDING: Every sentence in your output must trace directly to a sentence or phrase in the original text. Do not introduce facts, claims, definitions, examples, or analogies that are not already present.
2. FIDELITY: Preserve all technical terms, numerical values, acronyms, and named concepts exactly as they appear. Do not substitute synonyms for technical terms (e.g. do not replace "transformer" with "neural architecture").
3. SCOPE: Do not expand on ideas beyond what the original states. If the original is vague, keep it vague — do not clarify with new detail.
4. FORM: Write in continuous prose. No bullet points, no numbered lists, no headers, no tables.
5. TONE: Remove all meta-commentary, self-referential notes, and editing artifacts (e.g. "[needs citation]", "TODO", "rough draft").
6. OMISSION: If a sentence in the original is unclear but contains a technical claim, preserve its meaning faithfully rather than dropping it.
</strict_constraints>

<output_format>
Return only the rewritten text. No preamble, no explanation, no commentary.
</output_format>"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=700
    )

    return response.choices[0].message.content.strip()


def split_sections(content):
    """
    Splits chapter markdown into sections.
    Returns list of (title, text) tuples.
    """
    sections = []
    current_title = "Introduction"
    current_text = []

    for line in content.split("\n"):
        if line.startswith("## "):
            if current_text:
                sections.append((current_title, "\n".join(current_text).strip()))
            current_title = line[3:].strip()
            current_text = []
        else:
            current_text.append(line)

    if current_text:
        sections.append((current_title, "\n".join(current_text).strip()))

    return sections


def process():
    if not BOOK_DIR.exists():
        print(f" Book directory not found: {BOOK_DIR}")
        return


    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(BOOK_DIR, BACKUP_DIR)
    print(f" Backup created → output/book_backup/")

    files = sorted([f for f in os.listdir(BOOK_DIR) if f.endswith(".md")])
    rewrite_log = []
    total_rewritten = 0

    for filename in files:
        path = BOOK_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()


        sections = split_sections(content)
        chapter_modified = False
        new_sections = []

        for title, text in sections:
            if needs_rewrite(text):
                print(f"     Rewriting: [{filename}] → {title}")

                try:
                    new_text = rewrite_section(title, text)
                    new_sections.append((title, new_text))
                    rewrite_log.append({
                        "file": filename,
                        "section": title,
                        "status": "rewritten"
                    })
                    total_rewritten += 1
                    chapter_modified = True

                except Exception as e:
                    print(f"    Failed: {title} — {e}")
                    new_sections.append((title, text))  
                    rewrite_log.append({
                        "file": filename,
                        "section": title,
                        "status": f"failed: {e}"
                    })
            else:
                new_sections.append((title, text))

        
        if chapter_modified:
        
            new_content = ""
            for title, text in new_sections:
                if title == "Introduction":
                    new_content += text + "\n\n"
                else:
                    new_content += f"## {title}\n\n{text}\n\n"

            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(f"    Saved: {filename}")


    log_path = OUTPUT_DIR / "rewrite_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "total_sections_rewritten": total_rewritten,
            "details": rewrite_log
        }, f, indent=2)

    print(f"\n Rewrite Summary:")
    print(f"   Sections rewritten: {total_rewritten}")
    print(f"   Log saved → output/rewrite_log.json")
    print(f"   Backup preserved → output/book_backup/")

    if total_rewritten == 0:
        print("    All sections passed quality check — no rewrites needed")


if __name__ == "__main__":
    process()