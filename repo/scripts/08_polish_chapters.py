import os
import re
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BOOK_DIR = BASE_DIR / "output" / "book"
BACKUP_DIR = BASE_DIR / "output" / "book_pre_polish"

GENERIC_PHRASES = [
    "in the realm of",
    "to illustrate this concept",
    "it is crucial",
    "the core idea is",
    "in this context",
]

AI_PHRASES = [
    "as we delve into",
    "it becomes increasingly evident",
    "in this chapter",
    "we will explore",
]


def remove_editor_notes(text):
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        l = line.strip().lower()

        if re.match(r"^\d+\.\s*(removed|added|improved|simplified|kept)", l):
            continue
        if l.startswith("•") and any(word in l for word in [
            "removed", "added", "improved", "simplified",
            "kept", "adjusted", "changed", "emphasized"
        ]):
            continue
        if any(phrase in l for phrase in [
            "removed unnecessary",
            "improved clarity",
            "made minor adjustments",
            "added transitional",
            "kept the original meaning"
        ]):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def remove_duplicate_lines(text):
    lines = text.split("\n")
    cleaned = []
    prev = ""
    for line in lines:
        if line.strip() and line.strip().lower() == prev.strip().lower():
            continue
        cleaned.append(line)
        prev = line
    return "\n".join(cleaned)


def remove_duplicate_bold_titles(text):
    return re.sub(r"\n\*\*(.*?)\*\*\n", "\n", text)


def fix_commas(text):
    text = text.replace(".,", ".")
    text = text.replace(" ,", ",")
    text = text.replace("\n,", "\n")
    text = re.sub(r"\.\s*,", ".", text)
    return text


def fix_encoding(text):
    
    text = text.replace("￾", "")
    text = text.replace("\x00", "")
    text = re.sub(r"[\x80-\x9f]", "", text) 
    return text


def clean_generic_phrases(text):
    for phrase in GENERIC_PHRASES:
        
        text = re.sub(
            rf"\b{re.escape(phrase)}\b,?\s*",
            " ",
            text,
            flags=re.IGNORECASE
        )
    return text


def remove_ai_phrases(text):
    for phrase in AI_PHRASES:
        text = re.sub(
            rf"\b{re.escape(phrase)}\b,?\s*",
            " ",
            text,
            flags=re.IGNORECASE
        )
    return text


def apply_grammar_fixes(text):
    fixes = [
        (r"\bLLMs natural\b", "LLMs in natural"),
        (r"\bLLMs ability\b", "LLMs' ability"),
        (r"\bmodel performance is\b", "the model performance is"),
        (r"\btokenization is,?\s*", "Tokenization is "),
    ]
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text)
    return text


def clean_spacing(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^(#{1,6} .+)\n{3,}", r"\1\n\n", text)
    text = re.sub(r"(?m)^(#{1,6} .+)\n{2,}(?=[^#])", r"\1\n\n", text)
    return text.strip()


def count_changes(original, modified):
    orig_lines = set(original.split("\n"))
    mod_lines = set(modified.split("\n"))
    removed = orig_lines - mod_lines
    return len(removed)


def polish_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        original = f.read()

    content = original

    
    content = remove_editor_notes(content)
    content = remove_duplicate_lines(content)
    content = remove_duplicate_bold_titles(content)
    content = fix_encoding(content)
    content = fix_commas(content)
    content = apply_grammar_fixes(content)
    content = remove_ai_phrases(content)
    content = clean_generic_phrases(content)
    content = clean_spacing(content)

    
    changes = count_changes(original, content)
    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f" Polished {filepath.name} ({changes} lines changed)")
    else:
        print(f" Skipped {filepath.name} (no changes needed)")

    return changes


def main():
    if not BOOK_DIR.exists():
        print(f" Book directory not found: {BOOK_DIR}")
        return

    
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(BOOK_DIR, BACKUP_DIR)
    print(f" Backup created → output/book_pre_polish/\n")

    files = sorted([f for f in BOOK_DIR.iterdir() if f.suffix == ".md"])
    total_changes = 0

    for filepath in files:
        changes = polish_file(filepath)
        total_changes += changes

    print(f"\n Polish Summary:")
    print(f"   Files processed: {len(files)}")
    print(f"   Total lines changed: {total_changes}")
    print(f"   Backup preserved → output/book_pre_polish/")
    print(f"\n Next: python 09_quality_control.py")


if __name__ == "__main__":
    main()