import os
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BOOK_DIR = BASE_DIR / "output" / "book"
OUTPUT_DIR = BASE_DIR / "output"


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
    r"This section could not be fully generated",
]


MIN_WORDS = 300
MAX_BULLET_RATIO = 3  


def check_quality(text, filename):
    issues = []
    words = len(text.split())


    if words < MIN_WORDS:
        issues.append(f"Too short ({words} words, min {MIN_WORDS})")


    for pattern in BAD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            issues.append(f"Bad pattern found: '{pattern}'")


    bullet_count = len(re.findall(r"^[\-\*•] ", text, re.MULTILINE))
    bullet_ratio = (bullet_count / max(1, words)) * 100
    if bullet_ratio > MAX_BULLET_RATIO:
        issues.append(f"Too many bullets ({bullet_count} bullets, ratio {bullet_ratio:.1f}%)")

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 20]
    if len(sentences) != len(set(sentences)):
        issues.append("Repeated sentences detected")

    return {
        "file": filename,
        "words": words,
        "issues": issues,
        "quality": "bad" if len(issues) >= 2 else "ok" if len(issues) == 1 else "good"
    }


def scan_book():
    if not BOOK_DIR.exists():
        print(f" Book directory not found: {BOOK_DIR}")
        return

    files = sorted([f for f in os.listdir(BOOK_DIR) if f.endswith(".md")])

    if not files:
        print(" No chapter files found")
        return

    print(f" Scanning {len(files)} chapters...\n")

    results = []
    bad_files = []
    ok_files = []

    for filename in files:
        path = BOOK_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        result = check_quality(content, filename)
        results.append(result)

        status = "✅" if result["quality"] == "good" else "⚠️ " if result["quality"] == "ok" else "❌"
        print(f"{status} {filename:<30} {result['words']:>5} words  {result['quality'].upper()}")

        if result["issues"]:
            for issue in result["issues"]:
                print(f"      → {issue}")

        if result["quality"] == "bad":
            bad_files.append(filename)
        elif result["quality"] == "ok":
            ok_files.append(filename)

    
    print(f"\n{'='*55}")
    print(f" Quality Summary:")
    print(f"   Good:  {len(files) - len(bad_files) - len(ok_files)}/{len(files)}")
    print(f"     OK:    {len(ok_files)}/{len(files)}")
    print(f"    Bad:   {len(bad_files)}/{len(files)}")

    if bad_files:
        print(f"\n   Files needing rewrite: {bad_files}")

    report_path = OUTPUT_DIR / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "total_chapters": len(files),
            "good": len(files) - len(bad_files) - len(ok_files),
            "ok": len(ok_files),
            "bad": len(bad_files),
            "bad_files": bad_files,
            "ok_files": ok_files,
            "details": results
        }, f, indent=2)

    print(f"\n Report saved → output/quality_report.json")
    return bad_files


if __name__ == "__main__":
    scan_book()