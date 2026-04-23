import json
import os
import re
import time
from pathlib import Path

from config import get_groq_client

client = get_groq_client()

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
BOOK_DIR = OUTPUT_DIR / "book"
REPORT_PATH = OUTPUT_DIR / "verification_report.json"

MODEL = "llama-3.1-8b-instant"


def parse_status(result: str) -> str:
    match = re.search(r"STATUS:\s*(PASS|FAIL)", result or "", re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def parse_confidence(result: str) -> float:
    match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", result or "", re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def get_failed_items():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    failed = []
    for item in report.get("results", []):
        status = item.get("status", "").upper()
        if status == "FAIL":
            failed.append(item)
    return failed


def fix_section(section_text: str, file_name: str, section_id: int) -> str:
    prompt = f"""
    Corrects factual inaccuracies in a single markdown section.
    Minimal scope — no rewrites, no additions, no structural changes.
    """
    prompt = f"""You are a factual correction editor for a technical textbook on Large Language Models.

<source_file>{file_name}</source_file>
<section_id>{section_id}</section_id>

<original_section>
{section_text}
</original_section>

<task>
Correct ONLY the factual inaccuracies in the section above.
</task>

<correction_rules>
1. MINIMAL INTERVENTION: Change the smallest unit of text that fixes the error — a value, a name, a definition. Do not rewrite surrounding sentences unless they are themselves inaccurate.
2. PRESERVE VOICE: Keep the original sentence structure, phrasing, and tone. A corrected sentence should be indistinguishable in style from the original.
3. NO ADDITIONS: Do not introduce facts, examples, context, or caveats that are not already present in the original text.
4. NO OMISSIONS: Do not remove accurate content. If a sentence is correct, it must appear verbatim in the output.
5. NO STRUCTURE CHANGES: Do not add, remove, or modify headings, bullet points, paragraph breaks, or markdown formatting.
6. SCOPE: If the section contains no factual inaccuracies, return the original text unchanged.
</correction_rules>

<what_counts_as_a_factual_error>
- Wrong numerical values (parameter counts, dates, benchmark scores, layer counts)
- Incorrect technical definitions (e.g. misdescribing what attention computes)
- Wrong attribution (model names, authors, organizations, paper titles)
- Causal or mechanistic errors (e.g. claiming softmax produces logits, not probabilities)
- Contradictions within the section itself
</what_counts_as_a_factual_error>

<what_does_not_count>
- Imprecise but not wrong phrasing
- Stylistic choices (passive voice, word order, vocabulary preferences)
- Vagueness or omission of detail
- Disagreements in framing or emphasis
</what_does_not_count>

<output_format>
Return ONLY the corrected section body.
No preamble. No explanation. No markdown fences. No commentary.
If a correction was made, output the section with only that correction applied.
If no correction was needed, output the original section exactly.
</output_format>""".strip()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()


def get_section_spans(content: str):
    """
    Returns markdown section blocks for headings that start with '## '.
    Each span is:
      {
        "heading": "...",
        "start": int,       # start of heading
        "body_start": int,  # end of heading line
        "end": int          # start of next section heading or EOF
      }
    """
    matches = list(re.finditer(r"(?m)^##\s+.+$", content))
    spans = []

    for i, match in enumerate(matches):
        start = match.start()
        body_start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        spans.append(
            {
                "heading": match.group(0).rstrip(),
                "start": start,
                "body_start": body_start,
                "end": end,
            }
        )

    return spans


def replace_section_by_id(content: str, section_id: int, new_body: str):
    """
    Tries both 0-based and 1-based section indices.
    Only replaces the body under the matching heading.
    """
    spans = get_section_spans(content)
    if not spans:
        return content, False

    candidate_indices = [section_id, section_id - 1]

    for idx in candidate_indices:
        if 0 <= idx < len(spans):
            span = spans[idx]
            updated = (
                content[:span["body_start"]]
                + "\n\n"
                + new_body.strip()
                + "\n\n"
                + content[span["end"]:]
            )
            return updated, True

    return content, False


def update_chapter_file(file_name: str, section_id: int, section_text: str):
    file_path = BOOK_DIR / file_name
    if not file_path.exists():
        print(f" Missing file: {file_name}")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    fixed_body = fix_section(section_text, file_name, section_id)
    updated, ok = replace_section_by_id(content, section_id, fixed_body)

    if not ok:
        
        if section_text.strip() in content:
            updated = content.replace(section_text.strip(), fixed_body, 1)
            ok = True

    if ok:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated)
        print(f" Fixed only section {section_id} in {file_name}")
        return True

    print(f" Could not locate section {section_id} in {file_name}")
    return False


def main():
    if not REPORT_PATH.exists():
        print(" verification_report.json not found")
        return

    failed_items = get_failed_items()

    if not failed_items:
        print(" No FAIL sections found — nothing to fix")
        return

    print(f" Fixing {len(failed_items)} failed section(s) only...\n")

    fixed = 0
    skipped = 0

    for item in failed_items:
        file_name = item.get("file")
        section_id = item.get("section_id")
        title = item.get("title", "")

        if not file_name or section_id is None:
            print(f"Skipping incomplete item: {item}")
            skipped += 1
            continue

        
        file_path = BOOK_DIR / file_name
        if not file_path.exists():
            print(f" Missing file: {file_name}")
            skipped += 1
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            spans = get_section_spans(content)
            candidate_indices = [section_id, section_id - 1]
            section_text = None
            
            for idx in candidate_indices:
                if 0 <= idx < len(spans):
                    span = spans[idx]
                    section_text = content[span["body_start"]:span["end"]].strip()
                    break
            
            if not section_text:
                print(f" Could not find section {section_id} in {file_name}")
                skipped += 1
                continue

            if update_chapter_file(file_name, section_id, section_text):
                fixed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f" Error fixing {file_name} section {section_id}: {e}")
            skipped += 1

        time.sleep(1)  

    print("\n Fix pass complete")
    print(f"   Fixed : {fixed}")
    print(f"   Skipped: {skipped}")
    print("Next: rerun 11_verify_facts.py")


if __name__ == "__main__":
    main()