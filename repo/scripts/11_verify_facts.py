import os
import json
import re
import time
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import get_groq_client

client = get_groq_client()

BASE_DIR = Path(__file__).resolve().parent.parent
BOOK_DIR = BASE_DIR / "output" / "book"
OUTPUT_DIR = BASE_DIR / "output"
DB_DIR = BASE_DIR / "data" / "vectordb"
REPORT_PATH = OUTPUT_DIR / "verification_report.json"

MODEL = "llama-3.1-8b-instant"

MIN_SOURCE_SCORE = 0.35
MAX_SECTION_CHARS = 700
MAX_CONTEXT_CHARS = 1000
CALL_PAUSE_SECONDS = 0.5

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(" Model loaded")


def load_vector_store():
    if not DB_DIR.exists():
        raise FileNotFoundError(f"Vector DB not found: {DB_DIR}")

    chroma = chromadb.PersistentClient(path=str(DB_DIR))
    collection = chroma.get_collection("transcripts")
    print(f" Vector store loaded — {collection.count()} vectors")
    return collection


def retrieve_context(collection, query: str, n: int = 4) -> tuple[str, float]:
    embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    best_score = 0.0

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        score = round(1 - dist, 3)
        best_score = max(best_score, score)

        if score >= MIN_SOURCE_SCORE:
            title = meta.get("title", "Unknown")
            chunks.append(f"[{title} | score:{score}]\n{doc}")

    context = "\n\n".join(chunks) if chunks else "No relevant source found."
    return context[:MAX_CONTEXT_CHARS], best_score


def split_sections(text: str) -> list[dict]:
    matches = list(re.finditer(r"(?m)^##\s+(.+)$", text))
    sections = []

    if not matches:
        return sections

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        if len(body) > 50:
            sections.append({"title": title, "body": body})

    return sections


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_model_json(raw: str) -> dict:
    raw = strip_code_fences(raw)
    return json.loads(raw)


def normalize_confidence(status: str, parsed_confidence: float, support_strength: float) -> float:
    parsed_confidence = max(0.0, min(1.0, float(parsed_confidence)))
    support_strength = max(0.0, min(1.0, float(support_strength)))

    if status == "PASS":
        anchored = 0.45 + (0.40 * support_strength)
        confidence = min(parsed_confidence, anchored, 0.92)
        return round(confidence, 2)

    if status == "FAIL":
        anchored = 0.60 + (0.30 * (1.0 - support_strength))
        confidence = max(parsed_confidence, anchored)
        return round(min(confidence, 0.99), 2)

    return round(parsed_confidence, 2)


def verify_section(title: str, body: str, collection) -> dict:
    body = body[:MAX_SECTION_CHARS]
    context, support_strength = retrieve_context(collection, f"{title} {body[:200]}")

    if "No relevant source found." in context:
        return {
            "status": "FAIL",
            "confidence": 0.0,
            "reason": "No supporting source found in transcript vector store.",
            "raw": "",
            "support_strength": support_strength,
        }

    prompt = f"""
    Verifies factual consistency between a section and its source content.
    Returns structured JSON with verdict, confidence, and granular evidence.
    """
    prompt = f"""You are a factual consistency verifier for a technical textbook on Large Language Models.

<source_content>
{context}
</source_content>

<section_to_verify>
<title>{title}</title>
<body>{body}</body>
</section_to_verify>

<task>
Determine whether every factual claim in the section is directly supported by the source content above.
Do NOT rewrite or suggest edits. Only assess.
</task>

<verification_rules>
1. GROUNDING: Each claim in the section must map to an explicit statement in the source content. Inferences, extrapolations, and reasonable extensions do NOT count as support.
2. DEFINITIONS: Any definition of a technical term must match the source's definition exactly in meaning. A definition not present in the source is unsupported.
3. NUMBERS AND NAMES: Any specific value (parameter counts, dates, percentages, model names, author names) must appear verbatim in the source.
4. OMISSION IS NOT FAILURE: The section may omit information from the source — that is fine. Only additions and contradictions are failures.
5. SEVERITY: Distinguish between critical failures (wrong facts, invented definitions, fabricated values) and minor issues (slight imprecision, overly general phrasing).
</verification_rules>

<confidence_scale>
Assign confidence based on how strongly the source supports the section:
- 0.9–1.0: Every claim has a direct, unambiguous match in the source
- 0.7–0.89: Most claims are supported; one or two are general but not contradicted
- 0.5–0.69: Some claims are supported but others are vague, imprecise, or weakly grounded
- 0.0–0.49: One or more claims are contradicted, invented, or have no basis in the source
</confidence_scale>

<decision_rule>
- PASS if confidence >= 0.70 AND no critical failures exist
- FAIL if confidence < 0.70 OR any critical failure exists
</decision_rule>

<output_format>
Return ONLY a single valid JSON object. No preamble, no explanation outside the JSON, no markdown fences.

Schema:
{{
  "status": "PASS" or "FAIL",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one sentence stating the primary finding>",
  "unsupported_claims": ["<verbatim or near-verbatim claim from section>", ...],
  "critical_failure": true or false
}}

If there are no unsupported claims, set "unsupported_claims" to [].
</output_format>""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=180,
        )
        raw = response.choices[0].message.content.strip()
        data = parse_model_json(raw)

        status = str(data.get("status", "UNKNOWN")).upper().strip()
        if status not in {"PASS", "FAIL"}:
            status = "UNKNOWN"

        parsed_confidence = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", "Could not parse")).strip()

        if support_strength < MIN_SOURCE_SCORE + 0.05 and status == "PASS":
            status = "FAIL"
            reason = "Source support was too weak to justify a PASS."

        confidence = normalize_confidence(status, parsed_confidence, support_strength)

        return {
            "status": status,
            "confidence": confidence,
            "reason": reason,
            "raw": raw,
            "support_strength": round(support_strength, 3),
        }

    except json.JSONDecodeError:
        return {
            "status": "ERROR",
            "confidence": 0.0,
            "reason": "Model response was not valid JSON",
            "raw": "",
            "support_strength": round(support_strength, 3),
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "confidence": 0.0,
            "reason": str(e),
            "raw": "",
            "support_strength": round(support_strength, 3),
        }


def load_previous_report():
    if not REPORT_PATH.exists():
        return {}

    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    previous = {}
    for item in data.get("results", []):
        key = (item.get("file"), item.get("section_id"))
        previous[key] = item
    return previous


def main():
    if not BOOK_DIR.exists():
        print(f" Book directory not found: {BOOK_DIR}")
        return

    try:
        collection = load_vector_store()
    except Exception as e:
        print(f" Failed to load vector store: {e}")
        return

    previous_report = load_previous_report()
    files = sorted([f for f in os.listdir(BOOK_DIR) if f.endswith(".md")])
    print(f"\n Verifying {len(files)} chapters...\n")

    all_results = []
    total_pass = 0
    total_fail = 0
    total_error = 0
    total_skipped = 0

    for filename in files:
        path = BOOK_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = split_sections(content)
        print(f" {filename} — {len(sections)} sections")

        file_results = []

        for i, sec in enumerate(sections):
            title = sec["title"]
            body = sec["body"]
            prev = previous_report.get((filename, i))

            # Skip previously passed sections
            if prev and prev.get("status") == "PASS":
                file_results.append(prev)
                total_pass += 1
                total_skipped += 1
                print(f"   [{i+1}/{len(sections)}] {title[:50]}...  SKIPPED (previous PASS)")
                continue

            print(f"   [{i+1}/{len(sections)}] {title[:50]}...")
            result = verify_section(title, body, collection)
            status = result["status"]

            icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
            print(f"   {icon} {status} ({result['confidence']:.2f}) — {result['reason'][:80]}")

            if status == "PASS":
                total_pass += 1
            elif status == "FAIL":
                total_fail += 1
            else:
                total_error += 1

            file_results.append({
                "file": filename,
                "section_id": i,
                "title": title,
                "status": status,
                "confidence": result["confidence"],
                "support_strength": result.get("support_strength", 0.0),
                "reason": result["reason"],
                "raw": result.get("raw", ""),
            })

            time.sleep(CALL_PAUSE_SECONDS)

        all_results.extend(file_results)
        print()

    report = {
        "total_sections": len(all_results),
        "pass": total_pass,
        "fail": total_fail,
        "error": total_error,
        "skipped_pass": total_skipped,
        "pass_rate": f"{(total_pass / max(1, len(all_results)) * 100):.1f}%",
        "results": all_results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 55)
    print(" Verification Summary:")
    print(f"    PASS:  {total_pass}")
    print(f"    FAIL:  {total_fail}")
    print(f"     ERROR: {total_error}")
    print(f"    SKIPPED (already PASS): {total_skipped}")
    print(f"   Pass rate: {report['pass_rate']}")
    print(f"\n Report saved → output/verification_report.json")

    if total_fail > 0:
        print(f"\n  {total_fail} sections failed — run 12_fix_failed_sections.py")
    else:
        print("\n All verified sections passed — ready for diagrams")

    print(" Next: python 13_generate_diagrams.py")


if __name__ == "__main__":
    main()