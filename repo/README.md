# LLM Book Project

Generate a professional textbook PDF from video transcripts using LLMs.

## Overview

This pipeline transforms video transcripts into a polished, textbook-quality PDF book about Large Language Models. It uses:

- **Groq** — For fast LLM inference (chapter generation, rewriting, fact verification)
- **ChromaDB** — Vector database for semantic search and context retrieval
- **Sentence Transformers** — Local embeddings for semantic similarity
- **DALL-E 3** — (Optional) AI-generated conceptual illustrations
- **WeasyPrint** — PDF generation from Markdown/HTML

## Prerequisites

### 1. Python Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys (`.env` file)

Create a `.env` file in the project root:

```env
# Required: Groq for LLM processing
GROQ_API_KEY=your_groq_api_key_here

# Optional: OpenAI for DALL-E image generation
# (leave blank to skip AI images)
OPENAI_API_KEY=your_openai_api_key_here
```

#### Getting API Keys

| Service | Where to get it |
|---------|-----------------|
| **Groq** | [console.groq.com](https://console.groq.com) — Free tier available |
| **OpenAI** | [platform.openai.com](https://platform.openai.com) — Paid, needed only for images |

> **Note:** If you don't add `OPENAI_API_KEY`, the diagram script will skip AI image generation but still create flowchart diagrams.

### 3. Data Requirements

Place your raw video transcripts in `data/raw/`:

```
data/
├── raw/
│   ├── video_01.json
│   ├── video_02.json
│   └── ...
```

Each transcript JSON should have this structure:

```json
{
  "video_id": "abc123",
  "video_index": 1,
  "title": "Introduction to Transformers",
  "chunks": [
    {"text": "First chunk content..."},
    {"text": "Second chunk content..."}
  ]
}
```

## Pipeline Scripts

Run scripts in order from the `scripts/` directory:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_get_transcripts.py` | Download video transcripts (if needed) |
| 2 | `02_clean_chunk.py` | Clean and normalize transcript chunks |
| 3 | `03_build_embeddings.py` | Build ChromaDB vector store |
| 4 | `04_generate_outline.py` | Generate book outline from transcripts |
| 5 | `05_fix_outline.py` | Refine and correct the generated outline |
| 6 | `06_generate_chapters.py` | Generate full chapter content |
| 7 | `07_retry_failed_sections.py` | Retry failed chapter sections |
| 8 | `08_polish_chapters.py` | Improve writing quality |
| 9 | `09_quality_control.py` | Check chapter quality |
| 10 | `10_rewrite_weak_sections.py` | Rewrite low-quality sections |
| 11 | `11_verify_facts.py` | Factual verification |
| 12 | `12_fix_failed_sections.py` | Fix failed verification sections |
| 13 | `13_generate_diagrams.py` | Generate diagrams + optional AI images |
| 14 | `14_build_pdf.py` | Build final PDF |

### Quick Start

```powershell
cd scripts

python 01_get_transcripts.py
python 02_clean_chunk.py
python 03_build_embeddings.py
python 04_generate_outline.py
python 05_fix_outline.py
python 06_generate_chapters.py
python 14_build_pdf.py
```

### Full Pipeline with All Steps

```powershell
cd scripts

python 01_get_transcripts.py
python 02_clean_chunk.py
python 03_build_embeddings.py
python 04_generate_outline.py
python 05_fix_outline.py
python 06_generate_chapters.py
python 07_retry_failed_sections.py
python 08_polish_chapters.py
python 09_quality_control.py
python 10_rewrite_weak_sections.py
python 11_verify_facts.py
python 12_fix_failed_sections.py
python 13_generate_diagrams.py
python 14_build_pdf.py
```

### Verify Facts Guidance

- Run `11_verify_facts.py` after the quality and rewrite steps.
- If a section still fails after 2-3 verification passes, do not keep rerunning it endlessly.
- Instead, inspect the failed section manually or adjust the chapter text before running `12_fix_failed_sections.py`.

## Project Structure

```
llm_book_project/
├── .env                    # API keys (create this)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/
│   ├── raw/                # Raw video transcripts
│   ├── clean/              # Cleaned transcripts
│   └── vectordb/           # ChromaDB vector store
├── output/
│   ├── book_outline.json   # Generated outline
│   ├── full_book.md        # Combined book markdown
│   ├── book/               # Individual chapters
│   │   ├── chapter_01.md
│   │   ├── chapter_02.md
│   │   └── ...
│   ├── diagrams/           # Generated images
│   └── LLM_Book.pdf        # Final output
└── scripts/
    ├── config.py           # API configuration
    ├── 01_get_transcripts.py
    ├── 02_clean_chunk.py
    ├── 03_build_embeddings.py
    ├── 04_generate_outline.py
    ├── 05_generate_chapters.py
    ├── 06_build_pdf.py
    └── ... (other scripts)
```

## Output

The final PDF is generated at: `output/LLM_Book.pdf`

## Troubleshooting

### "GROQ_API_KEY not found"
- Ensure `.env` file exists in project root
- Check the key is correctly formatted in `.env`

### "OPENAI_API_KEY not set" warnings
- This is normal if you didn't add an OpenAI key
- Diagram generation will continue without AI images

### ChromaDB errors
- Delete `data/vectordb/` folder and re-run `03_build_embeddings.py`

### PDF build fails
- Ensure `weasyprint` is installed: `pip install weasyprint`
- On Windows, may need GTK installed (see WeasyPrint docs)

## Customization

### Change the LLM Model

Edit `config.py` or individual scripts to use a different Groq model:

```python
MODEL = "llama-3.1-70b-versatile"  # or other available models
```

### Change Image Generation

In `12_generate_diagrams.py`:

```python
IMAGE_GEN_MODEL = "dall-e-2"  # cheaper alternative
```

### Adjust Chapter Length

In `05_generate_chapters.py`, modify `max_tokens` in the API call.

## License

MIT — Use as you like.