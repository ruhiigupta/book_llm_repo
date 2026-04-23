import os
import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from config import get_groq_client
import requests

client = get_groq_client()


IMAGE_GEN_MODEL = "dall-e-3"
IMAGE_GEN_SIZE = "1024x1024"

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
BOOK_DIR = OUTPUT_DIR / "book"
DIAGRAM_DIR = OUTPUT_DIR / "diagrams"
OUTLINE_PATH = OUTPUT_DIR / "book_outline.json"

MODEL = "llama-3.1-8b-instant"

DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def load_outline():
    with open(OUTLINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chapter_markdown(chapter_num: int) -> str:
    path = BOOK_DIR / f"chapter_{chapter_num:02d}.md"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_chapter_markdown(chapter_num: int, content: str) -> None:
    path = BOOK_DIR / f"chapter_{chapter_num:02d}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def fallback_spec(chapter_title: str, sections: list[str]) -> dict:
    nodes = [{"id": "start", "label": chapter_title}]
    for i, sec in enumerate(sections[:5], 1):
        nodes.append({"id": f"n{i}", "label": sec})
    nodes.append({"id": "end", "label": "Key takeaway"})

    edges = [{"source": "start", "target": "n1"}]
    for i in range(1, len(nodes) - 2):
        edges.append({"source": f"n{i}", "target": f"n{i+1}"})
    if len(nodes) > 2:
        edges.append({"source": f"n{len(nodes)-2}", "target": "end"})

    return {
        "title": chapter_title,
        "caption": f"Figure: {chapter_title}",
        "nodes": nodes,
        "edges": edges,
    }


def build_diagram_prompt(chapter: dict, chapter_content: str = "") -> str:
    sections = chapter.get("sections", [])
    
    
    title_lower = chapter["chapter_title"].lower()
    
    
    if "transformer" in title_lower or "attention" in title_lower:
        diagram_type = "architecture"
        style_hint = "Show the encoder-decoder architecture with attention layers"
    elif "training" in title_lower or "gradient" in title_lower or "backprop" in title_lower:
        diagram_type = "process"
        style_hint = "Show the training loop with forward pass, loss, backprop"
    elif "token" in title_lower or "embedding" in title_lower or "vocab" in title_lower:
        diagram_type = "conversion"
        style_hint = "Show text → tokens → embeddings transformation"
    elif "inference" in title_lower or "generate" in title_lower or "prompt" in title_lower:
        diagram_type = "flow"
        style_hint = "Show the autoregressive generation process"
    elif "rl" in title_lower or "reinforcement" in title_lower or "reward" in title_lower:
        diagram_type = "loop"
        style_hint = "Show the RLHF feedback loop"
    elif "evaluation" in title_lower or "benchmark" in title_lower or "metric" in title_lower:
        diagram_type = "comparison"
        style_hint = "Show evaluation metrics and benchmarks"
    else:
        diagram_type = "concept"
        style_hint = "Show the key concepts and their relationships"
    
    return f"""
You are a diagram-planning agent for a technical textbook on Large Language Models.

Create a single, meaningful diagram as STRICT JSON only.

DIAGRAM TYPE: {diagram_type}
{style_hint}

Requirements:
- Create 4-7 nodes that represent REAL concepts from this chapter, not just section titles
- Node labels should be educational: e.g., "Input Text" not "1. Introduction"
- Add edge labels that explain the relationship: e.g., "tokenized into", "produces", "feeds into"
- Make it visually intuitive for a student learning this topic
- Do not include code blocks
- Do not include markdown fences
- Output valid JSON only

JSON schema:
{{
  "title": "descriptive diagram title",
  "caption": "caption for the textbook figure",
  "style": "flowchart|architecture|process|cycle",
  "nodes": [
    {{"id": "n1", "label": "Concept A"}},
    {{"id": "n2", "label": "Concept B"}}
  ],
  "edges": [
    {{"source": "n1", "target": "n2", "label": "relationship"}}
  ]
}}

Chapter title: {chapter["chapter_title"]}

Sections:
{json.dumps(sections, ensure_ascii=False)}

Chapter content preview (use for context):
{chapter_content[:500] if chapter_content else "N/A"}
""".strip()


def parse_spec(raw: str, chapter: dict) -> dict:
    raw = strip_code_fences(raw)
    try:
        spec = json.loads(raw)
        if not isinstance(spec, dict):
            raise ValueError("spec is not an object")
        if "nodes" not in spec or "edges" not in spec:
            raise ValueError("missing nodes/edges")
        return spec
    except Exception:
        return fallback_spec(chapter["chapter_title"], chapter.get("sections", []))


def generate_spec(chapter: dict, chapter_content: str = "") -> dict:
    prompt = build_diagram_prompt(chapter, chapter_content)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600,
    )
    raw = response.choices[0].message.content.strip()
    return parse_spec(raw, chapter)


def normalize_spec(spec: dict, chapter: dict) -> dict:
    title = str(spec.get("title") or chapter["chapter_title"]).strip()
    caption = str(spec.get("caption") or f"Figure: {chapter['chapter_title']}").strip()
    style = str(spec.get("style") or "flowchart").strip().lower()

    nodes = spec.get("nodes", [])
    edges = spec.get("edges", [])

    clean_nodes = []
    seen = set()
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or f"n{idx+1}").strip()
        label = str(node.get("label") or node_id).strip()
        if node_id in seen:
            continue
        seen.add(node_id)
        clean_nodes.append({"id": node_id, "label": label})

    if len(clean_nodes) < 2:
        return fallback_spec(chapter["chapter_title"], chapter.get("sections", []))

    valid_ids = {n["id"] for n in clean_nodes}
    clean_edges = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("source") or "").strip()
        dst = str(edge.get("target") or "").strip()
        label = str(edge.get("label") or "").strip()
        if src in valid_ids and dst in valid_ids:
            clean_edges.append({"source": src, "target": dst, "label": label})

    if not clean_edges:
        clean_edges = [
            {"source": clean_nodes[i]["id"], "target": clean_nodes[i + 1]["id"], "label": ""}
            for i in range(len(clean_nodes) - 1)
        ]

    return {
        "title": title,
        "caption": caption,
        "style": style,
        "nodes": clean_nodes[:6],
        "edges": clean_edges[:8],
    }


def load_font(size: int):

    windows_candidates = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
    ]
    for path in windows_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    

    linux_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in linux_candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    
    return ImageFont.load_default()


def wrap_label(text: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) if len(text) > width else text


def draw_arrow(draw, start, end, fill, width=4):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=fill, width=width)


    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = 16
    head_angle = 0.45

    p1 = (x2, y2)
    p2 = (
        x2 - head_len * math.cos(angle - head_angle),
        y2 - head_len * math.sin(angle - head_angle),
    )
    p3 = (
        x2 - head_len * math.cos(angle + head_angle),
        y2 - head_len * math.sin(angle + head_angle),
    )
    draw.polygon([p1, p2, p3], fill=fill)


def render_png(spec: dict, out_path: Path) -> None:
    title = spec["title"]
    nodes = spec["nodes"]
    edges = spec["edges"]
    style = spec.get("style", "flowchart")

    width = 1600
    top_margin = 140
    bottom_margin = 90
    side_margin = 120
    box_w = 1120
    box_h = 110
    gap = 70

    
    style_colors = {
        "architecture": {"fill": "#e3f2fd", "outline": "#1565c0"}, 
        "process": {"fill": "#e8f5e9", "outline": "#2e7d32"},      
        "flow": {"fill": "#fff3e0", "outline": "#ef6c00"},          
        "cycle": {"fill": "#fce4ec", "outline": "#c2185b"},         
        "comparison": {"fill": "#f3e5f5", "outline": "#7b1fa2"},   
        "flowchart": {"fill": "#f8f8f8", "outline": "#333333"},     
    }
    colors = style_colors.get(style, style_colors["flowchart"])

    height = top_margin + bottom_margin + len(nodes) * box_h + (len(nodes) - 1) * gap
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    title_font = load_font(42)
    node_font = load_font(28)
    caption_font = load_font(22)


    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_x = (width - (title_bbox[2] - title_bbox[0])) // 2
    draw.text((title_x, 40), title, fill="black", font=title_font)

    center_x = width // 2
    box_x1 = center_x - box_w // 2
    box_x2 = center_x + box_w // 2

    positions = {}
    y = top_margin
    for node in nodes:
        positions[node["id"]] = (box_x1, y, box_x2, y + box_h)
        y += box_h + gap


    for edge in edges:
        src = positions.get(edge["source"])
        dst = positions.get(edge["target"])
        if not src or not dst:
            continue

        start = ((src[0] + src[2]) // 2, src[3])
        end = ((dst[0] + dst[2]) // 2, dst[1])

        draw_arrow(draw, start, end, fill=colors["outline"], width=4)

        label = edge.get("label", "").strip()
        if label:
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            draw.rounded_rectangle(
                (mid_x - 60, mid_y - 16, mid_x + 60, mid_y + 16),
                radius=8,
                fill=colors["fill"],
                outline=colors["outline"],
            )
            lbbox = draw.textbbox((0, 0), label, font=caption_font)
            lw = lbbox[2] - lbbox[0]
            lh = lbbox[3] - lbbox[1]
            draw.text((mid_x - lw // 2, mid_y - lh // 2), label, fill="black", font=caption_font)


    for node in nodes:
        x1, y1, x2, y2 = positions[node["id"]]
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=colors["fill"], outline=colors["outline"], width=3)

        label = wrap_label(node["label"], width=32)
        tbbox = draw.multiline_textbbox((0, 0), label, font=node_font, spacing=6, align="center")
        tw = tbbox[2] - tbbox[0]
        th = tbbox[3] - tbbox[1]
        tx = x1 + (box_w - tw) / 2
        ty = y1 + (box_h - th) / 2 - 4
        draw.multiline_text((tx, ty), label, fill="black", font=node_font, spacing=6, align="center")


    caption = spec.get("caption", "")
    cbbox = draw.textbbox((0, 0), caption, font=caption_font)
    cw = cbbox[2] - cbbox[0]
    cx = (width - cw) // 2
    draw.text((cx, height - 48), caption, fill="black", font=caption_font)

    img.save(out_path)




def generate_ai_image(chapter_title: str, chapter_num: int, style: str = "technical") -> str | None:
    """Generate an illustrative image using DALL-E."""
    from openai import OpenAI
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("   OPENAI_API_KEY not set, skipping AI image")
        return None
    
    style_descriptions = {
        "architecture": "technical diagram of neural network architecture, clean lines, blue and white, textbook style",
        "process": "flowchart of machine learning training, green and white, clean technical illustration",
        "flow": "data flow diagram, orange and white, technical textbook style",
        "cycle": "feedback loop diagram, pink and white, clean technical illustration",
        "comparison": "comparison chart, purple and white, technical textbook style",
        "flowchart": "technical flowchart, gray and white, clean diagram",
    }
    
    description = style_descriptions.get(style, "technical diagram for educational textbook")
    
    prompt = f"""Technical schematic diagram for a Large Language Models textbook.

Chapter: {chapter_title}
Visual concept: {description}

Visual specification:
- Pure white background (#FFFFFF), no shadows, no gradients, no texture
- Flat 2D line art only — no perspective, no 3D shading, no depth effects
- Geometric shapes: rectangles, arrows, circles — no decorative illustration
- Monochrome line strokes in dark gray (#1A1A1A), 1.5–2pt weight
- Accent color: single muted blue (#4A7FB5) for highlighted components only
- Generous whitespace between elements — never crowded or dense
- All arrows directional, orthogonal (90° bends only), clean terminations
- No text, labels, or annotations anywhere in the image
- No borders, frames, drop shadows, or outer containers
- Aspect ratio 4:3, centered composition with 10% margin on all sides
- Print-ready quality, 300 DPI equivalent detail level
- Style reference: MIT OpenCourseWare handout diagrams, not UI mockups or infographics
""".strip()
    
    try:
        client_openai = OpenAI(api_key=openai_key)
        
        response = client_openai.images.generate(
            model=IMAGE_GEN_MODEL,
            prompt=prompt,
            size=IMAGE_GEN_SIZE,
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        

        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        
        file_base = f"chapter_{chapter_num:02d}_ai_{slugify(chapter_title)}.png"
        out_path = DIAGRAM_DIR / file_base
        
        with open(out_path, "wb") as f:
            f.write(img_response.content)
        
        print(f"   AI image saved: {out_path.name}")
        return f"diagrams/{file_base}"
        
    except Exception as e:
        print(f"   AI image failed: {e}")
        return None


def diagram_block(rel_path: str, caption: str, marker_id: str) -> str:
    return (
        f"<!-- DIAGRAM_START:{marker_id} -->\n"
        f"![{caption}]({rel_path})\n\n"
        f"*{caption}*\n"
        f"<!-- DIAGRAM_END:{marker_id} -->"
    )


def upsert_diagram_in_chapter(chapter_num: int, rel_path: str, caption: str) -> None:
    path = BOOK_DIR / f"chapter_{chapter_num:02d}.md"
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    marker_id = f"chapter_{chapter_num:02d}"

    
    existing_pattern = re.compile(
        rf"<!-- DIAGRAM_START:{re.escape(marker_id)} -->.*?<!-- DIAGRAM_END:{re.escape(marker_id)} -->\n?",
        re.DOTALL,
    )
    content = re.sub(existing_pattern, "", content).strip()

    block = diagram_block(rel_path, caption, marker_id)

    lines = content.split("\n")

    insert_idx = None
    paragraph_count = 0

    for i, line in enumerate(lines):
    
        if line.strip() == "" or line.startswith("#"):
            continue
        
        paragraph_count += 1

    
        if paragraph_count == 2:
            insert_idx = i
            break

    if insert_idx is None:
    
        new_content = content + "\n\n" + block
    else:
        new_lines = lines[:insert_idx] + ["", block, ""] + lines[insert_idx:]
        new_content = "\n".join(new_lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content.strip() + "\n")


def main():
    outline = load_outline()
    chapters = outline.get("chapters", [])

    print(f" Generating diagrams for {len(chapters)} chapters...\n")

    for chapter in chapters:
        chapter_num = int(chapter["chapter_number"])
        chapter_title = chapter["chapter_title"]
        sections = chapter.get("sections", [])

        print(f"→ Chapter {chapter_num}: {chapter_title}")
        
    
        try:
            chapter_content = load_chapter_markdown(chapter_num)
        except Exception:
            chapter_content = ""

        spec = normalize_spec(generate_spec(chapter, chapter_content), chapter)
        style = spec.get("style", "flowchart")

    
        file_base = f"chapter_{chapter_num:02d}_{slugify(chapter_title)}.png"
        out_path = DIAGRAM_DIR / file_base

        render_png(spec, out_path)

    
        rel_path = f"diagrams/{file_base}"
        upsert_diagram_in_chapter(chapter_num, rel_path, spec["caption"])

        print(f"   Saved {out_path.name}")
        
    
        ai_rel_path = generate_ai_image(chapter_title, chapter_num, style)
        if ai_rel_path:
            ai_caption = f"Figure: {chapter_title} - Conceptual Illustration"
            upsert_diagram_in_chapter(chapter_num, ai_rel_path, ai_caption)
        
        print(f"   Inserted into chapter_{chapter_num:02d}.md\n")

    print(" Diagram generation complete")
    print(" Next: run 14_build_pdf.py")


if __name__ == "__main__":
    main()