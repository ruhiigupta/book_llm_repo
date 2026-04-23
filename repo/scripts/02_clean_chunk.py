import os
import json
import re
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")

os.makedirs(CLEAN_DIR, exist_ok=True)



def clean_text(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\b(music|applause|laughs)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(um|uh|you know|like)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def chunk_text(text, chunk_size=200, overlap=50):
    
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = []

    for sentence in sentences:
        words = sentence.split()


        while len(words) > chunk_size:
            chunks.append(" ".join(words[:chunk_size]))
            words = words[chunk_size - overlap:]

        sentence = " ".join(words)
        current_len = sum(len(s.split()) for s in current)

        if current_len + len(words) <= chunk_size:
            current.append(sentence)
        else:
            if current:
                chunk = " ".join(current).strip()
                chunks.append(chunk)

        
                overlap_words = chunk.split()[-overlap:]
                current = [" ".join(overlap_words), sentence]
            else:
                chunks.append(sentence)
                current = []

    if current:
        chunks.append(" ".join(current).strip())


    chunk_objects = [
        {"chunk_id": i, "text": c}
        for i, c in enumerate(chunks)
    ]

    return chunk_objects



def extract_text(data):
    transcript = data.get("transcript", "")

    if isinstance(transcript, list):
        return " ".join([x.get("text", "") for x in transcript])
    elif isinstance(transcript, str):
        return transcript
    else:
        return ""



def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transcript = data.get("transcript", "")

    if isinstance(transcript, list):
        texts = [x.get("text", "") for x in transcript]
        num_segments = len(texts)
    else:
        texts = [transcript]
        num_segments = 1

    full_text = " ".join(texts)
    cleaned = clean_text(full_text)


    chunks = chunk_text(cleaned, chunk_size=200, overlap=50)

    return {
        "title": data.get("title", ""),
        "video_id": data.get("video_id", ""),
        "video_index": data.get("video_index", ""),
        "num_chunks": len(chunks),
        "num_segments": num_segments,
        "chunks": chunks
    }



def main():
    files = sorted(
        [f for f in os.listdir(RAW_DIR)
        if f.startswith("video_") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    print(f" Processing {len(files)} files...")

    total_chunks = 0
    total_words = 0

    for file in tqdm(files, desc="Cleaning + Chunking"):
        path = os.path.join(RAW_DIR, file)
        processed = process_file(path)

        if processed["num_chunks"] == 0:
            print(f"Warning: No chunks generated for {file}")

        if processed["num_chunks"] < 3:
            print(f" Very low chunk count for {file}")

        total_chunks += processed["num_chunks"]

        for c in processed["chunks"]:
            total_words += len(c["text"].split())

        out_path = os.path.join(CLEAN_DIR, file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)

    avg_chunk_len = total_words / total_chunks if total_chunks > 0 else 0

    print(f"\n Done — {len(files)} files saved to data/clean/")
    print(f" Total chunks: {total_chunks}")
    print(f" Avg chunks/video: {total_chunks / len(files):.2f}")
    print(f"Avg chunk length: {round(avg_chunk_len)} words") 


if __name__ == "__main__":
    main()