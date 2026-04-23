import os
import json
import hashlib
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
DB_DIR = os.path.join(BASE_DIR, "data", "vectordb")

os.makedirs(DB_DIR, exist_ok=True)


MODEL_NAME = "all-MiniLM-L6-v2"
print(" Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print(" Model loaded")


def stable_id(text, prefix):
    """
    Create a stable, deterministic ID from text.
    Prevents duplication across reruns.
    """
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{h}"



def load_all_chunks():
    files = sorted(
        [f for f in os.listdir(CLEAN_DIR)
        if f.startswith("video_") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    all_chunks = []
    seen_texts = set()

    for filename in files:
        with open(os.path.join(CLEAN_DIR, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, chunk in enumerate(data["chunks"]):
            text = chunk["text"] if isinstance(chunk, dict) else chunk
            text = text.strip()

            if not text:
                continue

        
            if text in seen_texts:
                continue
            seen_texts.add(text)

            prefix = f"{data['video_index']:02d}_{idx:03d}"

            all_chunks.append({
                "id": stable_id(text, prefix),
                "text": text,
                "title": data.get("title", ""),
                "video_id": data.get("video_id", ""),
                "video_index": data.get("video_index", 0),
            })

    print(f" Loaded {len(all_chunks)} unique chunks from {len(files)} videos")


    avg_len = sum(len(c["text"].split()) for c in all_chunks) / len(all_chunks)
    print(f" Avg chunk length: {round(avg_len)} words")

    return all_chunks



def get_embeddings(texts):
    return model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True  
    ).tolist()



def build_or_load_vector_store(chunks):
    chroma = chromadb.PersistentClient(path=DB_DIR)

    try:
        collection = chroma.get_collection("transcripts")
        if collection.count() > 0:
            print(" Using cached embeddings")
            print(f" Existing vectors: {collection.count()}")
            return collection
    except:
        pass

    print(" Building new vector store...")

    try:
        chroma.delete_collection("transcripts")
    except:
        pass

    collection = chroma.create_collection(
        name="transcripts",
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 64

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
        batch = chunks[i:i + batch_size]

        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]

        metadatas = [{
            "title": c["title"],
            "video_id": c["video_id"],
            "video_index": c["video_index"]
        } for c in batch]

        embeddings = get_embeddings(texts)

        collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )

    print(f" Stored {len(chunks)} embeddings")


    log = {
        "total_chunks": len(chunks),
        "model": MODEL_NAME,
        "embedding_type": "local_sentence_transformer",
        "similarity": "cosine",
        "batch_size": batch_size
    }

    with open(os.path.join(DB_DIR, "build_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(" Build log saved")

    return collection



def test_retrieval(collection):
    print("\n Testing retrieval quality...")
    print("=" * 60)

    queries = [
        "What is a transformer and how does attention work?",
        "How does tokenization work in LLMs?",
        "What is gradient descent and backpropagation?"
    ]

    for query in queries:
        query_embedding = get_embeddings([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )

        print(f"\n Query: {query}")

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            score = round(1 - dist, 3)

            print(f"  Result {i+1} | Score: {score} | From: {meta['title']}")
            print(f"  Preview: {doc[:120]}...")

    print("\n" + "=" * 60)



if __name__ == "__main__":
    chunks = load_all_chunks()

    if not chunks:
        raise ValueError(" No chunks found. Run preprocessing first.")

    collection = build_or_load_vector_store(chunks)

    test_retrieval(collection)

    print("\n Step 3 Complete — Vector store ready!")