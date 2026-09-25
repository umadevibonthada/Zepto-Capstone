from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


DOCS_DIR = Path("docs")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zepto_policies"


def load_documents():
    documents = []
    ids = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        documents.append(text)
        ids.append(file_path.stem)

    return ids, documents


def create_vector_store():
    ids, documents = load_documents()

    print("Documents loaded:", len(documents))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print("ChromaDB collection:", COLLECTION_NAME)
    print("Documents stored:", collection.count())


if __name__ == "__main__":
    create_vector_store()