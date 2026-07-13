"""
End-to-end Phase 2 pipeline: ingest sample documents, chunk them, and
index them into Qdrant.

Run with: python scripts/run_ingestion_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking.chunker import chunk_document
from app.chunking.indexer import index_chunks
from app.core.models import AccessLevel
from app.ingestion.pipeline import ingest_folder

ACCESS_LEVELS = {
    "public_handbook.md": AccessLevel.PUBLIC,
    "internal_engineering.html": AccessLevel.INTERNAL,
    "confidential_finance.pdf": AccessLevel.CONFIDENTIAL,
}


def main() -> None:
    print("Ingesting documents from data/ ...")
    documents = ingest_folder("data", ACCESS_LEVELS)
    print(f"  -> Loaded {len(documents)} documents")

    total_chunks = 0
    for document in documents:
        chunks = chunk_document(document)
        print(f"  -> '{document.title}' ({document.access_level.value}): {len(chunks)} chunks")

        indexed_count = index_chunks(chunks)
        total_chunks += indexed_count

    print(f"\nDone. Indexed {total_chunks} chunks total into Qdrant.")


if __name__ == "__main__":
    main()