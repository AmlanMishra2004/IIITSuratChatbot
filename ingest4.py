from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm
import os, json, glob, hashlib


def extract_text_recursive(obj):
    """
    Recursively walk a JSON object and extract all text fields.
    Works for dicts, lists, and nested combinations.
    """
    texts = []
    if isinstance(obj, dict):
        for v in obj.values():
            texts.extend(extract_text_recursive(v))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text_recursive(item))
    elif isinstance(obj, str):
        if obj.strip():
            texts.append(obj.strip())
    return texts


def flatten_json_for_metadata(y, parent_key='', sep='.'):
    """
    Flattens only dicts for metadata.
    Lists are kept as-is to avoid exploding into 1000 keys.
    """
    items = []
    if isinstance(y, dict):
        for k, v in y.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_json_for_metadata(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
    else:
        items.append((parent_key, y))
    return dict(items)


def make_doc_id(file_id, index, chunk, prefix=""):
    """Generate a unique and stable doc_id using hash of content."""
    chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}{file_id}_{index}_{chunk_hash}"


def ingest(
    input_dirs=["SpecialPDFjson", "NormalPDFjson2", "text_json_folder"],
    persist_dir="./sql_chroma_db",
    batch_size=64
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    # Load/create vector store
    embedding = FastEmbedEmbeddings()
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )

    # Get existing IDs
    print("🔍 Checking existing vectors in DB...")
    existing_ids = set(vector_store._collection.get()["ids"])
    print(f"   Found {len(existing_ids)} existing vectors")

    documents, ids = [], []

    # Iterate over all .json files in given folders
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"⚠️ Skipping missing folder: {input_dir}")
            continue

        for file_path in glob.glob(os.path.join(input_dir, "*.json")):
            file_id = os.path.splitext(os.path.basename(file_path))[0]
            print(f"📂 Processing {file_id} from {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid JSON file: {file_path}")
                    continue

            # Normalize to list of objects
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                print(f"⚠️ Unexpected JSON structure in {file_path}, skipping")
                continue

            for obj_index, obj in enumerate(data):
                if not isinstance(obj, dict):
                    continue

                # Metadata-safe flattening
                flat_obj = flatten_json_for_metadata(obj)
                safe_metadata = {
                    k: v for k, v in flat_obj.items()
                    if isinstance(v, (str, int, float)) and len(str(v)) < 200
                }

                # Extract text (deep scan dicts + lists)
                text_fields = extract_text_recursive(obj)
                full_text = "\n".join(text_fields)

                if not full_text.strip():
                    continue  # skip empty objects

                for i, chunk in enumerate(splitter.split_text(full_text)):
                    doc_id = make_doc_id(file_id, obj_index * 1000 + i, chunk)

                    if doc_id in existing_ids:
                        continue  # already indexed

                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                **safe_metadata,
                                "file_id": file_id,
                                "obj_index": obj_index,
                                "chunk_id": i,
                                "source": file_id
                            },
                        )
                    )
                    ids.append(doc_id)

    print(f"📦 Prepared {len(documents)} new chunks to index")

    # Insert in batches
    for i in tqdm(range(0, len(documents), batch_size), desc="Indexing docs"):
        batch = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vector_store.add_documents(batch, ids=batch_ids)

    print("✅ Ingestion complete (resumable & updatable).")
    print(f"   Total vectors in DB now: {vector_store._collection.count()}")
    return True


if __name__ == "__main__":
    ingest()
