from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm
import os, json, glob, hashlib


def extract_text_recursive(obj):
    texts = []
    if isinstance(obj, dict):
        for v in obj.values():
            texts.extend(extract_text_recursive(v))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text_recursive(item))
    elif isinstance(obj, str) and obj.strip():
        texts.append(obj.strip())
    return texts


def flatten_json_for_metadata(y, parent_key='', sep='.'):
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


def make_doc_id(file_id, index, chunk):
    chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:8]
    return f"{file_id}_{index}_{chunk_hash}"


def chunk_normal_pdf(data, file_id):
    """Custom chunking for NormalPDFjson3 - one chunk per semester/section"""
    docs, ids = [], []
    
    if isinstance(data, dict):
        for idx, (key, courses) in enumerate(data.items()):
            # Create semantic chunk for each semester/section
            content = f"{key}:\n" + "\n".join(courses) if isinstance(courses, list) else str(courses)
            
            doc_id = make_doc_id(file_id, idx, content)
            docs.append(Document(
                page_content=content,
                metadata={
                    "file_id": file_id,
                    "section": key,
                    "chunk_id": idx,
                    "source": file_id
                }
            ))
            ids.append(doc_id)
    
    return docs, ids


def chunk_default(data, file_id, splitter):
    """Default chunking for other folders"""
    docs, ids = [], []
    
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        return docs, ids
    
    for obj_index, obj in enumerate(data):
        if not isinstance(obj, dict):
            continue
        
        flat_obj = flatten_json_for_metadata(obj)
        safe_metadata = {
            k: v for k, v in flat_obj.items()
            if isinstance(v, (str, int, float)) and len(str(v)) < 200
        }
        
        text_fields = extract_text_recursive(obj)
        full_text = "\n".join(text_fields)
        
        if not full_text.strip():
            continue
        
        for i, chunk in enumerate(splitter.split_text(full_text)):
            doc_id = make_doc_id(file_id, obj_index * 1000 + i, chunk)
            docs.append(Document(
                page_content=chunk,
                metadata={
                    **safe_metadata,
                    "file_id": file_id,
                    "obj_index": obj_index,
                    "chunk_id": i,
                    "source": file_id
                }
            ))
            ids.append(doc_id)
    
    return docs, ids


def save_chunks_to_json(file_id, docs, ids, output_dir="chunkJson"):
    """Save chunks for one file to a JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    chunks_data = [{
        "id": doc_id,
        "content": doc.page_content,
        "metadata": doc.metadata
    } for doc, doc_id in zip(docs, ids)]
    
    output_path = os.path.join(output_dir, f"{file_id}_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)


def ingest(
    input_dirs=["SpecialPDFjson", "NormalPDFjson3", "text_json_folder"],
    persist_dir="./sql_chroma_db",
    batch_size=64
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=100, length_function=len, add_start_index=True
    )
    
    embedding = FastEmbedEmbeddings()
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    
    print("🔍 Checking existing vectors...")
    existing_ids = set(vector_store._collection.get()["ids"])
    print(f"   Found {len(existing_ids)} existing vectors")
    
    documents, ids = [], []
    
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"⚠️ Skipping: {input_dir}")
            continue
        
        for file_path in glob.glob(os.path.join(input_dir, "*.json")):
            file_id = os.path.splitext(os.path.basename(file_path))[0]
            print(f"📂 Processing {file_id}")
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON: {file_path}")
                continue
            
            # Use custom chunking for NormalPDFjson3
            if "NormalPDFjson3" in input_dir:
                batch_docs, batch_ids = chunk_normal_pdf(data, file_id)
            else:
                batch_docs, batch_ids = chunk_default(data, file_id, splitter)
            
            # Save chunks to JSON file
            save_chunks_to_json(file_id, batch_docs, batch_ids)
            
            # Filter out existing docs
            for doc, doc_id in zip(batch_docs, batch_ids):
                if doc_id not in existing_ids:
                    documents.append(doc)
                    ids.append(doc_id)
    
    print(f"📦 Indexing {len(documents)} new chunks")
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Indexing"):
        batch = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vector_store.add_documents(batch, ids=batch_ids)
    
    print(f"✅ Complete. Total vectors: {vector_store._collection.count()}")
    return True


if __name__ == "__main__":
    ingest()