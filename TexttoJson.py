import os
import json

input_folder = "cleanedText2"
output_file = "all_texts.jsonl"

def main():
    all_entries = []
    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        if not content:
            continue  # skip empty files
        
        entry = {
            "text": content,
            "metadata": {
                "doc_name": filename.replace(".txt", ""),
                "doc_type": "web_extracted"
            }
        }
        all_entries.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Conversion complete. Saved {len(all_entries)} entries to {output_file}")

if __name__ == "__main__":
    main()
