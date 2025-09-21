import os
import json
import pdfplumber
import pytesseract
from PIL import Image

input_folder = "mergedPDF"
output_folder = "PDF_jsonl_folder2"
os.makedirs(output_folder, exist_ok=True)

def process_pdf(file_path):
    result = {"paragraphs": [], "tables": []}

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # 1. Extract text
            text = page.extract_text()
            if text:
                result["paragraphs"].append(text.strip())
            else:
                # OCR fallback for scanned PDFs
                img = page.to_image(resolution=300).original
                text = pytesseract.image_to_string(img, lang="eng")
                if text.strip():
                    result["paragraphs"].append(text.strip())

            # 2. Extract tables
            tables = page.extract_tables()
            for table in tables:
                result["tables"].append(table)

    return result


for filename in os.listdir(input_folder):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".pdf", ".json"))

        print(f"Processing {filename}...")
        extracted = process_pdf(file_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, ensure_ascii=False, indent=2)

print("✅ All PDFs processed into JSON!")
