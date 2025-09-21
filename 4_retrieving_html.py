import os
import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
import glob

session = requests.Session()
headers = {
    'User-Agent': UserAgent().random,
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.google.com'
}

base_url = 'https://iiitsurat.ac.in/'
list_of_urls = [base_url]
visited = set()

# ✅ Create folders
os.makedirs("clgSite", exist_ok=True)
os.makedirs("clgText", exist_ok=True)
os.makedirs("clgPDF", exist_ok=True)

# ✅ Unique filename helper
def unique_filename(folder, base_name, ext):
    filename = f"{base_name}{ext}"
    counter = 1
    while os.path.exists(os.path.join(folder, filename)):
        filename = f"{base_name}_{counter}{ext}"
        counter += 1
    return filename

# ✅ Deduplication helper
def deduplicate(folder):
    seen = {}
    for file in glob.glob(os.path.join(folder, "*")):
        with open(file, "rb") as f:
            content = f.read()
            file_hash = hashlib.md5(content).hexdigest()
        if file_hash in seen:
            print(f"Duplicate found, deleting: {file}")
            os.remove(file)
        else:
            seen[file_hash] = file


while list_of_urls:
    url = list_of_urls.pop()
    key = (url, tuple(sorted(session.cookies.items())))
    if key in visited:
        continue
    try:
        # ✅ Handle PDFs
        if url.lower().endswith(".pdf"):
            pdf_name = url.split("/")[-1] or "file.pdf"
            pdf_base = os.path.splitext(pdf_name)[0]
            pdf_file = unique_filename("clgPDF", pdf_base, ".pdf")

            with session.get(url, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(os.path.join("clgPDF", pdf_file), "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            visited.add(url)
            continue

        # ✅ Handle HTML
        r = session.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        path = urlparse(url).path
        if not path or path == "/":
            path = "index"
        safe_title = "".join(c if c.isalnum() else "_" for c in path) or "index"

        html_file = unique_filename("clgSite", safe_title, ".html")
        txt_file = unique_filename("clgText", safe_title, ".txt")

        with open(os.path.join("clgSite", html_file), "w", encoding="utf-8") as f:
            f.write(soup.prettify())

        with open(os.path.join("clgText", txt_file), "w", encoding="utf-8") as f:
            f.write(soup.get_text())

        # ✅ Collect more links
        for link in soup.find_all('a', href=True):
            new_url = urljoin(url, link['href'])
            if urlparse(new_url).netloc == urlparse(base_url).netloc:
                if new_url not in visited:
                    list_of_urls.append(new_url)

        visited.add(key)

    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        visited.add(key)

# ✅ Run deduplication at the end
deduplicate("clgSite")
deduplicate("clgText")
deduplicate("clgPDF")
