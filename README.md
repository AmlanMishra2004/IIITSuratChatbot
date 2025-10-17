IIIT Surat Website Chatbot

This project involves building a Retrieval-Augmented Generation (RAG) chatbot for the IIIT Surat website. The chatbot retrieves answers from scraped website content and PDFs using embeddings stored in a vector database.

Overview

The project extracts textual data and PDFs from the official IIIT Surat website using a custom web scraping pipeline. The scraped content is cleaned, chunked, and converted into embeddings for retrieval. The final chatbot is built using LangChain and presented through a Streamlit interface.

Features

Automated Web Scraping:
Uses the requests library with session handling and fake headers to simulate realistic browser requests.

Recursive Crawling:
Parses and visits extracted URLs using BeautifulSoup until no new links remain.

PDF Extraction:
Downloads and processes PDFs, including those embedded via <embed> tags.

Text Cleaning:
Removes redundant header and footer text manually (cannot be reliably automated).

Vector Database Integration:
Converts cleaned and chunked text into embeddings stored in Chroma.

Chatbot Pipeline:
Uses LangChain for retrieval and response generation.

User Interface:
Built using Streamlit for an interactive chatbot experience.

Challenges Faced

Session-Dependent Pages:
Some URLs returned different content depending on the browsing session (e.g., ECE vs CSE faculty pages shared the same URL).

Incomplete Scraping:
Certain URLs were skipped depending on traversal strategy (DFS vs BFS). Combined results and deduplicated outputs were required.

PDF Detection:
Many PDFs were inside <embed> tags instead of standard <a> links.

Redundant Content:
Common headers and footers appeared across pages and required manual removal.

Irrelevant PDFs:
Several documents (e.g., MOUs) were not relevant and had to be filtered manually to speed up chatbot responses.

PDF Text Extraction:
Scanned documents required OCR; structured data like tables and charts were hard to extract.

Chunking Issues:
Related information sometimes got split across chunks during text segmentation.

Tech Stack
Component	Tool / Library
Scraping	requests, BeautifulSoup
Data Storage	Local filesystem
Embeddings	LangChain + Chroma
Vector Database	ChromaDB
Interface	Streamlit
Language	Python
Project Workflow

Create an HTTP session with requests and fake headers.

Scrape base URL and recursively extract text, URLs, and PDFs.

Handle both <a> and <embed> tags for document retrieval.

Manually clean redundant content (headers/footers).

Filter irrelevant documents.

Chunk text and generate embeddings.

Store embeddings in Chroma Vector DB.

Build chatbot using LangChain.

Deploy UI using Streamlit.

Limitations

Certain steps, such as text cleaning, filtering irrelevant PDFs, and extracting data from scanned documents, cannot be fully automated.

Quality of PDF extraction depends heavily on the document format and scan clarity.

Too many large PDFs can slow retrieval and degrade chatbot performance.

Conclusion

This project demonstrates the practical challenges of automating web scraping and information extraction for real-world data. While much of the pipeline can be scripted, manual judgment remains essential for content cleaning, relevance filtering, and complex document parsing.

Author

Amlan Mishra
