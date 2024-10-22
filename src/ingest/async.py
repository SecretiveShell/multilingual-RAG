import pdfplumber
import sys
from ollama import AsyncClient
import hashlib
import asyncio

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

file = sys.argv[1]

OLLAMA_URL = 'http://localhost:11434/api'
LLM_MODEL = 'llama3:8b'
EMBED_MODEL = 'nomic-embed-text'

client = chromadb.PersistentClient()
embed = embedding_functions.OllamaEmbeddingFunction(
    url=f"{OLLAMA_URL}/embeddings",
    model_name=EMBED_MODEL
)
collection = client.get_or_create_collection('pdf-data', embedding_function=embed)

ollama = AsyncClient(host=OLLAMA_URL)

async def parse_page(text, page_num, num_pages) :
    pageid = f"{page_num}/{num_pages}"
    print(f"[[Processing {pageid}]]")

    # translate and format the document
    response = await ollama.chat(model=LLM_MODEL, messages=[
        {
            'role': 'user',
            'content': f'Translate this document without including any other content. Ignore page numbers, navigation information, organisers and other similar information:\n\n\n{text}',
        },
    ])
    text = response['message']['content']
    text = text.split('\n', 1)[-1].strip() # remove the first line "here is your translated document:" message

    print(f"[[Translated {pageid}]]")

    # check if the text is actually useful for RAG
    response = await ollama.chat(model=LLM_MODEL, messages=[
        {
            'role': 'user',
            'content': f'Do you think this document has any useful raw knowledge? reject anything that has slideshow information only (yes/no):\n\n\n{text}',
        },
    ])
    useful = response['message']['content'].split()[0].lower()
    if not "yes" in useful: 
        print(f"[[Skipping {pageid}]]")
        return
    
    print(f"[[Adding {pageid}]]")

    hash = hashlib.sha1(text.encode('utf-32')).hexdigest()
    collection.add(ids=hash, documents=text)

    print(f"[[Added {pageid}]]")

async def main() :
    jobs = []
    with pdfplumber.open(file) as pdf:
        num_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages) :
            jobs.append(parse_page(page.extract_text(), page_num+1, num_pages))

    await asyncio.gather(*jobs)
    
    print("[[Done]]")

asyncio.run(main())