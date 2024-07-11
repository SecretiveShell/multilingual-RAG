from ollama import Client
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import langdetect

OLLAMA_URL = 'http://localhost:11434/api'
LLM_MODEL = 'llama3:8b'
EMBED_MODEL = 'nomic-embed-text'

client = chromadb.PersistentClient()
embed = embedding_functions.OllamaEmbeddingFunction(
    url=f"{OLLAMA_URL}/embeddings",
    model_name=EMBED_MODEL
)
collection = client.get_or_create_collection('pdf-data', embedding_function=embed)

ollama = Client(host=OLLAMA_URL)

while True:
        question = input('Enter your question: ')
        
        # make the question english
        response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'user',
                'content': f'translate this question using clear and concise english:\n\n\n{question}',
            },
        ])
        transformed_question = response['message']['content']

        # perform the dummy search
        response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'user',
                'content': f'create a short answer for this question:\n\n\n{transformed_question}',
            },
        ])
        query = response['message']['content']

        document = collection.query(query_texts=query)['documents'][0][0]

        # answer the user with the result
        response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'user',
                'content': f'Using this context:\n\n\n{document}\n\n\nAnswer this question in clear concise english. Do not say anything other then the response:\n\n\n{question}',
            },
        ])
        final_response = response['message']['content']


        # translate to correct language
        language = langdetect.detect(question)
        language = "ukranian" if language == "uk" else "english"
        response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'user',
                'content': f'Translate this text to {language}. Do not include extra notes or content:\n\n\n{final_response}',
            },
        ])
        final_response = response['message']['content']

        print(final_response)