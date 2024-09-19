import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define the path where files will be temporarily stored before processing
TEMP_FILE_DIR = 'temp_files/'
os.makedirs(TEMP_FILE_DIR, exist_ok=True)

@csrf_exempt
def list_uploaded_files(request):
    if request.method == 'GET':
        try:
            if os.path.exists(TEMP_FILE_DIR):
                filenames = os.listdir(TEMP_FILE_DIR)
                files_only = [f for f in filenames if os.path.isfile(os.path.join(TEMP_FILE_DIR, f))]

                return JsonResponse({'files': files_only}, status=200)
            else:
                return JsonResponse({'files': []}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file_data')

        if not uploaded_file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        file_path = os.path.join(TEMP_FILE_DIR, uploaded_file.name)
        file_name = default_storage.save(file_path, uploaded_file)
        
        try:

            document_loader = PyPDFLoader(file_path)
            documents = document_loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splitted_documents = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            vectorstore = Chroma.from_documents(
                documents=splitted_documents,
                embedding=embeddings,
                persist_directory="chroma_db"
            )
            vectorstore.persist()
            return JsonResponse({'message': 'File processed and stored successfully in ChromaDB'}, status=200)

        except Exception as e:
            # Handle exceptions (e.g., file format issues, storage issues)
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def _retrieve_content(query: str) -> str:
    """Search Chroma DB for relevant document information based on a query."""
    retrieved_content = ""
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="chroma_db",
                          embedding_function=embeddings)
    
    retriever = vectorstore.similarity_search(query, k=5)
    for n, doc in enumerate(retriever):
        text = doc.page_content
        # metadata = str(doc.metadata)
        content = f"{text}\n"
        retrieved_content += content

    return retrieved_content

@csrf_exempt
def retrieve_content(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        query = data["query"]
        answer = _retrieve_content(query)

        return JsonResponse({'retrieved_content': answer}, status=200)
    