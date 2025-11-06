# app.py
import os
import uuid
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.document_loaders import CSVLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import httpx
client = httpx.Client(verify=False)
# Source - https://stackoverflow.com/questions/76106366/how-to-use-tiktoken-in-offline-mode-computer
# Posted by VarBird
# Retrieved 2025-11-06, License - CC BY-SA 4.0


tiktoken_cache_dir = r"C:\Demo\tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# validate
assert os.path.exists(os.path.join(tiktoken_cache_dir,"...................."))


# --------------------------
# Load environment variables
# --------------------------
#load_dotenv()
#groq_api_key = os.getenv("GROQ_API_KEY", "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")

# --------------------------
# Streamlit caching functions
# --------------------------
"""@st.cache_resource
def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)"""


embedding_model = OpenAIEmbeddings(
 base_url="https://genailab.tcs.in",
 model="azure/genailab-maas-text-embedding-3-large",
 api_key=".........",
 http_client=client)

llm = ChatOpenAI(
 base_url="https://genailab.tcs.in",
 model="azure_ai/genailab-maas-DeepSeek-V3-0324",
 api_key="..........",
 http_client=client
)


@st.cache_data
def generate_embeddings(texts, _model):
    # OpenAIEmbeddings provides embed_documents()
    return _model.embed_documents(texts)



# --------------------------
# Load CSV
# --------------------------
loader = CSVLoader(file_path="C:/Demo/data/insurance_faq.csv")
documents = loader.load()
texts = [doc.page_content for doc in documents]

# --------------------------
# Initialize Embedding Model & Vector Store
# --------------------------
#embedding_model = load_embedding_model()
embeddings_file = "faq_embeddings.npy"

if os.path.exists(embeddings_file):
    embeddings = np.load(embeddings_file)
else:
    embeddings = generate_embeddings(texts, embedding_model)
    np.save(embeddings_file, embeddings)



##-----------------vector store--------------
@st.cache_resource
def initialize_vector_store(documents, embeddings, persist_dir="vector_store"):
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="insurance_faq_collection")

    # Only add documents if collection is empty
    if collection.count() == 0:
        ids, docs_texts, embeddings_list, metadatas = [], [], [], []
        for doc_obj, embedding in zip(documents, embeddings):
            unique_id = str(uuid.uuid4())
            ids.append(unique_id)
            docs_texts.append(doc_obj.page_content)
            # OpenAIEmbeddings already returns a list, no need to convert
            embeddings_list.append(embedding)
            metadatas.append(doc_obj.metadata)

        collection.add(ids=ids, documents=docs_texts, embeddings=embeddings_list, metadatas=metadatas)
    return collection

vector_collection = initialize_vector_store(documents, embeddings)

# --------------------------
# RAG Retriever
# --------------------------
class RAGRetriever:
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedding_model.embed_query(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            distances = results['distances'][0]
            for doc, distance in zip(documents, distances):
                retrieved_docs.append(doc)
        return retrieved_docs


rag_retriever = RAGRetriever(vector_collection, embedding_model)

# --------------------------
# ChatGroq LLM
# --------------------------

# --------------------------
# RAG Query Function
# --------------------------
def rag_simple(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join(results) if results else ""
    if not context:
        return "No relevant documents found."
    prompt = f"""Use the following context to answer the question:

Context:
{context}

Question: {query}

Answer:"""
    response = llm.invoke(prompt)
    return response.content

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Insurance FAQ RAG Assistant", page_icon="💡")
st.title("💡 Insurance Assistant")

top_k = st.sidebar.slider("Top K retrieved documents", 1, 10, 3)
query = st.text_input("Ask your insurance question:")

if query:
    with st.spinner("Retrieving answer..."):
        answer = rag_simple(query, rag_retriever, llm, top_k=top_k)
        st.subheader("Answer")
        st.write(answer)  # ONLY prints the answer
