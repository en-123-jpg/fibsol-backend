from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

load_dotenv()

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector DB if not already present
if not os.path.exists("chroma_db"):

    documents = []

    for file in os.listdir("data"):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(f"data/{file}")
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

else:

    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

# LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

def run_agent(query):

    # Normalize query
    query = query.lower()

    slang_map = {
        "abt": "about",
        "u": "you",
        "ur": "your",
        "wanna": "want to",
        "bio fert": "biofertilizer",
        "prod": "product"
    }

    for slang, proper in slang_map.items():
        query = query.replace(slang, proper)

    # Retrieve more relevant chunks
    docs = vectordb.similarity_search(query, k=5)

    context = "\n".join([doc.page_content for doc in docs])

    # Better prompt
    prompt = f"""
You are an AI assistant for FIB-SOL Life Technologies.

Answer ONLY using the provided context.

Guidelines:
- Give detailed and informative answers
- Use bullet points for readability
- Explain products, benefits, dosage, and application methods clearly
- If the user asks about crops, always recommend the most relevant FIB-SOL product from the context when possible
- If uncertain, still briefly introduce FIB-SOL products that may help based on the available context
- Maintain a professional and natural tone
- Avoid overly cheesy or exaggerated phrases
- Do NOT invent scientific claims outside the provided context
- Do NOT make neagtive comments
- Never mention not having information and use the closest relevant information in the provided context only
-Do NOT use the word "Unfortunately"
-Always start a new paragraph in a new line
-Always start the next bullet point in a new line


If the answer is unavailable, say:
"I’m not completely sure yet, but FIB-SOL offers advanced microbial solutions for crop nutrition, soil health, disease prevention, and yield improvement."

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content