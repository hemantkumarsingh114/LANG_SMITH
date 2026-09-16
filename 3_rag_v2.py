import os
from dotenv import load_dotenv
from langsmith import traceable
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

# Set Project Name:
os.environ['LANGCHAIN_PROJECT'] = 'RAG ChatBot'

# 1) Load PDF
PDF_PATH = "semantic_topics.pdf"

loader = PyMuPDFLoader(PDF_PATH)
docs = loader.load()

# 2) ---------- traced setup steps ----------
@traceable(name="load_pdf", tags=['pdf', 'loader'], metadata={'loader': 'PyMuPDFLoader'})
def load_pdf(path: str):
    loader = PyMuPDFLoader(path)
    return loader.load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore", tags=['embedding', 'vectorstore'], metadata={'embedding-model': 'gemini-embedding-001'})
def build_vectorstore(splits):
    emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    vs = FAISS.from_documents(splits, emb)
    return vs

# 3) You can also trace a “setup” umbrella span if you want:
@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs

# 4) ---------- pipeline ----------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

r'''llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
    temperature=0.5)
    
llm = ChatHuggingFace(llm=llm_endpoint)'''

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# 5) Build the index under traced setup
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) ---------- run a query (also traced) ----------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

# 7) Give the visible run name + tags/metadata so it’s easy to find:
config = {
    "run_name": "pdf_rag_query"
}

ans = chain.invoke(q, config=config)
print("\nA:", ans)