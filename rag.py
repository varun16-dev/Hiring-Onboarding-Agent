import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

from demo import create_llm



# STEP 1 : LOAD AND SPLIT MULTIPLE PDF FILES


def process_pdf(files: list = None):

    if files is None:
        files = [
            ("./company_handbook.pdf", "policy"),
            ("./job_description.pdf", "job_description"),
            ("./resume.pdf", "resume")
        ]

    combined_chunks = []

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    for file_path, document_type in files:

        print(f"Loading: {file_path}")

        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split document
        chunks = splitter.split_documents(documents)

        # Add metadata
        for chunk in chunks:
            chunk.metadata["document_type"] = document_type
            chunk.metadata["source"] = file_path

        combined_chunks.extend(chunks)

        print(
            f"Loaded {len(chunks)} chunks "
            f"from {file_path}"
        )

    print(f"Total chunks created: {len(combined_chunks)}")

    return combined_chunks



# STEP 2 : CREATE VECTOR EMBEDDINGS

embeddings = OllamaEmbeddings(
    model="all-minilm"
)

# STEP 3 : CREATE AND SAVE FAISS VECTOR STORE

def ingest_data():

    chunks = process_pdf()

    if not chunks:
        raise ValueError(
            "No document chunks were created. "
            "Check your PDF files."
        )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save FAISS index
    vector_store.save_local("./faiss_index")

    print("Data ingested successfully!")
    print("FAISS index saved successfully!")

    return vector_store

# LOAD EXISTING FAISS VECTOR STORE

def load_vector_store():

    vector_store = FAISS.load_local(
        "./faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("FAISS index loaded successfully!")

    return vector_store

# POLICY RETRIEVER

def get_policy_retriever(vector_store):

    return vector_store.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {
                "document_type": "policy"
            }
        }
    )

# RESUME RETRIEVER

def get_resume_retriever(vector_store):

    return vector_store.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {
                "document_type": "resume"
            }
        }
    )

# JOB DESCRIPTION RETRIEVER

def get_jd_retriever(vector_store):

    return vector_store.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {
                "document_type": "job_description"
            }
        }
    )

# RAG CHAIN

def rag_chain(query, vector_store, document_type=None):

    # Select appropriate retriever
    if document_type == "policy":

        retriever = get_policy_retriever(
            vector_store
        )

    elif document_type == "resume":

        retriever = get_resume_retriever(
            vector_store
        )

    elif document_type == "job_description":

        retriever = get_jd_retriever(
            vector_store
        )

    else:

        # Existing general behavior
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 3
            }
        )

    # Create LLM
    llm = create_llm()

    # Create RAG chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    # Run query
    return chain.invoke({
        "query": query
    })




if __name__ == "__main__":
    print("Hiring RAG System")
    
    # Create FAISS index
    vector_store = ingest_data()

    # Ask policy question
    query = input(
        "\nEnter your policy question:")

    response = rag_chain(query,vector_store,"policy")

    print("\nResponse:")
    print(response)