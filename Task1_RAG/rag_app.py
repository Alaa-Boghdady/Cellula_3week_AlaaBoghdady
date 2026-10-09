from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================================================
# 1. Load Professional Knowledge Base
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "alaa_cv.pdf"

print("\nLoading professional knowledge base...")

loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()

print(f"Pages loaded: {len(documents)}")


# =========================================================
# 2. Split Documents into Chunks
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# =========================================================
# 3. Create Embeddings
# =========================================================

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# 4. Create FAISS Vector Database
# =========================================================

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector database created successfully.")
print("RAG knowledge base is ready!")


# =========================================================
# 5. Helper Function
# =========================================================

def clean_text(text):
    """
    Clean retrieved text for easier terminal display.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# 6. Interactive Retrieval-Augmented Application
# =========================================================

while True:

    print("\n" + "=" * 60)

    query = input(
        "\nAsk a question about Alaa's professional profile "
        "(or type 'exit'): "
    ).strip()

    if query.lower() == "exit":
        print("\nRAG application closed.")
        break

    if not query:
        continue


    # =====================================================
    # 7. Semantic Retrieval
    # =====================================================

    results_with_scores = vector_store.similarity_search_with_score(
        query,
        k=min(5, len(chunks))
    )


    # =====================================================
    # 8. Display Grounded Answer
    # =====================================================

    print("\n" + "-" * 60)
    print("RETRIEVED PROFESSIONAL INFORMATION")
    print("-" * 60)

    relevant_results = []

    for document, score in results_with_scores:

        # Lower FAISS distance = more semantically similar
        if score < 1.5:
            relevant_results.append((document, score))


    if not relevant_results:

        print(
            "The provided documents do not contain enough "
            "information to answer this question."
        )

    else:

        for i, (document, score) in enumerate(
            relevant_results,
            start=1
        ):

            content = clean_text(document.page_content)

            print(f"\nRelevant Context {i}:")
            print(content)


    # =====================================================
    # 9. Display Sources
    # =====================================================

    print("\nSOURCES:")

    seen_sources = set()

    for document, score in relevant_results:

        source = Path(
            document.metadata.get("source", "Unknown")
        ).name

        page = document.metadata.get("page", 0) + 1

        source_info = f"{source}, page {page}"

        if source_info not in seen_sources:

            print(f"- {source_info}")

            seen_sources.add(source_info)

    if not relevant_results:
        print("- No sufficiently relevant source found.")