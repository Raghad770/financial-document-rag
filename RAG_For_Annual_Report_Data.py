"""
RAG_v3_single_script.py  (LangChain == 1.2.7 compatible)

What this script does:
1) Extract table-aware chunks from a PDF (text + tables) using your `extract_table_aware_chunks`.
2) Convert chunks -> LangChain Documents with metadata.
3) Split ONLY text documents (tables untouched).
4) Build Chroma vector store + MMR retriever.
5) Build BM25 retriever.
6) Custom ensemble retriever (BM25 + MMR) using Runnable API (.invoke()).
7) HyDE expansion + Query rewriting + multi-query retrieval aggregation.
8) Final answer generation from retrieved context.

Notes:
- Requires env var OPENAI_API_KEY OR pass api_key to OpenAIEmbeddings/ChatOpenAI.
- Ensure `extract_table_aware_chunks` is importable (change the import line if needed).
"""
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import umap
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import  BM25Retriever
from langchain_core.retrievers import BaseRetriever

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langsmith.wrappers import wrap_openai 
import langsmith as ls
from langsmith.run_helpers import get_current_run_tree
from pdf_table_aware import extract_table_aware_chunks
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
import hashlib

# Hybrid Search / RRF
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List
from collections import defaultdict
# ===== Your PDF chunker (change this import to your module path) =====
# Example: from ingestion.pdf_table_aware import extract_table_aware_chunks

# =========================
# CONFIG
# =========================
FILE_PATH = "./data/microsoft-annual-report.pdf"

CHROMA_DIR = "./chroma_db_lc"
CHROMA_COLLECTION = "microsoft-annual-report-lc"
RESET_CHROMA = False  # set True once if you want to wipe persisted DB

MMR_K = 8
MMR_FETCH_K = 50
MMR_LAMBDA = 0.3

BM25_K = 20
ENSEMBLE_K = 10

FINAL_CONTEXT_DOCS = 6
MAX_UNIQUE_DOCS = 30

# If you want to pass key explicitly:
# OPENAI_API_KEY = "..."
# Otherwise, set env var OPENAI_API_KEY
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# =========================
# HELPERS
# =========================
def word_wrap(text, width=100):
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x) for x in line) + len(line) + len(w) > width:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        key = (
            d.metadata.get("source"),
            d.metadata.get("page"),
            d.metadata.get("table_id"),
            d.page_content[:120],
        )
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def needs_table(query: str) -> bool:
    q = query.lower()
    return any(x in q for x in ["fy", "revenue", "operating income", "$", "million", "billion", "segment", "total"])


def debug_print_retrieved(docs: List[Document], title="RETRIEVED", max_chars=220):
    print("\n" + "=" * 90)
    print(title, "| COUNT:", len(docs))
    print("=" * 90)
    for i, d in enumerate(docs[:10]):
        print(f"\nRANK #{i}")
        print("meta:", {k: d.metadata.get(k) for k in ["type", "page", "table_id", "source"]})
        print(d.page_content[:max_chars].replace("\n", " "))
    print("\n" + "=" * 90)

def context_hash(docs: List[Document], top_n=FINAL_CONTEXT_DOCS) -> str:
    ctx = "\n\n".join(d.page_content for d in docs[:top_n])
    return hashlib.md5(ctx.encode("utf-8")).hexdigest()


# =========================
# ENSEMBLE RETRIEVER (LangChain 1.2.7)
# =========================
class BM25MMREnsembleRetriever(BaseRetriever):
    bm25: BaseRetriever
    vector: BaseRetriever
    k: int = ENSEMBLE_K

    def _dedupe(self, docs: List[Document]) -> List[Document]:
        return dedupe_docs(docs)

    # ✅ correct hook used by .invoke()
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # retrievers are Runnables -> use invoke()
        bm25_docs = self.bm25.invoke(query)
        vec_docs = self.vector.invoke(query)

        merged = self._dedupe(bm25_docs + vec_docs)

        # soft preference: tables first on numeric intent
        if needs_table(query):
            merged.sort(key=lambda d: 0 if d.metadata.get("type") == "table" else 1)

        return merged[: self.k]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        bm25_docs = await self.bm25.ainvoke(query)
        vec_docs = await self.vector.ainvoke(query)

        merged = self._dedupe(bm25_docs + vec_docs)

        if needs_table(query):
            merged.sort(key=lambda d: 0 if d.metadata.get("type") == "table" else 1)

        return merged[: self.k]


# =========================
# MAIN
# =========================
def main():
    print("RUNNING FILE:", os.path.abspath(__file__))
    print("\n--- Chunking PDF into table-aware chunks ---")
    chunks = extract_table_aware_chunks(FILE_PATH)

    # Convert chunks -> LangChain Docs
    def to_langchain_docs(chunks):
        docs = []
        for ch in chunks:
            docs.append(
                Document(
                    page_content=ch["text"],
                    metadata={
                        "type": ch["type"],        # "text" or "table"
                        "page": ch["page"],
                        "source": ch["source"],
                        "table_id": ch.get("table_id"),
                    },
                )
            )
        return docs

    documents = to_langchain_docs(chunks)

    # Split only TEXT docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200)

    def split_only_text(docs):
        out = []
        for d in docs:
            if d.metadata.get("type") == "text":
                out.extend(text_splitter.split_documents([d]))
            else:
                out.append(d)
        return out

    documents = split_only_text(documents)

    print(f"Total documents after splitting: {len(documents)}")
    print("Sample document metadata:", documents[0].metadata)
    print("Sample document preview:", documents[0].page_content[:200].replace("\n", " "))

    table_docs = [d for d in documents if d.metadata.get("type") == "table"]
    if table_docs:
        print("Sample TABLE preview:", table_docs[0].page_content[:200].replace("\n", " "))
    else:
        print("WARNING: No table docs found!")

    # Reset Chroma (optional)
    if RESET_CHROMA and os.path.exists(CHROMA_DIR):
        print(f"\n--- Resetting Chroma DB at {CHROMA_DIR} ---")
        shutil.rmtree(CHROMA_DIR)

    # Setup embeddings + vectorstore
    print("\n--- Building Chroma vectorstore ---")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    try:
        print("CHROMA COUNT:", vectorstore._collection.count())
        print("CHROMA COLLECTION:", vectorstore._collection.name)
    except Exception:
        pass

    # Vector retriever (MMR)
    retriever_vec_mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": MMR_K, "fetch_k": MMR_FETCH_K, "lambda_mult": MMR_LAMBDA},
    )

    # BM25 retriever
    retriever_bm25 = BM25Retriever.from_documents(documents)
    retriever_bm25.k = BM25_K

    # Ensemble
    ensemble_retriever = BM25MMREnsembleRetriever(
        bm25=retriever_bm25,
        vector=retriever_vec_mmr,
        k=ENSEMBLE_K,
    )

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY) if OPENAI_API_KEY else ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # HyDE
    augment_prompt = ChatPromptTemplate.from_template(
        """You are a helpful expert financial research assistant.
Provide a hypothetical example answer that might be found in a document like an annual report.

Question: {question}"""
    )
    augment_chain = augment_prompt | llm | StrOutputParser()

    # Rewrite queries
    rewrite_prompt = ChatPromptTemplate.from_template(
        """Generate 4 diverse search queries for retrieving information from a financial annual report.
Make them different: one keyword-heavy, one table-focused, one entity-focused, one paraphrase.
Return only the queries, one per line.

Question: {question}"""
    )
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # Final answer
    generation_prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the provided context.

Rules:
- If ALL parts of the question are answered by the context, provide the answer.
- If SOME parts are answered and others are not, answer what is found and clearly say which part is missing.
- If NOTHING is found, say: NOT FOUND IN DOCUMENT.
- Copy numbers and facts EXACTLY as shown in the context.
- Be concise (max 3 sentences).

Context:
{context}

Question:
{question}

Answer:"""
)

    generation_chain = generation_prompt | llm | StrOutputParser()

    # =========================
    # QUERY (change this each run)
    # =========================
    original_query = "What were the FY2023 revenues by segment (Productivity & Business Processes / Intelligent Cloud / More Personal Computing)?"

    # original_query = "What were total revenue and operating income in FY2023 vs FY2022?"

    print("\n===== PIPELINE START =====")
    print("ORIGINAL_QUERY USED:", repr(original_query))

    print("\n--- Generating HyDE Answer ---")
    expanded_answer = augment_chain.invoke({"question": original_query})
    print("HyDE (first 300):\n", word_wrap(expanded_answer[:300], width=100))

    joint_query = f"{original_query} {expanded_answer}"

    print("\n--- Rewriting Queries ---")
    rewritten_queries_str = rewrite_chain.invoke({"question": original_query})
    print("RAW REWRITE OUTPUT:\n", rewritten_queries_str)

    rewritten_queries = [q.strip() for q in rewritten_queries_str.split("\n") if q.strip()]
    print("PARSED REWRITES:", rewritten_queries)

    search_queries = [joint_query] + rewritten_queries

    print("\nSEARCH QUERIES USED (first 180 chars):")
    for q in search_queries:
        print("-", q[:180].replace("\n", " "))

    # Retrieval (multi-query aggregation)
    all_retrieved_docs = []
    seen_keys = set()

    print("\n--- Retrieving Documents ---")
    for q in search_queries:
        docs = ensemble_retriever.invoke(q)  # ✅ correct in langchain 1.2.7
        if docs:
            print("\nQUERY:", q[:90].replace("\n", " "))
            print("TOP DOC META:", docs[0].metadata)
            print("TOP DOC SNIP:", docs[0].page_content[:150].replace("\n", " "))

        for d in docs:
            key = (d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("table_id"), d.page_content[:120])
            if key not in seen_keys:
                seen_keys.add(key)
                all_retrieved_docs.append(d)

        if len(all_retrieved_docs) >= MAX_UNIQUE_DOCS:
            break

    all_retrieved_docs = dedupe_docs(all_retrieved_docs)

    if needs_table(original_query):
        all_retrieved_docs.sort(key=lambda d: 0 if d.metadata.get("type") == "table" else 1)

    debug_print_retrieved(all_retrieved_docs, title="AGGREGATED UNIQUE DOCS", max_chars=220)
    print("CONTEXT HASH:", context_hash(all_retrieved_docs, top_n=FINAL_CONTEXT_DOCS))

    relevant_docs = all_retrieved_docs[:FINAL_CONTEXT_DOCS]
    context = "\n\n".join(d.page_content for d in relevant_docs)

    print("\n--- Generating Final Answer ---")
    final_answer = generation_chain.invoke({"context": context, "question": original_query})
    print("\n================ FINAL ANSWER ================\n", final_answer)
    print("=============================================\n")


if __name__ == "__main__":
    main()
