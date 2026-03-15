from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore[attr-defined]
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────── Ingestion ───────────────────────────
video_id = 'Gfr50f6ZBvo'

try:
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id, languages=['en'])
    transcript_text = " ".join(chunk.text for chunk in fetched)
except Exception as e:
    print(f"No captions available for this video: {e}")
    raise SystemExit(1)

# ─────────────────────────── Indexing ────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

raw_chunks = splitter.split_text(transcript_text)
chunks = [Document(page_content=chunk) for chunk in raw_chunks]

# embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
embeddings = OllamaEmbeddings(model='nomic-embed-text')

vector_store = FAISS.from_documents(chunks, embeddings)

# ─────────────────────────── Retriever ───────────────────────────
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}
)

# ─────────────────────────── Augmentation ────────────────────────
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


prompt = PromptTemplate(
    template="""
    Answer the question based on the following context.

    Context:
    {context}

    Question: {question}

    If the answer is not present in the context, say "I don't know".
    """,
    input_variables=['context', 'question']
)

# ─────────────────────────── Generation ──────────────────────────
parser = StrOutputParser()
# model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
model = ChatOllama(model='llama3.2')

# Parallel chain
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

final_chain = parallel_chain | prompt | model | parser

query = "What is Deepmind?"
result = final_chain.invoke(query)

print(result)