from __future__ import annotations

import io
import re
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from pydantic import BaseModel, Field

app = FastAPI(title="Topic Modeling API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UploadResponse(BaseModel):
    upload_id: str
    num_documents: int
    message: str


class ProcessRequest(BaseModel):
    upload_id: str = Field(..., description="ID dari endpoint /upload")
    num_topics: int = Field(5, ge=1)
    num_words: int = Field(10, ge=1)
    passes: int = Field(10, ge=1)


class TopicTerm(BaseModel):
    word: str
    weight: float


class TopicResult(BaseModel):
    topic_id: int
    top_words: List[str]
    top_terms: List[TopicTerm]


class TopicModelResponse(BaseModel):
    upload_id: str
    num_documents: int
    num_topics: int
    num_words: int
    passes: int
    topics: List[TopicResult]


TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")
UPLOAD_STORE: Dict[str, List[List[str]]] = {}


def preprocess_text(text: str) -> List[str]:
    text = str(text).lower()
    tokens = TOKEN_PATTERN.findall(text)
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


@app.get("/")
def root() -> dict:
    return {"message": "FastAPI Topic Modeling service is running."}


@app.post("/upload", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(..., description="CSV file with a text column"),
    text_column: Optional[str] = Form(None, description="Name of the text column in CSV"),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File harus berformat CSV.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File upload kosong.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV tidak valid: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV tidak memiliki baris data.")

    if text_column is None:
        object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if not object_columns:
            raise HTTPException(
                status_code=400,
                detail="Kolom teks tidak ditemukan. Kirim text_column.",
            )
        text_column = object_columns[0]

    if text_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Kolom '{text_column}' tidak ditemukan pada CSV.",
        )

    documents = df[text_column].dropna().astype(str).map(preprocess_text).tolist()
    documents = [doc for doc in documents if doc]

    if not documents:
        raise HTTPException(
            status_code=400,
            detail="Tidak ada token valid setelah preprocessing.",
        )

    upload_id = str(uuid4())
    UPLOAD_STORE[upload_id] = documents

    return UploadResponse(
        upload_id=upload_id,
        num_documents=len(documents),
        message="Upload berhasil. Lanjutkan ke endpoint /process.",
    )


@app.post("/process", response_model=TopicModelResponse)
async def process_lda(payload: ProcessRequest) -> TopicModelResponse:
    documents = UPLOAD_STORE.get(payload.upload_id)
    if documents is None:
        raise HTTPException(
            status_code=404,
            detail="upload_id tidak ditemukan. Upload ulang file terlebih dahulu.",
        )

    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    if not corpus or all(len(doc_bow) == 0 for doc_bow in corpus):
        raise HTTPException(
            status_code=400,
            detail="Tidak ada term untuk diproses oleh LDA.",
        )

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=payload.num_topics,
        passes=payload.passes,
        random_state=42,
    )

    topics: List[TopicResult] = []
    for topic_id, word_probs in lda_model.show_topics(
        num_topics=payload.num_topics,
        num_words=payload.num_words,
        formatted=False,
    ):
        top_terms = [TopicTerm(word=word, weight=float(weight)) for word, weight in word_probs]
        topics.append(
            TopicResult(
                topic_id=topic_id,
                top_words=[term.word for term in top_terms],
                top_terms=top_terms,
            )
        )

    return TopicModelResponse(
        upload_id=payload.upload_id,
        num_documents=len(documents),
        num_topics=payload.num_topics,
        num_words=payload.num_words,
        passes=payload.passes,
        topics=topics,
    )
