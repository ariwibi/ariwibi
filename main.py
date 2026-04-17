from __future__ import annotations

import io
import re
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from pydantic import BaseModel, Field

app = FastAPI(title="Topic Modeling API", version="2.2.0")

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
    auto_topics: bool = False
    min_topics: int = Field(2, ge=2)
    max_topics: int = Field(10, ge=2)


class TopicTerm(BaseModel):
    word: str
    weight: float


class CoherenceCandidate(BaseModel):
    num_topics: int
    coherence_score: float


class AutoTopicInfo(BaseModel):
    enabled: bool
    best_num_topics: Optional[int] = None
    best_coherence_score: Optional[float] = None
    candidates: List[CoherenceCandidate] = []


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
    auto_topic_info: AutoTopicInfo
    topics: List[TopicResult]


TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")
UPLOAD_STORE: Dict[str, List[List[str]]] = {}

# Kamus slang sederhana (bisa diperluas sesuai kebutuhan)
SLANG_DICTIONARY: Dict[str, str] = {
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "tdk": "tidak",
    "bgt": "banget",
    "bgtt": "banget",
    "dr": "dari",
    "dgn": "dengan",
    "yg": "yang",
    "utk": "untuk",
    "aja": "saja",
    "sm": "sama",
    "krn": "karena",
    "trs": "terus",
    "blm": "belum",
    "udh": "sudah",
    "sdh": "sudah",
    "bkn": "bukan",
    "org": "orang",
    "dlm": "dalam",
    "jd": "jadi",
    "jgn": "jangan",
    "sy": "saya",
    "gw": "saya",
    "gue": "saya",
}


def normalize_slang(token: str) -> str:
    return SLANG_DICTIONARY.get(token, token)


def preprocess_text(text: str) -> List[str]:
    text = str(text).lower()
    tokens = TOKEN_PATTERN.findall(text)
    normalized_tokens = [normalize_slang(token) for token in tokens]
    return [token for token in normalized_tokens if token not in STOPWORDS and len(token) > 2]


def select_num_topics_by_coherence(
    corpus: List[List[tuple]],
    dictionary: corpora.Dictionary,
    texts: List[List[str]],
    min_topics: int,
    max_topics: int,
    passes: int,
) -> tuple[int, List[CoherenceCandidate], float]:
    if max_topics < min_topics:
        raise HTTPException(status_code=400, detail="max_topics harus >= min_topics.")

    if len(dictionary) < 2:
        raise HTTPException(
            status_code=400,
            detail="Kosakata terlalu sedikit untuk auto topic selection.",
        )

    upper_topics = min(max_topics, len(dictionary))
    lower_topics = min(min_topics, upper_topics)

    if lower_topics > upper_topics:
        raise HTTPException(
            status_code=400,
            detail="Rentang topic tidak valid untuk data saat ini.",
        )

    best_topic_count = lower_topics
    best_score = float("-inf")
    candidates: List[CoherenceCandidate] = []

    for topic_count in range(lower_topics, upper_topics + 1):
        lda_candidate = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=topic_count,
            passes=passes,
            random_state=42,
        )
        coherence_model = CoherenceModel(
            model=lda_candidate,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        )
        score = float(coherence_model.get_coherence())
        candidates.append(
            CoherenceCandidate(num_topics=topic_count, coherence_score=round(score, 6))
        )

        if score > best_score:
            best_score = score
            best_topic_count = topic_count

    return best_topic_count, candidates, round(best_score, 6)


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

    selected_num_topics = payload.num_topics
    auto_info = AutoTopicInfo(enabled=False, candidates=[])

    if payload.auto_topics:
        selected_num_topics, candidates, best_score = select_num_topics_by_coherence(
            corpus=corpus,
            dictionary=dictionary,
            texts=documents,
            min_topics=payload.min_topics,
            max_topics=payload.max_topics,
            passes=payload.passes,
        )
        auto_info = AutoTopicInfo(
            enabled=True,
            best_num_topics=selected_num_topics,
            best_coherence_score=best_score,
            candidates=candidates,
        )

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=selected_num_topics,
        passes=payload.passes,
        random_state=42,
    )

    topics: List[TopicResult] = []
    for topic_id, word_probs in lda_model.show_topics(
        num_topics=selected_num_topics,
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
        num_topics=selected_num_topics,
        num_words=payload.num_words,
        passes=payload.passes,
        auto_topic_info=auto_info,
        topics=topics,
    )
