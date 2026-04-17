from __future__ import annotations

import io
import re
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from pydantic import BaseModel

app = FastAPI(title="Topic Modeling API", version="1.0.0")


class TopicResult(BaseModel):
    topic_id: int
    top_words: List[str]


class TopicModelResponse(BaseModel):
    num_documents: int
    num_topics: int
    passes: int
    topics: List[TopicResult]


TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")


def preprocess_text(text: str) -> List[str]:
    text = str(text).lower()
    tokens = TOKEN_PATTERN.findall(text)
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


@app.get("/")
def root() -> dict:
    return {"message": "FastAPI Topic Modeling service is running."}


@app.post("/topic-modeling", response_model=TopicModelResponse)
async def topic_modeling(
    file: UploadFile = File(..., description="CSV file with a text column"),
    text_column: Optional[str] = Form(None, description="Name of the text column in CSV"),
    num_topics: int = Form(5, description="Number of topics for LDA"),
    num_words: int = Form(10, description="Top words per topic"),
    passes: int = Form(10, description="Training passes for LDA"),
) -> TopicModelResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    if num_topics < 1 or num_words < 1 or passes < 1:
        raise HTTPException(
            status_code=400,
            detail="num_topics, num_words, and passes must be >= 1.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows.")

    if text_column is None:
        object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if not object_columns:
            raise HTTPException(
                status_code=400,
                detail="No text column found. Please provide text_column.",
            )
        text_column = object_columns[0]

    if text_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{text_column}' not found in CSV.",
        )

    documents = (
        df[text_column]
        .dropna()
        .astype(str)
        .map(preprocess_text)
        .tolist()
    )

    documents = [doc for doc in documents if doc]
    if not documents:
        raise HTTPException(
            status_code=400,
            detail="No valid tokens after preprocessing. Check your text data.",
        )

    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    if all(len(doc_bow) == 0 for doc_bow in corpus):
        raise HTTPException(
            status_code=400,
            detail="No terms left for topic modeling after preprocessing.",
        )

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=42,
    )

    topics: List[TopicResult] = []
    for topic_id, word_probs in lda_model.show_topics(
        num_topics=num_topics,
        num_words=num_words,
        formatted=False,
    ):
        top_words = [word for word, _ in word_probs]
        topics.append(TopicResult(topic_id=topic_id, top_words=top_words))

    return TopicModelResponse(
        num_documents=len(documents),
        num_topics=num_topics,
        passes=passes,
        topics=topics,
    )
