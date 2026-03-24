"""
Topic modeling untuk dataset parkir liar menggunakan LDA.

Kebutuhan library:
- pandas
- scikit-learn
- nltk

Opsional:
- Sastrawi (untuk stemming Bahasa Indonesia)
- pyLDAvis (untuk visualisasi topik)

Cara pakai:
    python lda_topic_modeling_parkir_liar.py --csv data_parkir_liar.csv
"""

from __future__ import annotations

import argparse
import re
import string
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import nltk


def ensure_nltk_resources() -> None:
    """Pastikan resource NLTK yang dibutuhkan tersedia."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name)


def build_stemmer():
    """Coba inisialisasi stemmer Bahasa Indonesia (opsional)."""
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

        return StemmerFactory().create_stemmer()
    except Exception:
        return None


def preprocess_text(text: str, stop_words: set[str], stemmer=None) -> str:
    """
    Preprocessing teks:
    1) lowercase
    2) hapus tanda baca
    3) tokenize
    4) hapus stopwords Indonesia
    5) stemming (jika stemmer tersedia)
    """
    # 1) Lowercase
    text = text.lower()

    # 2) Hapus tanda baca dan karakter non-huruf/angka
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    # 3) Tokenisasi
    tokens = word_tokenize(text)

    # 4) Hapus stopwords dan token non-alfabet
    tokens = [tok for tok in tokens if tok.isalpha() and tok not in stop_words]

    # 5) Stemming (opsional)
    if stemmer is not None:
        tokens = [stemmer.stem(tok) for tok in tokens]

    # Gabungkan kembali token agar bisa diproses CountVectorizer
    return " ".join(tokens)


def print_top_words(
    lda_model: LatentDirichletAllocation,
    feature_names,
    n_top_words: int = 10,
) -> None:
    """Cetak 10 kata teratas untuk setiap topik."""
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"Topik {topic_idx + 1}: {', '.join(top_words)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Topic modeling LDA untuk dataset parkir liar"
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help='Path ke file CSV (harus memiliki kolom "text")',
    )
    parser.add_argument(
        "--topics",
        default=5,
        type=int,
        help="Jumlah topik LDA (default: 5)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Buat visualisasi pyLDAvis (jika library tersedia)",
    )
    args = parser.parse_args()

    # Pastikan resource NLTK tersedia
    ensure_nltk_resources()

    # Muat data CSV
    if not args.csv.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {args.csv}")

    df = pd.read_csv(args.csv)

    # Validasi kolom yang dibutuhkan
    if "text" not in df.columns:
        raise ValueError('CSV harus memiliki kolom bernama "text"')

    # Ubah nilai kosong menjadi string kosong agar preprocessing aman
    df["text"] = df["text"].fillna("").astype(str)

    # Ambil stopwords bahasa Indonesia dari NLTK
    stop_words = set(stopwords.words("indonesian"))

    # Coba aktifkan stemming (opsional)
    stemmer = build_stemmer()
    if stemmer is None:
        print("[INFO] Stemmer tidak tersedia. Lanjut tanpa stemming.")

    # Preprocess seluruh dokumen
    df["clean_text"] = df["text"].apply(
        lambda x: preprocess_text(x, stop_words=stop_words, stemmer=stemmer)
    )

    # Ubah teks menjadi document-term matrix (DTM) dengan CountVectorizer
    vectorizer = CountVectorizer(min_df=2)
    dtm = vectorizer.fit_transform(df["clean_text"])

    # Latih model LDA dengan 5 topik (default)
    lda = LatentDirichletAllocation(
        n_components=args.topics,
        random_state=42,
        learning_method="batch",
    )
    lda.fit(dtm)

    # Cetak 10 kata teratas tiap topik
    print("\n=== Top 10 kata untuk setiap topik ===")
    feature_names = vectorizer.get_feature_names_out()
    print_top_words(lda, feature_names, n_top_words=10)

    # Opsional: visualisasi pyLDAvis
    if args.visualize:
        try:
            import pyLDAvis
            import pyLDAvis.lda_model

            vis = pyLDAvis.lda_model.prepare(lda, dtm, vectorizer)
            output_html = "lda_parkir_liar_viz.html"
            pyLDAvis.save_html(vis, output_html)
            print(f"\n[INFO] Visualisasi disimpan ke: {output_html}")
        except Exception as exc:
            print(f"[INFO] Visualisasi dilewati (pyLDAvis tidak tersedia): {exc}")


if __name__ == "__main__":
    main()
