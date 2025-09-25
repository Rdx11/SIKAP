import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

from agent.qa_agent import answer_query
from models.classifier import LABELS, fit_and_save, load_model, predict
from nlp.embedding import VectorIndex
from nlp.ingest import build_chunks
from utils.report import make_markdown_report

load_dotenv()

st.set_page_config(page_title="AI Legal/Policy Agent", layout="wide")

STORAGE = Path("storage")
INDEX_DIR = STORAGE / "index"
UPLOADS_DIR = STORAGE / "uploads"
MODEL_PATH = STORAGE / "impact_classifier.pkl"

for path in (STORAGE, INDEX_DIR, UPLOADS_DIR):
    path.mkdir(parents=True, exist_ok=True)

SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    """Return a filesystem-safe version of the uploaded filename."""
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    sanitized = SAFE_FILENAME_PATTERN.sub("_", normalized)
    sanitized = sanitized.strip("._") or "upload"
    return sanitized


@st.cache_resource(show_spinner=False)
def get_index() -> VectorIndex:
    idx = VectorIndex(dim=384, storage_dir=str(INDEX_DIR))
    try:
        idx.load()
    except Exception as err:  # pragma: no cover - Streamlit context only
        st.warning(f"Gagal memuat index tersimpan: {err}")
    return idx


def ensure_index_in_session() -> VectorIndex:
    if "vector_index" not in st.session_state or st.session_state.vector_index is None:
        st.session_state.vector_index = get_index()
    return st.session_state.vector_index


st.sidebar.header("Konfigurasi")
use_llm = st.sidebar.toggle("Use LLM (Gemini)", value=False)
top_k = st.sidebar.slider("Top-K Context", 1, 10, 5)
st.sidebar.caption("Upload PDF/DOCX -> Build Index -> Ajukan Pertanyaan.")

st.title("SIJAGA - Dashboard")
st.write("Prototype untuk analisis regulasi dan rekomendasi kebijakan berbasis dokumen.")

uploaded = st.file_uploader(
    "Unggah dokumen hukum (PDF/DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

if st.button("Build / Update Index"):
    if not uploaded:
        st.warning("Unggah minimal satu dokumen terlebih dahulu.")
    else:
        idx = ensure_index_in_session()
        total_files = len(uploaded)
        status_placeholder = st.empty()
        progress_bar = st.progress(0.0)
        meta_batch: List[Dict] = []
        texts_batch: List[str] = []
        for position, file in enumerate(uploaded, start=1):
            safe_name = sanitize_filename(file.name)
            buf_path = UPLOADS_DIR / f"uploads_{safe_name}"
            with open(buf_path, "wb") as f:
                f.write(file.getbuffer())
            status_placeholder.info(f"Memproses {file.name} ({position}/{total_files})")
            chunks = build_chunks(str(buf_path), doc_id=file.name)
            for chunk in chunks:
                meta = dict(chunk.meta)
                meta["doc_id"] = chunk.doc_id
                meta["chunk_index"] = len(meta_batch) + 1
                texts_batch.append(chunk.text)
                meta_batch.append(meta)
            progress_bar.progress(position / total_files)
        if texts_batch:
            idx.add_texts(texts_batch, meta_batch)
            idx.save()
            st.session_state.vector_index = idx
            status_placeholder.success(f"Index terbangun. Total potongan: {len(meta_batch)}")
        else:
            status_placeholder.warning("Tidak ada teks yang dapat diindeks dari dokumen yang diunggah.")
        progress_bar.empty()

st.markdown("---")

question = st.text_input(
    "Ajukan pertanyaan kebijakan / isu",
    placeholder="Contoh: Apa dasar hukum retribusi parkir di kabupaten?",
)

idx = ensure_index_in_session()

if question:
    if idx.is_empty():
        st.warning("Index kosong. Unggah dokumen lalu bangun index.")
    else:
        hits_raw = idx.search(question, k=top_k)
        if not hits_raw:
            st.info("Tidak ditemukan konteks yang relevan untuk pertanyaan tersebut.")
        else:
            hits = []
            for score, meta in hits_raw:
                record = dict(meta)
                record["score"] = score
                record.setdefault("text", meta.get("text", ""))
                hits.append(record)
            qa = answer_query(question, hits, use_llm=use_llm)
            if qa.mode == "llm_fallback":
                st.info("Mode LLM tidak tersedia. Menampilkan ringkasan konteks sebagai fallback.")
            st.subheader("Jawaban")
            st.markdown(qa.answer)

            with st.expander("Konteks (Top-K)"):
                for i, hit in enumerate(hits, start=1):
                    location_bits = []
                    if hit.get("page"):
                        location_bits.append(f"hal. {hit['page']}")
                    if hit.get("section"):
                        location_bits.append(f"paragraf {hit['section']}")
                    if hit.get("section_chunk"):
                        location_bits.append(f"bagian {hit['section_chunk']}")
                    location = f" ({', '.join(location_bits)})" if location_bits else ""
                    score_text = f" - skor {hit['score']:.3f}" if hit.get("score") is not None else ""
                    source = hit.get("doc_id") or hit.get("source", "-")
                    st.markdown(f"**{i}. {source}**{location}{score_text}")
                    st.write(hit.get("text", "")[:1000])

            st.markdown("---")
            st.subheader("Klasifikasi Dampak (Demo)")
            model = load_model(str(MODEL_PATH))
            if model is None and hits:
                example_texts = [h["text"] for h in hits]
                example_labels = [LABELS[i % len(LABELS)] for i in range(len(example_texts))]
                fit_and_save(example_texts, example_labels, str(MODEL_PATH))
                model = load_model(str(MODEL_PATH))
            if model and hits:
                pred = predict(" ".join([h["text"] for h in hits[:3]]), model)
                st.write(f"Kategori dominan: **{pred.label}** (skor: ~{pred.proba:.2f})")
            else:
                st.info("Belum ada model dampak atau konteks untuk diklasifikasikan.")

            report_markdown = make_markdown_report(question, qa.answer, hits)
            st.download_button(
                "Download Laporan (Markdown)",
                report_markdown,
                file_name="laporan_rekomendasi.md",
            )

st.caption("Prototype Magang - Pemerintah Kabupaten Sumbawa")
