from __future__ import annotations

import datetime as dt
from typing import Dict, List


def _format_location(hit: Dict) -> str:
    parts = []
    if hit.get("page"):
        parts.append(f"hal. {hit['page']}")
    if hit.get("section"):
        parts.append(f"paragraf {hit['section']}")
    if hit.get("section_chunk"):
        parts.append(f"bagian {hit['section_chunk']}")
    return ", ".join(parts)


def make_markdown_report(query: str, answer: str, hits: List[Dict], rec: str = "") -> str:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["# Rekomendasi Kebijakan - Laporan", f"_Generated: {ts}_", ""]
    lines += ["## Pertanyaan", query, ""]
    lines += ["## Jawaban/Analisis", answer, ""]
    lines += ["## Referensi Konteks"]
    for i, hit in enumerate(hits, 1):
        source = hit.get("source") or hit.get("doc_id") or "-"
        score = hit.get("score", 0.0)
        location = _format_location(hit)
        location_note = f" ({location})" if location else ""
        lines.append(f"{i}. **{source}**{location_note} - skor: {score:.3f}")
        snippet = hit.get("text", "")[:800]
        lines.append(f"   > {snippet}")
    if rec:
        lines += ["", "## Rekomendasi Sistem", rec]
    return "\n".join(lines)