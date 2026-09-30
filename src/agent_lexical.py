"""
Лексический retriever (BM25) по data/<profile>/chunks/chunks.jsonl.

Зачем: воспроизводимый прогон сценариев у любого проверяющего без скачивания весов
multilingual-e5-small (~470 МБ), torch и Qdrant. Это настоящий поиск по тем же чанкам,
что лежат в индексе, а не заготовленные ответы; интерфейс совпадает с
QdrantRetrieverAdapter (search / list_documents / close).

Основной режим агента — семантический поиск через Qdrant (--retriever qdrant, по умолчанию).

Токенизация: нижний регистр, ё->е, основа = первые 4 символа слова длиной от 4 букв.
Для русского это грубый, но рабочий заменитель стемминга: рынок/рынка -> «рынк».
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

STEM_LEN = 4

STOPWORDS = {
    "какие", "какой", "какая", "каких", "каком", "какое", "каков", "какова", "каковы", "сколько",
    "входят", "входит", "есть", "этого", "этой", "этом", "этот", "там", "было", "были", "была",
    "может", "также", "который", "которые", "найди", "найти", "расскажи", "подскажи", "пожалуйста",
    "что", "такое", "чего", "через", "между", "почему", "зачем", "когда", "где", "кто", "для",
    "при", "про", "его", "её", "них", "ними", "они", "она", "оно", "это", "как", "так", "или",
    "top_k", "топ", "фрагменты", "фрагментов", "документах", "документы", "базе", "знаний",
    "какую", "каким", "свой", "свои", "всех", "весь", "всего", "этих", "того", "тому",
}


def stems(text: str) -> List[str]:
    words = re.findall(r"[a-zа-яё0-9]+", (text or "").lower().replace("ё", "е"))
    return [w[:STEM_LEN] for w in words if len(w) >= 4 and w not in STOPWORDS]


class LexicalRetrieverAdapter:
    def __init__(self, chunks_path: str, k1: float = 1.5, b: float = 0.75):
        path = Path(chunks_path)
        if not path.exists():
            raise FileNotFoundError(
                f"нет файла чанков {chunks_path}. Соберите его: python src/build_index.py --profile public --no-embed")
        self.k1, self.b = k1, b
        self.chunks: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))
        if not self.chunks:
            raise RuntimeError(f"файл чанков {chunks_path} пуст")

        self._tf: List[Counter] = []
        df: Counter = Counter()
        for chunk in self.chunks:
            meta = chunk.get("metadata", {})
            tokens = stems(f"{meta.get('document_name', '')} {chunk.get('text', '')}")
            tf = Counter(tokens)
            self._tf.append(tf)
            df.update(tf.keys())
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = sum(self._len) / len(self._len)
        n = len(self.chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, top_k: int, category: Optional[str] = None,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        from agent_tools import payload_matches
        q_terms = set(stems(query))
        scored = []
        for i, tf in enumerate(self._tf):
            meta = self.chunks[i].get("metadata", {})
            if not payload_matches(meta, category, filters):
                continue
            score = 0.0
            for term in q_terms:
                freq = tf.get(term)
                if not freq:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg_len)
                score += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        results = []
        for rank, (score, i) in enumerate(scored[:top_k], start=1):
            chunk = self.chunks[i]
            meta = dict(chunk.get("metadata", {}))
            results.append({
                "rank": rank,
                "score": round(score, 4),
                "chunk_id": meta.get("chunk_id"),
                "document_id": meta.get("document_id"),
                "text": chunk.get("text", ""),
                "metadata": meta,
            })
        return results

    def facets(self) -> Dict[str, Any]:
        from agent_tools import FacetAccumulator
        acc = FacetAccumulator()
        for chunk in self.chunks:
            acc.add(chunk.get("metadata", {}))
        return acc.result()["facets"]

    def list_documents(self) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for chunk in self.chunks:
            meta = chunk.get("metadata", {})
            key = meta.get("file_name") or meta.get("document_name", "?")
            entry = seen.setdefault(key, {
                "document_name": meta.get("document_name", key),
                "category": meta.get("category", "n/a"),
                "file_name": meta.get("file_name", key),
                "version": meta.get("version", "n/a"),
                "chunks": 0,
            })
            entry["chunks"] += 1
        return sorted(seen.values(), key=lambda d: d["file_name"])

    def close(self) -> None:
        pass
