"""
Метаданные карточек реестров для чанков.

tools/csv_to_cards.py пишет рядом с карточками cards_meta.jsonl: одна строка на
карточку с ключом (file_name, card_number) и полями card_type, card_id, card_title,
year, year_end, industry, service, client, status.

Чанк карточки начинается с её заголовка «N. …». Это верно и для частей длинной
карточки: chunk_by_sections повторяет заголовок в начале каждой части. По имени
файла и номеру из заголовка чанк получает метаданные своей карточки, а chunk_pipeline
кладёт их в metadata — оттуда они попадают в payload Qdrant и работают как фильтры.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

CARD_FIELDS = ('card_type', 'card_id', 'card_title', 'year', 'year_end',
               'industry', 'service', 'client', 'status')
CARD_HEADING = re.compile(r'^\s*(\d{1,3})\.\s')

CardIndex = Dict[Tuple[str, int], Dict[str, Any]]


def load_card_metadata(paths: Iterable[str]) -> CardIndex:
    index: CardIndex = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"   метаданные карточек не найдены: {p} (пропущено)")
            continue
        for line in p.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            meta = {k: item.get(k) for k in CARD_FIELDS
                    if item.get(k) not in (None, '', [])}
            index[(item['file_name'], int(item['card_number']))] = meta
    return index


def card_meta_for_chunk(index: CardIndex, file_name: Optional[str], text: str) -> Optional[Dict[str, Any]]:
    if not index or not file_name:
        return None
    m = CARD_HEADING.match(text)
    if not m:
        return None
    return index.get((file_name, int(m.group(1))))
