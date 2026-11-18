"""
Проверка привязки ответа подагента «Документы» к источникам (требование Q1, правило эскалации E1).

Ответ режется на утверждения: предложения и пункты списков. Не проверяются вводные строки перед
списком («…включает:»), строки «Источники: …» и утверждения короче min_terms значимых слов.

Утверждение подкреплено, если:
  1. у него есть ссылка [Фрагмент N] — в нём самом, в его абзаце или в строке «Источники: …»;
  2. фрагмент N реально был передан модели;
  3. доля значимых основ утверждения, которые есть в процитированных фрагментах, >= min_overlap;
  4. все числа утверждения есть в процитированных фрагментах. Номер пункта списка («1.»)
     числом не считается, а однозначные числа проверяются: «7 сегментов» вместо «5» — ошибка.
     (Валидатор baseline пропускал однозначные числа.)

Ответ без длинных утверждений (только короткие пункты) подкреплён, если в нём есть ссылки на
полученные фрагменты.

Это лексическая проверка. Она ловит утверждения без ссылки, ссылки «не туда» и чужие числа, но
не смысловую ошибку при верных словах. Инструмент замера tools/run_control.py реализует те же
правила своим кодом, чтобы не проверять агента его же реализацией.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

STOP = {"какие", "какой", "какая", "каких", "который", "которые", "также", "этого", "этой", "этом", "есть",
        "было", "были", "была", "будет", "между", "через", "после", "более", "менее", "очень", "может",
        "можно", "нужно", "этот", "такие", "такой", "всего", "всех", "данных", "данные", "документах",
        "фрагмент", "фрагмента", "фрагменте", "фрагменты", "источник", "источники", "базы", "знаний"}
CITE = re.compile(r"(?i)фрагмент\w*\s*((?:\d+\s*(?:,|и|;)\s*)*\d+)")
NUM = re.compile(r"(?<![\w.])\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?")
SOURCES_LINE = re.compile(r"(?i)^\s*источники")
SKIP_LINE = re.compile(r"(?i)^\s*(источники|использованные|предупреждени|⚠)")
LIST_NUMBER = re.compile(r"(?m)^\s*\d{1,2}[.)]\s+")


def _terms(text: str) -> Set[str]:
    words = re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е"))
    return {w[:4] for w in words if len(w) >= 4 and w not in STOP and not w.isdigit()}


def _cited(text: str) -> Set[int]:
    return {int(n) for group in CITE.findall(text) for n in re.findall(r"\d+", group)}


def _numbers(text: str) -> Set[str]:
    cleaned = LIST_NUMBER.sub(" ", CITE.sub(" ", text))
    return {re.sub(r"[ \u00a0]", "", raw).replace(",", ".") for raw in NUM.findall(cleaned)}


def _fragment_numbers(text: str) -> Set[str]:
    joined = re.sub(r"(?<=\d)[ \u00a0](?=\d{3}\b)", "", text)
    return {re.sub(r"[ \u00a0]", "", raw).replace(",", ".") for raw in NUM.findall(joined)}


@dataclass
class GroundingResult:
    grounded: bool
    units: int
    unsupported: List[Dict[str, str]] = field(default_factory=list)
    cited: List[int] = field(default_factory=list)          # ссылки на реально полученные фрагменты

    def summary(self) -> Dict[str, object]:
        return {"grounded": self.grounded, "units": self.units, "supported": self.units - len(self.unsupported),
                "unsupported": self.unsupported[:5], "cited": self.cited}


def check_grounding(answer: str, fragments: Dict[int, str], min_overlap: float = 0.5,
                    min_terms: int = 3) -> GroundingResult:
    answer = answer or ""
    answer_cites = {n for line in answer.split("\n") if SOURCES_LINE.match(line) for n in _cited(line)}
    units, unsupported = 0, []
    for paragraph in re.split(r"\n\s*\n", answer):
        lines = [line for line in paragraph.split("\n") if line.strip() and not SKIP_LINE.match(line)]
        if not lines:
            continue
        paragraph_cites = _cited("\n".join(lines))
        for line in lines:
            for unit in re.split(r"(?<=[.!?])\s+", line):
                if unit.rstrip().endswith(":"):
                    continue
                terms = _terms(CITE.sub(" ", unit))
                if len(terms) < min_terms:
                    continue
                units += 1
                refs = _cited(unit) or paragraph_cites or answer_cites
                texts = [fragments[r] for r in refs if r in fragments]
                if not texts:
                    unsupported.append({"unit": unit.strip()[:200], "why": "нет ссылки на полученный фрагмент"})
                    continue
                overlap = len(terms & set().union(*(_terms(t) for t in texts))) / len(terms)
                missing = sorted(_numbers(unit) - _fragment_numbers(" ".join(texts)))
                if overlap < min_overlap:
                    unsupported.append({"unit": unit.strip()[:200],
                                        "why": f"утверждения нет в процитированном фрагменте (общих слов {overlap:.0%})"})
                elif missing:
                    unsupported.append({"unit": unit.strip()[:200],
                                        "why": f"чисел {', '.join(missing)} нет в процитированном фрагменте"})
    valid = sorted(_cited(answer) & set(fragments))
    grounded = not unsupported and (units > 0 or bool(valid))
    return GroundingResult(grounded, units, unsupported, valid)
