"""
Каталог карточек реестров для перечислений.

Семантический поиск отвечает на вопрос «что больше всего похоже на запрос» и возвращает
не больше top_k фрагментов. Для вопросов «какие проекты были в отрасли X» это неверный
инструмент: из 18 карточек Pharma & HC до модели доходило не больше 10. Каталог читает
cards_meta.jsonl (его пишет tools/csv_to_cards.py) и выбирает ВСЕ карточки, подходящие
под условия: тип, отрасль, период, конкретные карточки прошлого ответа.

Каталог не зависит от Qdrant: люди и компания лежат в cards_meta.jsonl, поэтому после
изменения карточек достаточно перезапустить csv_to_cards.py, индекс пересобирать не нужно.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CARD_TYPE_RU = {"project": "проект", "proposal": "пропозал", "project_accounting": "проект учётной базы"}


TRANSLIT = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
            "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
            "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
            "э": "e", "ю": "yu", "я": "ya"}


def latin_key(word: str) -> str:
    """Упрощённая транслитерация для сравнения: «Сидоренко» -> «sidorenko»; латиница — нижний регистр без лишнего."""
    w = word.lower()
    if re.search(r"[а-яё]", w):
        w = "".join(TRANSLIT.get(ch, ch) for ch in w)
    w = w.replace("kh", "h").replace("ks", "x").replace("iy", "y").replace("yi", "y")   # варианты записи
    return re.sub(r"[^a-z]", "", w)


PERSON_FULL = re.compile(r"([А-ЯЁA-Z][а-яёa-z]+(?:-[А-ЯЁA-Z][а-яёa-z]+)?)\s+([А-ЯЁA-Z])\.(?:\s?[А-ЯЁA-Z]\.)?")
PEOPLE_FIELDS = (("team", "команда"), ("partner", "партнёр / директор"), ("employee", "имя сотрудника"))


def split_people(text: Any) -> List[str]:
    """«Иванов И.; Петров П.П.», «Иванов, Петров», ["Иванов И."] -> отдельные записи."""
    items = text if isinstance(text, list) else re.split(r"[;,\n]", text or "")
    result = []
    for item in items:
        item = re.sub(r"\s+", " ", str(item)).strip()      # точку не срезаем: она часть «Фамилия И.»
        if not item:
            continue
        full = PERSON_FULL.findall(item)
        if len(full) > 1:                         # несколько «Фамилия И.» в одной строке
            result += [f"{s} {i}." for s, i in full]
        else:
            result.append(item)
    return result


class CardCatalog:
    def __init__(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"нет каталога карточек {path}: выполните python tools/csv_to_cards.py")
        seen = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                seen[(item.get("card_type"), str(item.get("card_id")))] = item   # дубли по (тип, ID) схлопываются
        self.cards: List[Dict[str, Any]] = list(seen.values())
        self._build_people_index()

    # ------------------------------------------------------------------ люди

    def _build_people_index(self) -> None:
        """
        Канонический ключ человека — «Фамилия И.». Запись без инициалов («Иванов», «Иван Иванов»)
        приводится к нему, если в реестрах ровно один человек с такой фамилией; иначе остаётся как есть.
        """
        by_surname: Dict[str, set] = defaultdict(set)
        for card in self.cards:
            for field, _ in PEOPLE_FIELDS:
                for raw in split_people(card.get(field)):
                    m = PERSON_FULL.fullmatch(raw) or PERSON_FULL.match(raw)
                    if m:
                        by_surname[m.group(1).lower()].add(f"{m.group(1)} {m.group(2)}.")
        self._unique_surname = {s: next(iter(v)) for s, v in by_surname.items() if len(v) == 1}
        # латинская запись той же фамилии («Sidorenko Alexey» -> «Сидоренко А.»): только однозначные совпадения
        by_latin: Dict[str, set] = defaultdict(set)
        for keys in by_surname.values():
            for canon in keys:
                by_latin[latin_key(canon.split()[0])].add(canon)
        self._latin_surname = {k: next(iter(v)) for k, v in by_latin.items() if len(v) == 1 and k}
        for card in self.cards:
            roles: Dict[str, List[str]] = {}
            for field, _ in PEOPLE_FIELDS:
                for raw in split_people(card.get(field)):
                    key = self.person_key(raw)
                    roles.setdefault(key, [])
                    if field not in roles[key]:
                        roles[key].append(field)
            card["_people"] = roles

    def person_key(self, raw: str) -> str:
        m = PERSON_FULL.match(raw)
        if m:
            return f"{m.group(1)} {m.group(2)}."
        tokens = raw.split()
        for token in tokens:
            canon = self._unique_surname.get(token.lower())
            if canon:
                return canon
        for i, token in enumerate(tokens):
            canon = self._latin_surname.get(latin_key(token))
            if not canon:
                continue
            others = [t for j, t in enumerate(tokens) if j != i]
            initial = latin_key(canon.split()[1][0])[:1]
            # имя рядом с фамилией должно начинаться с той же буквы, что инициал: «Alexey» ~ «А.»
            if not others or latin_key(others[0])[:1] == initial:
                return canon
        return raw

    @staticmethod
    def surname_stem(surname: str) -> str:
        """
        Основа фамилии для поиска в косвенных падежах. Окончание отбрасывается, если оно меняется при склонении:
          «Белозёрова» -> «белозеров» (Белозёровой), «Камская» -> «камск» (Камской),
          «Ангарский» -> «ангарск» (Ангарского); «Петров» -> «петров» (Петрова, Петровым).
        «ё» и «е» не различаются.
        """
        s = surname.lower().replace("ё", "е")
        if s.endswith(("ая", "яя", "ий", "ый", "ой")) and len(s) > 5:
            return s[:-2]
        if s.endswith(("а", "я")) and len(s) > 4:
            return s[:-1]
        return s

    def find_person(self, text: str) -> List[str]:
        """Люди, упомянутые в вопросе фамилией в любом падеже: «проекты Белозёровой» -> ["Белозёрова Б."]."""
        tokens = [t.lower().replace("ё", "е")
                  for t in re.findall(r"[А-ЯЁA-Z][а-яёa-z]+(?:-[А-ЯЁA-Z][а-яёa-z]+)?", text)]
        exact, by_stem = [], []
        for surname, canon in self._unique_surname.items():
            full = surname.lower().replace("ё", "е")
            stem = self.surname_stem(surname)
            for base in tokens:
                if base == full and canon not in exact:
                    exact.append(canon)
                elif len(stem) >= 4 and base.startswith(stem) and len(base) - len(stem) <= 3 and canon not in by_stem:
                    by_stem.append(canon)
        # точное совпадение важнее: «Петрова» — это Петрова, а не Петров в родительном падеже
        if exact:
            return exact
        # однофамильцы разного пола: «-ой», «-ою» — только у женских фамилий («Петровой» — Петрова),
        # «-ым», «-ом», «-ому» — только у мужских («Петровым» — Петров)
        if len(by_stem) > 1:
            is_fem = lambda c: c.split()[0].lower().endswith(("а", "я"))
            if any(b.endswith(("ой", "ою")) for b in tokens):
                return [c for c in by_stem if is_fem(c)] or by_stem
            if any(b.endswith(("ым", "ом", "ому")) for b in tokens):
                return [c for c in by_stem if not is_fem(c)] or by_stem
        return by_stem

    def services(self) -> List[str]:
        counts: Dict[str, int] = {}
        for c in self.cards:
            for v in c.get("service") or []:
                counts[v] = counts.get(v, 0) + 1
        return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    def service_details(self) -> List[str]:
        counts: Dict[str, int] = {}
        for c in self.cards:
            for v in c.get("service_detail") or []:
                counts[v] = counts.get(v, 0) + 1
        return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    def aggregate_people(self, cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Сколько карточек у каждого человека: проекты и пропозалы отдельно, плюс роли."""
        stats: Dict[str, Dict[str, Any]] = {}
        for card in cards:
            for person, fields in (card.get("_people") or {}).items():
                row = stats.setdefault(person, {"person": person, "projects": 0, "proposals": 0, "other": 0,
                                                "roles": defaultdict(int), "examples": []})
                kind = {"project": "projects", "proposal": "proposals"}.get(card.get("card_type"), "other")
                row[kind] += 1
                for f in fields:
                    row["roles"][f] += 1
                if len(row["examples"]) < 3:
                    row["examples"].append(card.get("card_title", ""))
        ranked = sorted(stats.values(), key=lambda r: (-r["projects"], -(r["projects"] + r["proposals"] + r["other"]),
                                                       r["person"]))
        for r in ranked:
            r["roles"] = dict(r["roles"])
        return ranked

    def industries(self) -> List[str]:
        counts: Dict[str, int] = {}
        for c in self.cards:
            for v in c.get("industry") or []:
                counts[v] = counts.get(v, 0) + 1
        return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    def select(self, card_type: Optional[str] = None, industry: Optional[List[str]] = None,
               year_from: Optional[int] = None, year_to: Optional[int] = None,
               cards: Optional[List[Dict[str, str]]] = None, service: Optional[List[str]] = None,
               person: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        refs = {(c["card_type"], str(c["card_id"])) for c in cards} if cards else None
        result = []
        for c in self.cards:
            if refs is not None and (c.get("card_type"), str(c.get("card_id"))) not in refs:
                continue
            if card_type and c.get("card_type") != card_type:
                continue
            if industry and not set(c.get("industry") or []) & set(industry):
                continue
            if service and not set((c.get("service") or []) + (c.get("service_detail") or [])) & set(service):
                continue
            if person and not set(c.get("_people") or {}) & set(person):
                continue
            if year_from is not None and not (isinstance(c.get("year_end"), int) and c["year_end"] >= year_from):
                continue
            if year_to is not None and not (isinstance(c.get("year"), int) and c["year"] <= year_to):
                continue
            result.append(c)
        # новые сверху; без года — в конце
        result.sort(key=lambda c: (-(c.get("year") or 0), c.get("card_type", ""), c.get("card_title", "")))
        return result


def card_years(c: Dict[str, Any]) -> str:
    y, ye = c.get("year"), c.get("year_end")
    if not y:
        return "год не указан"
    return f"{y}–{ye}" if ye and ye != y else str(y)


def format_card_line(n: int, c: Dict[str, Any], include_people: bool = False) -> str:
    parts = [f"{n}. {c.get('card_title', 'Без названия')}"]
    if c.get("client"):
        parts.append(f"— {c['client']}")
    parts.append(f"({CARD_TYPE_RU.get(c.get('card_type'), c.get('card_type'))}, {card_years(c)})")
    line = " ".join(parts)
    if include_people:
        people = []
        if c.get("team"):
            people.append("команда: " + "; ".join(c["team"]))
        if c.get("partner"):
            people.append("партнёр / директор: " + c["partner"])
        if c.get("employee"):
            people.append("имя сотрудника: " + c["employee"])
        line += "\n   " + (" | ".join(people) if people else "люди в реестре не указаны")
    return line


def format_card_list(selected: List[Dict[str, Any]], limit: int, include_people: bool = False) -> str:
    lines = [format_card_line(i, c, include_people) for i, c in enumerate(selected[:limit], start=1)]
    if len(selected) > limit:
        lines.append(f"… показаны {limit} из {len(selected)}; уточните период, отрасль или тип реестра.")
    return "\n".join(lines)


def format_people_ranking(ranked: List[Dict[str, Any]], limit: int) -> str:
    role_names = dict(PEOPLE_FIELDS)
    lines = []
    for i, r in enumerate(ranked[:limit], start=1):
        roles = ", ".join(f"{role_names.get(k, k)} {v}" for k, v in r["roles"].items())
        lines.append(f"{i}. {r['person']} — проектов {r['projects']}, пропозалов {r['proposals']}"
                     + (f", прочих {r['other']}" if r["other"] else "") + f" (роли: {roles})")
    return "\n".join(lines)
