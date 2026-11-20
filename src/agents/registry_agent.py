"""
Подагент «Реестры»: перечисления, люди по прошлому ответу, рейтинги, проекты сотрудника.

Ответ строится выборкой из каталога карточек (инструмент list_cards) и форматируется программно,
без генерации моделью: список полный и точный, время ответа — доли секунды.

Здесь — распознавание области вопроса (отрасль, тип услуги, сотрудник, период, тип реестра),
правила, по которым вопрос относится к реестрам, проверка посторонней темы и форматирование ответа.
Выполнение (вызов list_cards, trace, память) остаётся за агентом.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_tools import detect_industries, detect_period, detect_services

ENUM = re.compile(r"(?i)^\s*(какие|каких|перечисли\w*|список|покажи|сколько|назови|выведи)(?![а-яё])|"
                  r"(?<![а-яё])все\s+(проект|пропозал)")
REGISTRY_NOUN = re.compile(r"(?i)(?<![а-яё])(проект|пропозал|кейс|опыт)")
PEOPLE = re.compile(r"(?i)(?<![а-яё])(кто|сотрудник\w*|команд\w*|участ\w*|люд\w*|человек\w*|исполнител\w*|"
                    r"делал\w*|выполнял\w*|работал\w*|партн[её]р\w*)(?![а-яё])")
AGG = re.compile(r"(?i)(больше\s+всего|(?<![а-яё])много(?![а-яё])|чаще\w*|наибольш\w*|(?<![а-яё])топ(?![а-яё])|"
                 r"самы[йеех]\s+опытн\w*|лидер\w*|рейтинг\w*)")
WHO = re.compile(r"(?i)(?<![а-яё])(кто|у\s+кого|сотрудник\w*|люди|людей|эксперт\w*)(?![а-яё])")
PERSON_WORK = re.compile(r"(?i)(?<![а-яё])(делал\w*|работал\w*|участвовал\w*|вел[аи]?|выполнял\w*|опыт\w*)(?![а-яё])")
# слова, которые не задают тему вопроса о реестрах
NEUTRAL = {"прое", "проп", "кейс", "опыт", "рабо", "отра", "сфер", "обла", "сект", "году", "года", "годы",
           "пери", "реес", "спис", "пере", "пока", "назо", "выве", "скол", "дела", "выпо", "реал", "вели",
           "наше", "наши", "всег", "посл", "нача", "ране", "позж", "пред", "мног", "боль", "чаще", "наиб",
           "самы", "лиде", "рейт", "эксп", "сотр", "люди", "люде", "сдел", "учас", "кого", "делал",
           "темы", "теме", "направ", "текущ", "этом", "прош", "лет", "года"}

KIND_INTRO = {"rank": "Кто чаще всего участвовал", "people": "Люди по карточкам из прошлого ответа",
              "list": "В реестрах найдено"}


def _yo(text: str) -> str:
    """Буква «ё» и «е» в сравнениях одинаковы: «Белозёровой» и «Белозеровой» — одна фамилия."""
    return text.lower().replace("ё", "е")


@dataclass
class Scope:
    """Область вопроса, распознанная по тексту."""
    industries: List[str] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    persons: List[str] = field(default_factory=list)
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    card_type: Optional[str] = None

    def args(self) -> Dict[str, Any]:
        return {k: v for k, v in (("industry", self.industries), ("service", self.services),
                                  ("person", self.persons), ("year_from", self.year_from),
                                  ("year_to", self.year_to), ("card_type", self.card_type)) if v}

    def __bool__(self) -> bool:
        return bool(self.args())


def detect_scope(query: str, catalog: Any) -> Scope:
    if catalog is None:
        return Scope()
    industries = detect_industries(query, catalog.industries())
    services = detect_services(query, catalog.services(), catalog.service_details())
    if industries and services:
        # одно слово и отрасль, и услуга («по логистике»): это тема услуги, а не отрасль клиента
        from agent_tools import INDUSTRY_SYNONYMS, SERVICE_SYNONYMS, covered_words, tag_keys
        service_phrases = [p for sv in services for p in SERVICE_SYNONYMS.get(sv, []) + tag_keys(sv) + [sv]]
        service_words = covered_words(query, service_phrases)
        industries = [ind for ind in industries
                      if covered_words(query, INDUSTRY_SYNONYMS.get(ind, []) + [ind]) - service_words]
    year_from, year_to = detect_period(query)
    only_proposals = re.search(r"(?i)пропозал|предложени", query) and not re.search(r"(?i)проект", query)
    return Scope(industries=industries, services=services, persons=catalog.find_person(query),
                 year_from=year_from, year_to=year_to, card_type="proposal" if only_proposals else None)


def residual_topic(query: str, scope: Scope) -> List[str]:
    """Содержательные слова вопроса, кроме служебных, отраслевых, типа услуги, фамилий и периода."""
    from agent_lexical import STOPWORDS, stems
    from agent_tools import INDUSTRY_SYNONYMS, SERVICE_SYNONYMS, _matches_synonyms, covered_words, tag_keys
    phrases = [p for category in scope.industries for p in INDUSTRY_SYNONYMS.get(category, []) + [category]]
    phrases += [p for service in scope.services for p in SERVICE_SYNONYMS.get(service, [])]
    phrases += [w for service in scope.services for w in tag_keys(service)]
    service_stems = {st for service in scope.services for st in stems(service)}
    # фамилии сравниваются по той же основе, что и при поиске сотрудника: с учётом падежа и «ё» (дефект baseline №8)
    from agent_cards import CardCatalog
    surnames = {CardCatalog.surname_stem(p.split()[0]) for p in scope.persons}
    covered = covered_words(query, phrases) if phrases else set()
    residual = []
    for word in re.findall(r"[a-zа-яё]+", _yo(query)):
        if len(word) < 4 or word in STOPWORDS or word[:4] in NEUTRAL:
            continue
        if word in covered or (phrases and _matches_synonyms(word, phrases)) or PEOPLE.fullmatch(word):
            continue
        if word[:4] in service_stems or any(word.startswith(sn) for sn in surnames):
            continue
        residual.append(word)
    return residual


@dataclass
class RegistryMatch:
    kind: Optional[str]                 # rank | people | list; None — вопрос не к реестрам
    args: Dict[str, Any] = field(default_factory=dict)
    residual: List[str] = field(default_factory=list)

    @property
    def intro(self) -> str:
        return KIND_INTRO.get(self.kind or "", "")


def match_registry(query: str, scope: Scope, followup_cards: List[Dict[str, str]]) -> RegistryMatch:
    """
    Правила, по которым вопрос относится к реестрам:
      rank   — «кто сделал больше всего / много / топ …» с областью или словом «проект»;
      people — «кто выполнял эти проекты» по карточкам прошлого ответа, без новой области;
      list   — перечисление с областью или вопрос о проектах сотрудника.
    Если кроме области в вопросе есть другая тема («проекты по закупкам в медицине» — это тег, а вот
    «как мы описывали подход…» — тема), вопрос не к реестрам: residual содержит эти слова.
    """
    args = scope.args()
    if AGG.search(query) and (WHO.search(query) or PEOPLE.search(query)) and (REGISTRY_NOUN.search(query) or scope):
        residual = residual_topic(query, scope)
        return RegistryMatch(None, residual=residual) if residual else RegistryMatch("rank", {**args, "group_by": "people"})
    if followup_cards and PEOPLE.search(query) and not scope:
        return RegistryMatch("people", {"cards": followup_cards, "include_people": True})
    if (ENUM.search(query) and REGISTRY_NOUN.search(query) and scope) or (
            scope.persons and (REGISTRY_NOUN.search(query) or PERSON_WORK.search(query))):
        residual = residual_topic(query, scope)
        if residual:
            return RegistryMatch(None, residual=residual)
        return RegistryMatch("list", {**args, "include_people": bool(PEOPLE.search(query)) and not scope.persons})
    return RegistryMatch(None)


def catalog_answer(result: Dict[str, Any], intro: str) -> str:
    from agent_cards import CARD_TYPE_RU, format_card_line
    filters = result.get("filters") or {}
    conditions = []
    if filters.get("industry"):
        conditions.append("отрасль: " + ", ".join(filters["industry"]))
    if filters.get("year_from") or filters.get("year_to"):
        conditions.append(f"период: {filters.get('year_from') or '…'}–{filters.get('year_to') or '…'}")
    if filters.get("card_type"):
        conditions.append("тип: " + CARD_TYPE_RU.get(filters["card_type"], filters["card_type"]))
    if filters.get("service"):
        conditions.append("тип услуги: " + ", ".join(filters["service"]))
    if filters.get("person"):
        conditions.append("сотрудник: " + ", ".join(filters["person"]))
    cond = f" ({'; '.join(conditions)})" if conditions else ""
    if not result["total"]:
        return f"По условиям{cond} карточек в реестрах нет."
    if result.get("group_by") == "people":
        by_type = result.get("by_type", {})
        head = (f"{intro}{cond}. Карточек: {result['total']} (проектов {by_type.get('project', 0)}, "
                f"пропозалов {by_type.get('proposal', 0)}), люди указаны в {result['cards_with_people']}, "
                f"разных людей: {result['people_total']}.")
        if not result.get("people"):
            return head + "\nВ этих карточках люди не указаны."
        return (head + "\n\n" + result["text"] + "\n\nСортировка по числу проектов: пропозал — это заявка, "
                "а не выполненный проект. Учтены поля «Команда», «Партнёр / директор», «Имя сотрудника»; "
                "человек считается один раз на карточку.\nИсточник: каталог карточек реестров.")
    lines = [f"{intro or 'Найдено'}{cond}: {result['total']}."]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for card in result["cards"]:
        groups.setdefault(card.get("card_type", ""), []).append(card)
    titles = {"project": "Проекты", "proposal": "Пропозалы", "project_accounting": "Проекты учётной базы"}
    for card_type, cards in groups.items():
        lines.append(f"\n{titles.get(card_type, card_type)} ({len(cards)}):")
        lines += [format_card_line(i, c, result.get("include_people")) for i, c in enumerate(cards, start=1)]
    if result["shown"] < result["total"]:
        lines.append(f"\nПоказаны {result['shown']} из {result['total']}: уточните период, отрасль или тип реестра.")
    lines.append("\nИсточник: каталог карточек реестров.")
    return "\n".join(lines)


def card_sources(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = ("card_type", "card_id", "card_title", "year", "year_end", "industry")
    return [{"fragment": i, "document_name": c.get("card_title", ""), "file_name": c.get("file_name", ""),
             "category": "", "chunk_index": "—", "score": 1.0,
             "card": {k: c[k] for k in keys if c.get(k) not in (None, "", [])}}
            for i, c in enumerate(cards, start=1)]
