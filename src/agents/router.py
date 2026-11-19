"""
Маршрутизатор: выбирает подагента для вопроса.

Правила проверяются по порядку, первое сработавшее определяет маршрут, правило и признаки
пишутся в trace:
  1. приветствие, благодарность                                  -> smalltalk
     просьба связаться с человеком (E4)                           -> escalation
  2. вопрос к реестрам (перечисление, рейтинг, люди, сотрудник)  -> registry
  3. название продукта из базы или вопрос о содержании документов -> documents
  4. уточнение прошлого ответа («а там…», «а сроки?»)            -> documents
  5. остальное — классификатор (короткий вызов модели без инструментов):
       registry      -> registry, если в вопросе есть область; иначе documents
       documents     -> documents
       out_of_scope  -> вопрос вне предметной области: вежливый отказ без поиска

Если классификатор недоступен или ответил не по формату, выбирается documents:
лучше поискать по базе, чем отказать.

Эскалация к человеку (E1–E4) добавляется отдельно (шаг 2.4.4).
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agents.escalation import USER_REQUEST
from agents.registry_agent import Scope, detect_scope, match_registry

SMALLTALK = re.compile(r"(?i)^\s*(привет|здравствуй|здравствуйте|добрый (день|вечер|утро)|спасибо|благодарю|пока)"
                       r"\b[\s!.,)]*$")
# вопрос о содержании документов: подход, результаты, описание, сравнение
DOC_INTENT = re.compile(r"(?i)(?<![а-яё])(как|почему|зачем|чем|что\s+такое|что\s+входит)(?![а-яё])|"
                        r"(?<![а-яё])(опиш\w*|описыва\w*|описани\w*|подход\w*|методик\w*|эффект\w*|результат\w*|"
                        r"преимуществ\w*|функци\w*|возможност\w*|документ\w*|база\s+знаний)")

CLASSIFIER_LABELS = ("registry", "documents", "out_of_scope")


@dataclass
class RouteDecision:
    route: str                          # registry | documents | smalltalk | out_of_scope
    rule: str
    kind: Optional[str] = None          # для registry: rank | people | list
    args: Dict[str, Any] = field(default_factory=dict)
    intro: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    used_classifier: bool = False

    def trace_data(self) -> Dict[str, Any]:
        return {"route": self.route, "rule": self.rule, "kind": self.kind, "classifier": self.used_classifier,
                "args": {k: v for k, v in self.args.items() if k != "cards"},
                "cards": len(self.args.get("cards") or []), **self.details}


class Router:
    def __init__(self, catalog_provider: Callable[[], Any], config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._catalog = catalog_provider
        self.use_classifier = bool(config.get("classifier", True))
        self.products = [p for p in config.get("products", []) if p]

    def mentions_product(self, query: str) -> Optional[str]:
        low = query.lower()
        for product in self.products:
            if re.search(r"(?<![a-zа-яё0-9])" + re.escape(product.lower()) + r"(?![a-zа-яё0-9])", low):
                return product
        return None

    def decide(self, query: str, followup_cards: List[Dict[str, str]], is_followup: bool,
               classify: Optional[Callable[[str], Optional[str]]] = None) -> RouteDecision:
        if SMALLTALK.match(query):
            return RouteDecision("smalltalk", "greeting")
        if USER_REQUEST.search(query):
            # E4: просьба связаться с человеком; ответственный — по теме вопроса
            topic = "registry" if re.search(r"(?i)реестр|проект|пропозал|сотрудник", query) else \
                "documents" if re.search(r"(?i)(?<![а-яё])кп(?![а-яё])|предложени|продукт|документ", query) else None
            return RouteDecision("escalation", "user_request", details={"topic": topic})

        catalog = self._catalog()
        scope = detect_scope(query, catalog) if catalog is not None else Scope()
        details: Dict[str, Any] = {"scope": scope.args()} if scope else {}

        if catalog is not None:
            match = match_registry(query, scope, followup_cards)
            if match.kind:
                return RouteDecision("registry", f"registry_{match.kind}", match.kind, match.args, match.intro, details)
            if match.residual:
                details["residual"] = match.residual

        product = self.mentions_product(query)
        if product:
            return RouteDecision("documents", "product_name", details={**details, "product": product})
        if DOC_INTENT.search(query):
            return RouteDecision("documents", "document_intent", details=details)
        if is_followup:
            return RouteDecision("documents", "followup", details=details)

        if not (self.use_classifier and classify):
            return RouteDecision("documents", "default_no_classifier", details=details)
        label = classify(query)
        details["classifier_label"] = label
        if label == "out_of_scope":
            return RouteDecision("out_of_scope", "classifier", details=details, used_classifier=True)
        if label == "registry":
            if scope and catalog is not None:
                return RouteDecision("registry", "classifier", "list", {**scope.args(), "include_people": False},
                                     "В реестрах найдено", details, used_classifier=True)
            return RouteDecision("documents", "classifier_registry_without_scope", details=details,
                                 used_classifier=True)
        if label == "documents":
            return RouteDecision("documents", "classifier", details=details, used_classifier=True)
        return RouteDecision("documents", "classifier_unavailable", details=details, used_classifier=True)


def parse_label(text: str) -> Optional[str]:
    """Первое слово ответа классификатора, если оно из списка; иначе None."""
    words = re.findall(r"[a-z_]+", (text or "").lower())
    for word in words[:3]:
        if word in CLASSIFIER_LABELS:
            return word
    return None
