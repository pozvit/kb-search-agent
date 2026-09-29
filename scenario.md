# Сценарий работы агента

**Назначение одним предложением:** агент отвечает на вопрос пользователя по корпоративной базе знаний, находит релевантные фрагменты через семантический поиск и формирует ответ строго по ним со ссылками на источники.

## Входные и выходные данные

| | Описание |
|---|---|
| **Вход** | Вопрос на русском языке, строка от 1 до 1000 символов. Опционально — явный параметр `top_k=N` (от 1 до 10) |
| **Обязательные данные** | Непустой вопрос. При отсутствии — статус `rejected` с подсказкой, как сформулировать вопрос |
| **Выход** | `status`, текст ответа, список источников (документ, файл, чанк, score), вызовы tools, предупреждения валидатора, trace |
| **Статусы** | `answered`, `not_found`, `smalltalk`, `rejected`, `error`, `degraded` |

## Основной сценарий

```
Пользователь: «Какие три цифровых решения входят в цифровое ядро AIMS?»
        ↓
1. RECEIVE     — запрос принят, создан AgentState и Trace
        ↓
2. GUARDRAILS  — строка, не пустая, ≤ 1000 символов, секретов нет → допущен
                 явных параметров (top_k=…) нет → ничего не фиксируется
        ↓
3. PLAN        — LLM (qwen2.5:7b-instruct) получает system prompt + память + вопрос + схемы 2 tools
                 → решение: tool_call search_knowledge_base(query="цифровое ядро AIMS решения", top_k=4)
        ↓
4. ACT         — tool проверяет аргументы (query непустой, top_k ∈ [1..10], category допустима)
                 → e5-эмбеддинг запроса → Qdrant → 4 чанка с метаданными
                 → фрагменты получают сквозные номера [Фрагмент 1..4] и попадают в реестр
        ↓
5. ANSWER      — LLM получает результаты tool и формирует ответ со ссылками [Фрагмент N]
        ↓
6. VALIDATE    — ссылки ⊆ реестра (лишние удаляются с предупреждением);
                 числа в ответе ищутся в тексте фрагментов (неподтверждённые помечаются)
        ↓
7. RESPOND     — status=answered, ответ, источники (03_aims_asset_management.pdf, чанк 8),
                 память += 2 сообщения, trace сохранён
```

Типичная стоимость запроса: **2 вызова LLM** (план + ответ) и **1 вызов tool**.

## Точки принятия решений

| Точка | Условие | Переход |
|---|---|---|
| `guardrails` | пустой / не строка / > 1000 символов / секрет | → `rejected`, LLM и tools не вызываются |
| `user_params` | в запросе `top_k=N` или «топ-N» | значение фиксируется и подставляется в tool поверх значения от LLM |
| `route` | LLM ответила без tool, и это приветствие | → `smalltalk` |
| `route` | LLM ответила без tool на вопрос по базе | → `forced_search`: поиск выполняется принудительно |
| `tool_failed` | `index_unavailable` / `internal_error` | → `error`, шаблонный ответ, LLM больше не вызывается |
| `tool_failed` | `invalid_argument` по параметру пользователя | → `error`: молча исправлять значение пользователя нельзя |
| `tool_failed` | `invalid_argument` по аргументу, придуманному LLM | ошибка возвращается LLM для исправления, следующая итерация |
| `empty_results` | поиск успешен, 0 фрагментов | → `not_found` без второго вызова LLM |
| `iteration_limit` | 3 круга «LLM → tools» исчерпаны | финальный вызов LLM без tools |
| `llm_unavailable` | Ollama не отвечает / таймаут / HTTP-ошибка | → `degraded`: резервный поиск, фрагменты без генерации |
| `validation` | в ответе фраза «Недостаточно информации…» | → `not_found`, источники не показываются |

## Альтернативные и ошибочные сценарии

| ID | Тип | Запрос | Цепочка | Ожидаемый результат |
|---|---|---|---|---|
| `alternative_not_found` | alternative | Стоимость лицензии SIPS для 500 сотрудников | guardrails → plan → search → answer → validate | `not_found`, фраза «Недостаточно информации…», ни одной суммы |
| `alternative_list_documents` | alternative | Какие документы есть в базе знаний? | guardrails → plan → **list_documents** → answer | Перечислены 5 продуктов |
| `alternative_smalltalk` | alternative | Привет! | guardrails → plan → respond | `smalltalk`, 0 вызовов tools |
| `error_top_k_range` | error | …AIMS, top_k=0 | guardrails → lock → plan → search(**error**) → шаблон | `error / invalid_argument`, «от 1 до 10», 1 вызов LLM |
| `error_top_k_type` | error | Что такое SIPS? top_k=пять | то же | `error / invalid_argument`, «целым числом» |
| `guardrail_secret` | error | …ключ sk-proj-… | guardrails(**block**) | `rejected / secret_detected`, 0 LLM, ключ отсутствует в trace |
| `guardrail_empty` | error | «   » | guardrails(**block**) | `rejected / empty_query` |
| `tool_failure_index` | tool_failure | вопрос по AIMS; коллекции нет | plan → search(**index_unavailable**) → шаблон | `error`, подсказка `build_index`, ни одного факта из документов |
| `tool_failure_llm` | tool_failure | вопрос по AIMS; Ollama недоступна | plan(**LLMError**) → search(degraded) → шаблон | `degraded / llm_unavailable`, найденные фрагменты без генерации |
| `recovery_followup` | recovery | 1) Что такое SIPS? 2) А какой объём электрификации путей там был? | memory → plan → search(query с «SIPS») → answer | во 2-м шаге query содержит SIPS, ответ 135 км, память 2 → 4 |

Декларативное описание со всеми критериями прохождения — `config/agent_scenarios.yaml`.
