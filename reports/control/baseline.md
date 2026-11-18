# Контрольный набор: baseline

Версия `baseline-2026-10-02`, LLM `ollama`, retriever `qdrant`, 2026-11-18T08:04:58, правила проверки привязки v2.

| | Метрика | Значение | Цель |
|---|---|---|---|
| Q1 | Ответы `documents` без привязки к источникам | 0,2 | 0 |
| Q2 | Точность маршрутизации | 0,588 | ≥ 0,90 |
| Q3 | Эскалации: полнота | 0,0 | 1,00 |
| Q3 | Эскалации: точность | — | ≥ 0,80 |
| Q4 | Полнота перечислений | 0,83 | 1,00 |
| Q6 | Время, маршрут registry, медиана, с | 0,0 | < 1 |
| Q6 | Время, маршрут documents, медиана, с | 33,75 | ≤ 36 |
| Q7 | Промпт шага «план», медиана, токенов | 2464,0 | ≤ 900 |
| Q8 | Доля принудительного поиска | 0,5 | не выше baseline |
| — | Шагов пройдено | 8/17 | 17/17 |

| Шаг | Маршрут | Статус | Время, с | Итог | Не пройдено |
|---|---|---|---|---|---|
| CS-01.1 | documents | not_found | 12,7 | ✗ | status, sources_any, facts_all |
| CS-02.1 | documents | answered | 38,0 | ✓ |  |
| CS-03.1 | documents | not_found | 12,8 | ✗ | status, sources_any, facts_all |
| CS-04.1 | documents | not_found | 28,9 | ✗ | route, status, escalation |
| CS-05.1 | registry | answered | 0,0 | ✓ |  |
| CS-06.1 | registry | answered | 0,0 | ✓ |  |
| CS-06.2 | registry | answered | 0,0 | ✓ |  |
| CS-07.1 | registry | answered | 0,0 | ✓ |  |
| CS-08.1 | documents | answered | 49,8 | ✗ | route, completeness |
| CS-09.1 | documents | answered | 29,5 | ✗ | route, completeness |
| CS-10.1 | smalltalk | smalltalk | 6,3 | ✓ |  |
| CS-11.1 | documents | not_found | 38,5 | ✗ | route, status, tool_calls_max |
| CS-12.1 | documents | answered | 39,6 | ✗ | route, status, escalation |
| CS-13.1 | documents | answered | 46,7 | ✗ | route, status, escalation, llm_regenerations |
| CS-14.1 | documents | degraded | 5,2 | ✓ |  |
| CS-15.1 | documents | error | 14,7 | ✗ | route, status, escalation |
| CS-16.1 | rejected | rejected | 0,0 | ✓ |  |
