# Контрольный набор: final

Версия `baseline-2026-10-02-25-g4c687a0-dirty`, LLM `ollama`, retriever `qdrant`, 2026-11-20T10:57:40, правила проверки привязки v2.

| | Метрика | Значение | Цель |
|---|---|---|---|
| Q1 | Ответы `documents` без привязки к источникам | 0,0 | 0 |
| Q2 | Точность маршрутизации | 1,0 | ≥ 0,90 |
| Q3 | Эскалации: полнота | 1,0 | 1,00 |
| Q3 | Эскалации: точность | 1,0 | ≥ 0,80 |
| Q4 | Полнота перечислений | 1,0 | 1,00 |
| Q6 | Время, маршрут registry, медиана, с | 0,0 | < 1 |
| Q6 | Время, маршрут documents, медиана, с | 46,03 | ≤ 36 |
| Q7 | Промпт шага «план», медиана, токенов | 665 | ≤ 900 |
| Q8 | Доля принудительного поиска | 0,0 | не выше baseline |
| — | Шагов пройдено | 17/17 | 17/17 |

| Шаг | Маршрут | Статус | Время, с | Итог | Не пройдено |
|---|---|---|---|---|---|
| CS-01.1 | documents | answered | 46,0 | ✓ |  |
| CS-02.1 | documents | answered | 41,3 | ✓ |  |
| CS-03.1 | documents | answered | 74,1 | ✓ |  |
| CS-04.1 | escalation | escalated | 38,5 | ✓ |  |
| CS-05.1 | registry | answered | 0,0 | ✓ |  |
| CS-06.1 | registry | answered | 0,0 | ✓ |  |
| CS-06.2 | registry | answered | 0,0 | ✓ |  |
| CS-07.1 | registry | answered | 0,0 | ✓ |  |
| CS-08.1 | registry | answered | 0,0 | ✓ |  |
| CS-09.1 | registry | answered | 5,3 | ✓ |  |
| CS-10.1 | smalltalk | smalltalk | 0,0 | ✓ |  |
| CS-11.1 | smalltalk | smalltalk | 2,6 | ✓ |  |
| CS-12.1 | escalation | escalated | 0,0 | ✓ |  |
| CS-13.1 | escalation | escalated | 84,8 | ✓ |  |
| CS-14.1 | documents | degraded | 6,3 | ✓ |  |
| CS-15.1 | escalation | escalated | 7,0 | ✓ |  |
| CS-16.1 | rejected | rejected | 0,0 | ✓ |  |
