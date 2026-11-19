# Контрольный набор: week2

Версия `baseline-2026-10-02-14-g1ad32d0`, LLM `ollama`, retriever `qdrant`, 2026-11-19T09:10:17, правила проверки привязки v2.

| | Метрика | Значение | Цель |
|---|---|---|---|
| Q1 | Ответы `documents` без привязки к источникам | 0,0 | 0 |
| Q2 | Точность маршрутизации | 1,0 | ≥ 0,90 |
| Q3 | Эскалации: полнота | 1,0 | 1,00 |
| Q3 | Эскалации: точность | 1,0 | ≥ 0,80 |
| Q4 | Полнота перечислений | 1,0 | 1,00 |
| Q6 | Время, маршрут registry, медиана, с | 0,0 | < 1 |
| Q6 | Время, маршрут documents, медиана, с | 45,03 | ≤ 36 |
| Q7 | Промпт шага «план», медиана, токенов | 619 | ≤ 900 |
| Q8 | Доля принудительного поиска | 0,0 | не выше baseline |
| — | Шагов пройдено | 16/17 | 17/17 |

| Шаг | Маршрут | Статус | Время, с | Итог | Не пройдено |
|---|---|---|---|---|---|
| CS-01.1 | documents | answered | 45,0 | ✓ |  |
| CS-02.1 | documents | answered | 43,0 | ✓ |  |
| CS-03.1 | documents | answered | 73,1 | ✗ | facts_all |
| CS-04.1 | escalation | escalated | 33,5 | ✓ |  |
| CS-05.1 | registry | answered | 0,0 | ✓ |  |
| CS-06.1 | registry | answered | 0,0 | ✓ |  |
| CS-06.2 | registry | answered | 0,0 | ✓ |  |
| CS-07.1 | registry | answered | 0,0 | ✓ |  |
| CS-08.1 | registry | answered | 0,0 | ✓ |  |
| CS-09.1 | registry | answered | 4,6 | ✓ |  |
| CS-10.1 | smalltalk | smalltalk | 0,0 | ✓ |  |
| CS-11.1 | smalltalk | smalltalk | 2,5 | ✓ |  |
| CS-12.1 | escalation | escalated | 0,0 | ✓ |  |
| CS-13.1 | escalation | escalated | 83,3 | ✓ |  |
| CS-14.1 | documents | degraded | 5,0 | ✓ |  |
| CS-15.1 | escalation | escalated | 6,7 | ✓ |  |
| CS-16.1 | rejected | rejected | 0,0 | ✓ |  |
