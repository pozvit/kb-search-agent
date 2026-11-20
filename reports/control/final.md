# Контрольный набор: final

Версия `final-2026-11-20`, LLM `ollama`, retriever `qdrant`, 2026-11-20T11:23:38, правила проверки привязки v2.

| | Метрика | Значение | Цель |
|---|---|---|---|
| Q1 | Ответы `documents` без привязки к источникам | 0,0 | 0 |
| Q2 | Точность маршрутизации | 1,0 | ≥ 0,90 |
| Q3 | Эскалации: полнота | 1,0 | 1,00 |
| Q3 | Эскалации: точность | 1,0 | ≥ 0,80 |
| Q4 | Полнота перечислений | 1,0 | 1,00 |
| Q6 | Время, маршрут registry, медиана, с | 0,0 | < 1 |
| Q6 | Время, маршрут documents, медиана, с | 41,58 | ≤ 36 |
| Q7 | Промпт шага «план», медиана, токенов | 665 | ≤ 900 |
| Q8 | Доля принудительного поиска | 0,0 | не выше baseline |
| — | Шагов пройдено | 17/17 | 17/17 |

| Шаг | Маршрут | Статус | Время, с | Итог | Не пройдено |
|---|---|---|---|---|---|
| CS-01.1 | documents | answered | 41,6 | ✓ |  |
| CS-02.1 | documents | answered | 37,5 | ✓ |  |
| CS-03.1 | documents | answered | 67,9 | ✓ |  |
| CS-04.1 | escalation | escalated | 35,6 | ✓ |  |
| CS-05.1 | registry | answered | 0,0 | ✓ |  |
| CS-06.1 | registry | answered | 0,0 | ✓ |  |
| CS-06.2 | registry | answered | 0,0 | ✓ |  |
| CS-07.1 | registry | answered | 0,0 | ✓ |  |
| CS-08.1 | registry | answered | 0,0 | ✓ |  |
| CS-09.1 | registry | answered | 2,5 | ✓ |  |
| CS-10.1 | smalltalk | smalltalk | 0,0 | ✓ |  |
| CS-11.1 | smalltalk | smalltalk | 2,5 | ✓ |  |
| CS-12.1 | escalation | escalated | 0,0 | ✓ |  |
| CS-13.1 | escalation | escalated | 60,4 | ✓ |  |
| CS-14.1 | documents | degraded | 4,8 | ✓ |  |
| CS-15.1 | escalation | escalated | 6,3 | ✓ |  |
| CS-16.1 | rejected | rejected | 0,0 | ✓ |  |
