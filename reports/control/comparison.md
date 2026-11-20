# Сравнение: baseline_final → final

Одинаковые данные (`public_v2`) и сценарии. baseline_final: `baseline-2026-10-02`; final: `baseline-2026-10-02-25-g4c687a0-dirty`.

| | Метрика | baseline_final | final | Цель |
|---|---|---|---|---|
| Q1 | Ответы `documents` без привязки к источникам | 0,2 | 0,0 | 0 |
| Q2 | Точность маршрутизации | 0,588 | 1,0 | ≥ 0,90 |
| Q3 | Эскалации: полнота | 0,0 | 1,0 | 1,00 |
| Q3 | Эскалации: точность | — | 1,0 | ≥ 0,80 |
| Q4 | Полнота перечислений | 0,83 | 1,0 | 1,00 |
| Q6 | Время, маршрут registry, медиана, с | 0,0 | 0,0 | < 1 |
| Q6 | Время, маршрут documents, медиана, с | 36,01 | 46,03 | ≤ 36 |
| Q7 | Промпт шага «план», медиана, токенов | 2464,0 | 665 | ≤ 900 |
| Q8 | Доля принудительного поиска | 0,5 | 0,0 | не выше baseline |
| — | Шагов пройдено | 8/17 | 17/17 | 17/17 |

| Шаг | baseline_final | final |
|---|---|---|
| CS-01.1 | ✗ documents / not_found | ✓ documents / answered |
| CS-02.1 | ✓ documents / answered | ✓ documents / answered |
| CS-03.1 | ✗ documents / not_found | ✓ documents / answered |
| CS-04.1 | ✗ documents / not_found | ✓ escalation / escalated |
| CS-05.1 | ✓ registry / answered | ✓ registry / answered |
| CS-06.1 | ✓ registry / answered | ✓ registry / answered |
| CS-06.2 | ✓ registry / answered | ✓ registry / answered |
| CS-07.1 | ✓ registry / answered | ✓ registry / answered |
| CS-08.1 | ✗ documents / answered | ✓ registry / answered |
| CS-09.1 | ✗ documents / answered | ✓ registry / answered |
| CS-10.1 | ✓ smalltalk / smalltalk | ✓ smalltalk / smalltalk |
| CS-11.1 | ✗ documents / not_found | ✓ smalltalk / smalltalk |
| CS-12.1 | ✗ documents / answered | ✓ escalation / escalated |
| CS-13.1 | ✗ documents / answered | ✓ escalation / escalated |
| CS-14.1 | ✓ documents / degraded | ✓ documents / degraded |
| CS-15.1 | ✗ documents / error | ✓ escalation / escalated |
| CS-16.1 | ✓ rejected / rejected | ✓ rejected / rejected |
