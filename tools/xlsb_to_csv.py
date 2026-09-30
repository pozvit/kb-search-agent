"""
Конвертация баз проектов и пропозалов из .xlsb в CSV.

Зачем отдельный инструмент: .xlsb — бинарный формат Excel, его не читает ни pandas
без движка, ни loader.py проекта. Плюс данные в книгах лежат не плоско: основные
таблицы соседствуют с матрицами тегов «проект × тег», где значения ячеек — служебные
ссылки для формул поиска, а не данные. Прямой экспорт «лист в CSV» дал бы сотни
столбцов формульного мусора.

Что делает:
  1. Основные таблицы (проекты, пропозалы) — как есть, с приведением дат и ID.
  2. Матрицы тегов разворачивает в длинный формат: одна строка = один тег проекта.
     Это сохраняет детальные подтеги, которых нет в основной таблице: там только
     8 категорий услуг, а в матрице 69 конкретных тегов.
  3. Матрицу участия сотрудников разворачивает так же: строка = сотрудник в проекте.
     Сохраняет всех участников, включая уже не работающих в компании.
  4. Справочник клиентов и таблицу наличия материалов — как есть.
  5. Лист checked (учётная база проектов с описаниями и Key findings) — целиком,
     с сохранением переносов строк в текстовых полях. Это единственный источник
     содержательного текста о проектах: в основной таблице есть только названия и теги.

Запуск:
    python tools/xlsb_to_csv.py                       # из data/raw_xlsb в data/raw_csv
    python tools/xlsb_to_csv.py --src X --out Y
    python tools/xlsb_to_csv.py --skip-financial      # checked без стоимости и номеров договоров

Требуется: pip install pyxlsb
"""

import argparse
import csv
import datetime as dt
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyxlsb import open_workbook

# Excel хранит даты как число дней от 1899-12-30 (с известной ошибкой 1900 года,
# которую эта база не задевает — все даты после 1970).
EXCEL_EPOCH = dt.date(1899, 12, 30)

# Заголовки, по которым определяется конец блока тегов в матрице «проект × тег».
# Всё, что левее первого такого столбца, — теги; правее — служебные колонки формул.
MATRIX_ANCHORS = {'Партнер', 'Директор/РП', 'Клиент', 'Наименование проекта',
                  'ID проекта', 'Отрасль', 'Тип услуги'}

# Столбцы основных таблиц, которые нужно трактовать как даты
DATE_HEADERS = {'Дата начала', 'Дата окончания - факт'}

# Текстовые столбцы листа checked, в которых сохраняются переносы строк
# (абзацы и списки в описаниях и Key findings)
CHECKED_MULTILINE = {'Описание проекта', 'Key findings', 'Примечания'}

# Столбцы, которые убирает --skip-financial
FINANCIAL_COLUMNS = {'Стоимость с НДС, руб.', '№ договора (бухгалтерия)'}


def excel_date(value: Any) -> str:
    """Excel-серийная дата -> ISO. Нечисловое и явно нереальное возвращает как есть."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return '' if value is None else str(value).strip()
    if not 1 <= num <= 100000:
        return str(value)
    return (EXCEL_EPOCH + dt.timedelta(days=int(num))).isoformat()


def clean_cell(value: Any) -> str:
    """
    Приводит значение к строке для CSV.

    pyxlsb отдаёт все числа как float, поэтому целые ID превращаются в «1.0».
    Целые возвращаем без дробной части, иначе ID и годы будут нечитаемы.
    """
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'да' if value else 'нет'
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return re.sub(r'\s+', ' ', str(value)).strip()


def clean_multiline(value: Any) -> str:
    """Как clean_cell, но сохраняет переносы строк: пробелы схлопываются внутри строки."""
    if not isinstance(value, str):
        return clean_cell(value)
    lines = [re.sub(r'[ \t\u00a0]+', ' ', line).strip() for line in value.replace('\r', '\n').split('\n')]
    text = '\n'.join(lines)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def clean_tags(value: Any) -> str:
    """
    Нормализует список тегов «Стратегия; Маркетинг и продажи; » -> через «; »
    без висящего разделителя. Такие хвосты мешают и поиску, и чтению.
    """
    text = clean_cell(value)
    parts = [p.strip() for p in text.split(';') if p.strip()]
    return '; '.join(parts)


def read_rows(path: Path, sheet: str, limit: Optional[int] = None) -> List[Dict[int, Any]]:
    """Читает лист как список разреженных строк {индекс столбца: значение}."""
    rows: List[Dict[int, Any]] = []
    with open_workbook(str(path)) as wb:
        with wb.get_sheet(sheet) as sh:
            for i, row in enumerate(sh.rows()):
                if limit is not None and i >= limit:
                    break
                rows.append({c.c: c.v for c in row})
    return rows


def dense(row: Dict[int, Any], width: int) -> List[Any]:
    return [row.get(i) for i in range(width)]


def write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Excel на Windows иначе показывает кириллицу как крякозябры,
    # а Python читает такой файл без дополнительных настроек.
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {path.name:34} {len(rows):5} строк, {len(header):3} колонок")


# ---------------------------------------------------------------- основные таблицы

def export_main_table(path: Path, sheet: str, out: Path, name: str) -> List[str]:
    """
    Экспортирует основную таблицу. Строка 0 — технические пояснения автора книги,
    строка 1 — заголовки, данные с строки 2.
    """
    rows = read_rows(path, sheet)
    header_row = rows[1]
    width = max(header_row) + 1
    headers = [clean_cell(h) or f'col_{i}' for i, h in enumerate(dense(header_row, width))]

    date_idx = {i for i, h in enumerate(headers) if h in DATE_HEADERS}
    tag_idx = {i for i, h in enumerate(headers)
               if h in ('Тип услуги', 'Отрасль') or h.startswith(('Команда проекта', 'Партнер/директор'))}

    data: List[List[str]] = []
    for row in rows[2:]:
        values = dense(row, width)
        if not any(v not in (None, '') for v in values):
            continue
        record = []
        for i, v in enumerate(values):
            if i in date_idx:
                record.append(excel_date(v))
            elif i in tag_idx:
                record.append(clean_tags(v))
            else:
                record.append(clean_cell(v))
        # строка считается данными, только если заполнен ID
        if record and record[0]:
            data.append(record)

    write_csv(out / f'{name}.csv', headers, data)
    return headers


# ------------------------------------------------------------------ матрицы тегов

def export_tag_matrix(path: Path, sheet: str, out: Path, name: str,
                      tag_kind: str) -> None:
    """
    Разворачивает матрицу «проект × тег» в длинный формат.

    Раскладка листа: строка 0 — категории (заполнена разреженно, над первым тегом
    каждой группы), строка 1 — названия тегов, данные с строки 3. Непустая ячейка
    означает, что тег применён; само значение — служебная ссылка для формул поиска
    и смысловой нагрузки не несёт.
    """
    rows = read_rows(path, sheet)
    cat_row, tag_row = rows[0], rows[1]

    # граница блока тегов: первый столбец со «служебным» заголовком или числом
    limit = max(tag_row) + 1
    for col in sorted(tag_row):
        value = clean_cell(tag_row[col])
        if value in MATRIX_ANCHORS or re.fullmatch(r'\d+', value):
            limit = col
            break

    tags: Dict[int, str] = {}
    for col in range(limit):
        value = clean_cell(tag_row.get(col))
        if value and value != '-':
            tags[col] = value

    # категория: протягиваем значение строки 0 вправо до следующей заполненной
    categories: Dict[int, str] = {}
    current = ''
    for col in range(limit):
        head = clean_cell(cat_row.get(col))
        if head and head not in ('Тип услуги', 'Отрасли'):
            current = head
        # теги верхнего уровня стоят до первой категории — помечаем явно,
        # иначе колонка «Категория» у них остаётся пустой
        categories[col] = current or 'Верхний уровень'

    # столбец с ID проекта ищем по заголовку, а не по фиксированному номеру:
    # в двух листах он стоит на разных позициях
    id_col = next((c for c in sorted(tag_row)
                   if clean_cell(tag_row[c]) in ('ID проекта', 'ID')), None)
    if id_col is None:
        raise SystemExit(f"{sheet}: не найден столбец «ID проекта»")

    data: List[List[str]] = []
    for row in rows[3:]:
        project_id = clean_cell(row.get(id_col))
        if not project_id:
            continue
        for col, tag in tags.items():
            if clean_cell(row.get(col)):
                data.append([project_id, tag_kind, categories.get(col, ''), tag])

    write_csv(out / f'{name}.csv',
              ['ID проекта', 'Вид тега', 'Категория', 'Тег'], data)


# --------------------------------------------------------------- участие в проектах

def export_team_matrix(path: Path, sheet: str, out: Path, name: str) -> None:
    """
    Разворачивает матрицу «проект × сотрудник».

    Раскладка: строка 0 — флаг «в штате?», строка 1 — фамилии сотрудников,
    строка 2 — заголовки описательных столбцов, данные с строки 3.

    Матрица даёт участие по всем проектам, тогда как колонка «Команда проекта»
    в основной таблице заполнена только для части записей. В этой книге все
    перечисленные сотрудники помечены как действующие, поэтому «В штате»
    выходит однородным — столбец сохранён на случай появления уволенных.
    Отдельно выгружается employees.csv: там есть и сотрудники без проектов.
    """
    rows = read_rows(path, sheet)
    staff_row, name_row, header_row = rows[0], rows[1], rows[2]

    # Правее блока отметок лежит зеркальный блок формул, где те же фамилии
    # записаны с висящим разделителем («Иванов И.; ») — из него собирается
    # строка «Действующая команда». Обрезаем блок на первом таком столбце,
    # иначе в выгрузку попадут фантомные сотрудники.
    people: Dict[int, str] = {}
    for col in sorted(name_row):
        value = clean_cell(name_row[col])
        if not value:
            continue
        if ';' in value:
            break
        people[col] = value
    in_staff = {c: clean_cell(staff_row.get(c)) for c in people}

    id_col = next((c for c in sorted(header_row)
                   if clean_cell(header_row[c]) in ('ID проекта', 'ID')), None)
    if id_col is None:
        raise SystemExit(f"{sheet}: не найден столбец «ID проекта»")

    data: List[List[str]] = []
    for row in rows[3:]:
        project_id = clean_cell(row.get(id_col))
        if not project_id:
            continue
        for col, person in people.items():
            if clean_cell(row.get(col)):
                data.append([project_id, person, 'да' if in_staff.get(col) else 'нет'])

    write_csv(out / f'{name}.csv',
              ['ID проекта', 'Сотрудник', 'В штате'], data)

    # список сотрудников с числом проектов: в длинной таблице не видно тех,
    # у кого проектов нет вовсе, а для базы знаний это тоже факт
    counts: Dict[str, int] = {p: 0 for p in people.values()}
    for row in data:
        counts[row[1]] = counts.get(row[1], 0) + 1
    roster = [[person, 'да' if in_staff.get(col) else 'нет', str(counts.get(person, 0))]
              for col, person in sorted(people.items(), key=lambda kv: kv[1])]
    write_csv(out / f'{name.split("_")[0]}_employees.csv',
              ['Сотрудник', 'В штате', 'Проектов в базе'], roster)


# ------------------------------------------------------------------ простые листы

def export_flat(path: Path, sheet: str, out: Path, name: str,
                header_row_idx: int = 0, date_cols: Optional[set] = None,
                multiline_cols: Optional[set] = None, drop_cols: Optional[set] = None) -> None:
    """
    Экспортирует лист, у которого шапка на известной строке, без матриц.

    multiline_cols — заголовки столбцов, где сохраняются переносы строк;
    drop_cols — заголовки столбцов, которые не выгружаются.
    """
    rows = read_rows(path, sheet)
    if header_row_idx >= len(rows):
        print(f"  {sheet}: лист пуст, пропущен")
        return
    header_row = rows[header_row_idx]
    width = max(header_row) + 1
    headers = [clean_cell(h) or f'col_{i}' for i, h in enumerate(dense(header_row, width))]

    # у некоторых листов первый столбец пустой (визуальный отступ) — убираем
    offset = 1 if not headers[0] else 0
    headers = headers[offset:]
    dates = {i for i, h in enumerate(headers) if h in DATE_HEADERS} | (date_cols or set())
    multiline = {i for i, h in enumerate(headers) if h in (multiline_cols or set())}
    keep = [i for i, h in enumerate(headers) if h not in (drop_cols or set())]

    data: List[List[str]] = []
    for row in rows[header_row_idx + 1:]:
        values = dense(row, width)[offset:]
        record = [excel_date(v) if i in dates else clean_multiline(v) if i in multiline else clean_cell(v)
                  for i, v in enumerate(values)]
        if any(record):
            data.append([record[i] for i in keep])

    write_csv(out / f'{name}.csv', [headers[i] for i in keep], data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Конвертация .xlsb баз в CSV")
    parser.add_argument('--src', default='data/raw_xlsb', help='папка с .xlsb')
    parser.add_argument('--out', default='data/raw_csv', help='папка для CSV')
    parser.add_argument('--skip-financial', action='store_true',
                        help='выгрузить лист checked без стоимости и номеров договоров')
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    projects = src / 'База_проектов_AC.xlsb'
    proposals = src / 'База_пропозалов_AC.xlsb'
    for f in (projects, proposals):
        if not f.exists():
            raise SystemExit(f"Не найден {f}")

    print("База проектов:")
    export_main_table(projects, 'Основная база проектов', out, 'projects')
    export_tag_matrix(projects, 'Ф.теги проектов', out, 'project_tags_service', 'услуга')
    export_tag_matrix(projects, 'Отрас. теги проектов', out, 'project_tags_industry', 'отрасль')
    export_team_matrix(projects, 'Опыт сотрудников', out, 'project_team')
    export_flat(projects, 'Материалы', out, 'project_materials', header_row_idx=1)
    export_flat(projects, 'Клиенты_upd', out, 'clients', header_row_idx=0)
    export_flat(projects, 'checked', out, 'projects_checked', header_row_idx=0,
                multiline_cols=CHECKED_MULTILINE,
                drop_cols=FINANCIAL_COLUMNS if args.skip_financial else None)

    print("База пропозалов:")
    export_main_table(proposals, 'Основная база', out, 'proposals')
    export_tag_matrix(proposals, 'Ф.теги пропозалов', out, 'proposal_tags_service', 'услуга')
    export_tag_matrix(proposals, 'Отрас. теги пропозалов', out, 'proposal_tags_industry', 'отрасль')
    export_team_matrix(proposals, 'Опыт сотрудников', out, 'proposal_team')

    print(f"\nГотово. CSV в {out}/ — разделитель «;», кодировка utf-8-sig.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
