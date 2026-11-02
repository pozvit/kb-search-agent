"""
Сборка текстовых карточек проектов и пропозалов из CSV в базу знаний.

Зачем нужен отдельный шаг. loader.py умеет читать .csv, но класть таблицы
в базу знаний напрямую бесполезно: чанкер режет текст по разделам и предложениям,
а строка таблицы — это не предложение. Вдобавок теги, команда и детальные подтеги
лежат в отдельных файлах, и по одной строке projects.csv нельзя ответить,
кто делал проект и какие у него подтеги.

Что делает:
  1. Собирает по одной карточке на проект, пропозал и проект учётной базы (лист checked),
     подтягивая теги, команду и обезличенное описание клиента из связанных CSV.
  2. Группирует карточки по годам в markdown-файлы. Каждая карточка — раздел с номером,
     поэтому при стратегии чанкинга `section` она становится отдельным чанком.
  3. Пишет cards_meta.jsonl: метаданные каждой карточки (тип, ID, название, годы,
     отрасли, типы услуг, клиент, статус). chunk_pipeline.py переносит их в metadata
     чанков, а оттуда они попадают в payload Qdrant и работают как фильтры поиска.

Что учтено:
  - нет вводной строки файла: раньше она становилась отдельным чанком-пустышкой;
  - карточки без содержимого (есть ID, нет названия, клиента и тегов) не пишутся;
  - нумерованные строки внутри текста («1. …» в Key findings) превращаются в «1) …»,
    иначе чанкер принимает их за заголовки новых карточек;
  - отрасли двух реестров сведены к одной классификации (INDUSTRY_TOP),
    исходные значения тоже сохраняются.

Лист checked (учётная база 2010–2018) по умолчанию исключён: состав людей в нём почти
не пересекается с реестрами компании, происхождение данных не подтверждено. Его карточки
собираются только с флагом --include-checked; при обычном запуске ранее собранные карточки
учётной базы и их метаданные удаляются.

Запуск:
    python tools/csv_to_cards.py                     # data/raw_csv -> data/raw_cards
    python tools/csv_to_cards.py --include-checked   # вместе с карточками листа checked
    python tools/csv_to_cards.py --out data/raw_kb   # в отдельную папку
    python tools/csv_to_cards.py --anonymized        # без реальных названий, фамилий, сумм и номеров договоров
    python tools/csv_to_cards.py --only projects     # только одна из баз
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

CARD_META_FILE = 'cards_meta.jsonl'

SCHEMES = {
    'projects': {
        'files': {
            'main': 'projects.csv',
            'tags_service': 'project_tags_service.csv',
            'tags_industry': 'project_tags_industry.csv',
            'team': 'project_team.csv',
        },
        'card_type': 'project',
        'id': 'ID Проекта',
        'title': 'Наименование проекта',
        'title_anon': 'Обезличенное наименование проекта',
        'client': 'Реальный клиент (краткое наименование)',
        'client_full': 'Клиент по договору (полное наименование)',
        'client_anon': 'Обезличенное наименование клиента',
        'year': 'Год проекта',
        'label': 'Проекты',
        'category': 'internal/projects',
        'file_prefix': 'projects',
    },
    'proposals': {
        'files': {
            'main': 'proposals.csv',
            'tags_service': 'proposal_tags_service.csv',
            'tags_industry': 'proposal_tags_industry.csv',
            'team': 'proposal_team.csv',
        },
        'card_type': 'proposal',
        'id': 'ID',
        'title': 'Наименование пропозала',
        'title_anon': 'Обезличенное наименование проекта',
        'client': 'Реальный клиент (краткое наименование)',
        'client_full': 'Клиент по договору (полное наименование)',
        'client_anon': 'Обезличенное наименование клиента',
        'year': 'Год пропозала',
        'label': 'Пропозалы',
        'category': 'internal/proposals',
        'file_prefix': 'proposals',
    },
    'checked': {
        'files': {'main': 'projects_checked.csv'},
        'card_type': 'project_accounting',
        'id': 'Код проекта',
        # юридическое название длинное (в среднем 117 символов) — в заголовок идёт обезличенное
        'title': 'Обезличенное наименование проекта',
        'title_anon': 'Обезличенное наименование проекта',
        'client': 'Клиент по договору (краткое наименование)',
        'client_full': 'Клиент по договору (полное наименование)',
        'client_anon': None,
        'year': 'Год проекта',
        'label': 'Проекты учётной базы',
        'category': 'internal/projects_accounting',
        'file_prefix': 'projects_accounting',
    },
}

# Поля основных таблиц, которые выводятся в карточку как есть: (заголовок CSV, подпись)
# Люди идут раньше компании: на вопрос «кто делал» модель отвечала названием компании-исполнителя,
# потому что это поле стояло в карточке первым.
PLAIN_FIELDS = [
    ('Команда проекта (из числа действующих сотрудников)', 'Команда (действующие сотрудники)'),
    ('Партнер/директор (из числа действующих сотрудников)', 'Партнёр / директор'),
    ('Имя сотрудника', 'Имя сотрудника'),
    ('Внутренний код AC', 'Внутренний код'),
    ('Где реализован (ADL | AC | consulting | industry)', 'Где реализован'),
    ('Компания, где реализован проект', 'Компания, где выполнен проект'),
    ('Наименование (юр.) проекта по Договору', 'Наименование по договору'),
    ('Примечания, Теги проекта в свободной форме', 'Примечания'),
    ('Примечания, Теги в свободной форме', 'Примечания'),
]

# Поля листа checked в порядке вывода: (заголовок CSV, подпись, многострочное)
CHECKED_FIELDS = [
    ('Код проекта', 'Код проекта', False),
    ('Тип', 'Тип', False),
    ('Статус проекта', 'Статус', False),
    ('Название (юр.) проекта по Договору', 'Наименование по договору', False),
    ('Стоимость с НДС, руб.', 'Стоимость с НДС', False),
    ('№ договора (бухгалтерия)', '№ договора', False),
    ('"Якорный" клиент для сводных таблиц (краткое наименование)', 'Якорный клиент', False),
    ('Ответственный партнер', 'Ответственный партнёр', False),
    ('Директор / руководитель проекта', 'Директор / руководитель проекта', False),
    ('Команда проекта', 'Команда', False),
    ('Описание проекта', 'Описание проекта', True),
    ('Key findings', 'Key findings', True),
    ('Примечания', 'Примечания', True),
]

# Поля, которые в обезличенном режиме не выводятся
SENSITIVE_FIELDS = {
    'Внутренний код AC', 'Компания, где реализован проект', 'Имя сотрудника',
    'Наименование (юр.) проекта по Договору',
    'Команда проекта (из числа действующих сотрудников)',
    'Партнер/директор (из числа действующих сотрудников)',
    'Код проекта', 'Стоимость с НДС, руб.', '№ договора (бухгалтерия)',
    '"Якорный" клиент для сводных таблиц (краткое наименование)',
    'Ответственный партнер', 'Директор / руководитель проекта', 'Команда проекта',
}

# Отрасли листа checked -> категории верхнего уровня основных реестров.
# Сопоставление экспертное; если категория не подходит — поправьте здесь.
INDUSTRY_TOP = {
    'Государственный и некоммерческий сектор': 'Public',
    'Наука и образование': 'Public',
    'Машиностроение и ОПК': 'Industrial goods',
    'Металлургия и горнодобывающая отрасль': 'Industrial goods',
    'Лесозаготовительная, деревообрабатывающая и целлюлозно-бумажная промышленность': 'Industrial goods',
    'Строительство и управление недвижимостью': 'Construction & Real estate',
    'Логистика, грузовые и пассажирские перевозки': 'Transportation',
    'Топливно-энергетический комплекс': 'Energy (O&G / chem / utilities)',
    'Химическая промышленность': 'Energy (O&G / chem / utilities)',
    'ЖКХ': 'Energy (O&G / chem / utilities)',
    'Потребительские товары': 'Consumer goods & retail',
    'Оптовая и розничная торговля': 'Consumer goods & retail',
    'Туризм, HoReCa, развлечения': 'Consumer goods & retail',
    'Сельское хозяйство и рыболовство': 'Consumer goods & retail',
    'Финансы': 'Financial services',
    'ИТ, медиа и телекоммуникации': 'Telecom & Media & IT',
    'Здравоохранение и фармацевтика': 'Pharma & HC',
}

# Типы услуг листа checked -> названия основных реестров
SERVICE_ALIASES = {
    'HR': 'Организационное развитие / HR',
    'Организационное развитие': 'Организационное развитие / HR',
    'Законодательство / регулирование': 'Законодательное регулирование',
}
SERVICE_JUNK = {'0', '-', '—', 'n/a', 'N.A.'}

HEADING_LIKE = re.compile(r'^(\s*)(\d{1,3}(?:\.\d{1,3})*)\.(\s+)', re.MULTILINE)
YEAR_RE = re.compile(r'(19|20)\d{2}')


# ------------------------------------------------------------------ утилиты

def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    # заголовки с хвостовыми пробелами («Статус проекта ») приводим к обрезанным
    return [{(k or '').strip(): v for k, v in row.items()} for row in rows]


def field(row: Dict[str, str], name: Optional[str]) -> str:
    if not name:
        return ''
    return re.sub(r'[ \t\u00a0]+', ' ', (row.get(name) or '')).strip()


def one_line(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def safe_multiline(text: str) -> str:
    """Сохраняет абзацы и списки, но убирает строки, похожие на заголовки разделов («1. …» -> «1) …»)."""
    text = (text or '').replace('\r', '\n').strip()
    text = HEADING_LIKE.sub(lambda m: f"{m.group(1)}{m.group(2)}){m.group(3)}", text)
    return re.sub(r'\n{2,}', '\n', text)


def split_values(text: str) -> List[str]:
    """«A; B» и значения на отдельных строках -> [A, B]. Запятые не делят: они есть внутри названий отраслей."""
    return [v.strip() for v in re.split(r'[;\n]', text or '') if v.strip()]


def parse_years(value: str) -> Tuple[Optional[int], Optional[int]]:
    found = [int(m.group(0)) for m in YEAR_RE.finditer(value or '')]
    if not found:
        return None, None
    return min(found), max(found)


def format_money(value: str) -> str:
    try:
        number = float(str(value).replace(' ', '').replace(',', '.'))
    except ValueError:
        return value
    whole, frac = divmod(round(number * 100), 100)
    text = f"{int(whole):,}".replace(',', ' ')
    return f"{text},{int(frac):02d} руб." if frac else f"{text} руб."


def group_by_id(rows: List[Dict[str, str]], key: str, value: str) -> Dict[str, List[str]]:
    """Сворачивает длинную таблицу «ID -> значение» в словарь списков без дублей."""
    result: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        ident, item = (row.get(key) or '').strip(), (row.get(value) or '').strip()
        if ident and item and item not in result[ident]:
            result[ident].append(item)
    return result


def index_tags(tag_rows: List[Dict[str, str]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """ID -> детальные теги и ID -> теги верхнего уровня (один проход вместо поиска по всей таблице на каждую карточку)."""
    detail: Dict[str, List[str]] = defaultdict(list)
    top: Dict[str, List[str]] = defaultdict(list)
    for r in tag_rows:
        ident, tag = (r.get('ID проекта') or '').strip(), (r.get('Тег') or '').strip()
        if not ident or not tag:
            continue
        bucket = top if r.get('Категория') == 'Верхний уровень' else detail
        if tag not in bucket[ident]:
            bucket[ident].append(tag)
    return detail, top


def normalize_services(values: List[str]) -> List[str]:
    result = []
    for v in values:
        v = SERVICE_ALIASES.get(v, v)
        if v and v not in SERVICE_JUNK and v not in result:
            result.append(v)
    return result


def industries(values: List[str], top_tags: List[str]) -> List[str]:
    """Категории верхнего уровня + исходные значения (русские отрасли checked), без дублей."""
    result: List[str] = []
    for v in list(top_tags) + values:
        for item in (INDUSTRY_TOP.get(v), v):
            if item and item not in result:
                result.append(item)
    return result


# ------------------------------------------------------------------ карточки

def build_card(row: Dict[str, str], scheme: Dict[str, Any], number: int,
               tags: Dict[str, Tuple[Dict[str, List[str]], Dict[str, List[str]]]],
               team: Dict[str, List[str]], anonymized: bool) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Возвращает (текст карточки, метаданные) или None, если карточка пустая."""
    kind_checked = scheme['card_type'] == 'project_accounting'
    ident = field(row, scheme['id'])
    title = one_line(field(row, scheme['title_anon'] if anonymized else scheme['title'])
                     or field(row, scheme['title']) or field(row, scheme['title_anon']))
    client_col = scheme['client_anon'] if anonymized and scheme['client_anon'] else scheme['client']
    client = one_line(field(row, client_col)) if not (anonymized and kind_checked) else ''
    year_raw = field(row, scheme['year'])
    year, year_end = parse_years(year_raw)

    service_detail, service_top = tags.get('service', ({}, {}))
    industry_detail, industry_top = tags.get('industry', ({}, {}))
    services = normalize_services(split_values(field(row, 'Тип услуги')))
    industry_list = industries(split_values(field(row, 'Отрасль')), industry_top.get(ident, []))

    if not title and not client and not services and not industry_list:
        return None      # ID без содержимого: такая карточка — мусор в выдаче

    head = [title or 'Без названия']
    if client:
        head.append(f"— {client}")
    if year:
        head.append(f"({year}–{year_end})" if year_end and year_end != year else f"({year})")
    lines = [f"{number}. {' '.join(head)}"]

    if not anonymized:
        full = one_line(field(row, scheme['client_full']))
        anon = one_line(field(row, scheme['client_anon']))
        if client:
            lines.append(f"Клиент: {client}" + (f" ({full})" if full and full != client else ""))
        if anon:
            lines.append(f"Обезличенное описание клиента: {anon}")
    elif client:
        lines.append(f"Клиент: {client}")

    start, end = one_line(field(row, 'Дата начала')), one_line(field(row, 'Дата окончания - факт'))
    if start or end:
        lines.append(f"Период: {start or '?'} — {end or '?'}")
    elif year:
        lines.append(f"Год: {year_raw}")

    if kind_checked:
        for col, caption, multiline in CHECKED_FIELDS:
            if anonymized and col in SENSITIVE_FIELDS:
                continue
            value = field(row, col) if not multiline else (row.get(col) or '')
            if not value.strip():
                continue
            if col == 'Стоимость с НДС, руб.':
                value = format_money(value)
            if col == 'Статус проекта':
                value = value.strip().capitalize()
            if multiline:
                lines.append(f"{caption}:\n{safe_multiline(value)}")
            else:
                lines.append(f"{caption}: {one_line(value)}")
    else:
        for col, caption in PLAIN_FIELDS:
            if col not in row or (anonymized and col in SENSITIVE_FIELDS):
                continue
            value = one_line(field(row, col))
            if value:
                lines.append(f"{caption}: {value}")

    if services:
        lines.append(f"Тип услуги: {'; '.join(services)}")
    if service_detail.get(ident):
        lines.append(f"Детальные теги услуг: {'; '.join(service_detail[ident])}")
    if industry_list:
        lines.append(f"Отрасль: {'; '.join(industry_list)}")
    if industry_detail.get(ident):
        lines.append(f"Детальные отраслевые теги: {'; '.join(industry_detail[ident])}")
    if not anonymized and team.get(ident):
        # таблица участников дублирует поле «Команда»; выводим её, только если в ней есть кто-то ещё
        team_text = field(row, 'Команда проекта (из числа действующих сотрудников)')
        extra = [p for p in team[ident] if p not in team_text]
        if extra:
            lines.append(f"Участники: {'; '.join(team[ident])}")

    status = field(row, 'Статус проекта').capitalize() if kind_checked else ''
    # люди и компания — для перечислений без поиска (list_cards читает cards_meta.jsonl напрямую)
    people_re = re.compile(r'[А-ЯЁA-Z][а-яёa-z]+(?:-[А-ЯЁA-Z][а-яёa-z]+)?\s+[А-ЯЁA-Z]\.(?:\s?[А-ЯЁA-Z]\.)?')
    team_text = field(row, 'Команда проекта (из числа действующих сотрудников)') or field(row, 'Команда проекта')
    team_people = people_re.findall(team_text) or split_values(team_text)
    for person in team.get(ident, []):
        if person not in team_people:
            team_people.append(person)
    meta_people = {} if anonymized else {
        'team': team_people,
        'partner': one_line(field(row, 'Партнер/директор (из числа действующих сотрудников)')
                            or field(row, 'Ответственный партнер')),
        'employee': one_line(field(row, 'Имя сотрудника')),
        'company': one_line(field(row, 'Компания, где реализован проект')),
    }
    meta = {
        'card_number': number,
        'card_type': scheme['card_type'],
        'card_id': ident,
        'card_title': title or 'Без названия',
        'year': year,
        'year_end': year_end or year,
        'industry': industry_list,
        'service': services,
        'service_detail': list(service_detail.get(ident, [])),
        'client': client,
        'status': status or None,
        **{k: v for k, v in meta_people.items() if v},
    }
    return '\n'.join(lines), meta


def build_documents(src: Path, out: Path, kind: str, anonymized: bool
                    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    scheme = SCHEMES[kind]
    files = scheme['files']
    rows = read_csv(src / files['main'])
    if not rows:
        print(f"  {files['main']}: не найден или пуст, пропущено")
        return [], [], 0

    tags = {}
    if 'tags_service' in files:
        tags['service'] = index_tags(read_csv(src / files['tags_service']))
        tags['industry'] = index_tags(read_csv(src / files['tags_industry']))
    team = group_by_id(read_csv(src / files['team']), 'ID проекта', 'Сотрудник') if 'team' in files else {}

    by_year: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        year = field(row, scheme['year']) or 'без_года'
        by_year[re.sub(r'[^\w\-]+', '_', year)].append(row)

    written, card_meta, skipped = [], [], 0
    out.mkdir(parents=True, exist_ok=True)
    suffix = '_anon' if anonymized else ''
    for year in sorted(by_year):
        name = f"{scheme['file_prefix']}_{year}{suffix}.md"
        cards = []
        for row in by_year[year]:
            built = build_card(row, scheme, len(cards) + 1, tags, team, anonymized)
            if built is None:
                skipped += 1
                continue
            text, meta = built
            cards.append(text)
            card_meta.append({'file_name': name, **meta})
        if not cards:
            continue
        # без вводной строки: она становилась отдельным чанком-пустышкой
        body = '\n\n'.join(cards) + '\n'
        (out / name).write_text(body, encoding='utf-8')
        title = f"{scheme['label']} {year.replace('_', ' ')}"
        written.append({'file_name': name, 'document_name': title, 'category': scheme['category'],
                        'version': year, 'count': len(cards)})
        print(f"  {name:36} {len(cards):4} карточек, {len(body):8} символов")
    if skipped:
        print(f"  пропущено пустых карточек: {skipped}")
    return written, card_meta, skipped


def update_metadata(meta_path: Path, entries: List[Dict[str, Any]]) -> None:
    """
    Дописывает записи в metadata.yaml папки назначения, не трогая существующие.
    Без sidecar structurer.py проставит document_name = имя файла и category = uncategorized.
    """
    data: Dict[str, Any] = {}
    if meta_path.exists():
        data = yaml.safe_load(meta_path.read_text(encoding='utf-8')) or {}
    documents = data.setdefault('documents', {})
    added = 0
    for entry in entries:
        if entry['file_name'] in documents:
            continue
        documents[entry['file_name']] = {
            'document_name': entry['document_name'],
            'category': entry['category'],
            'version': entry['version'],
            'source_type': 'internal_registry',
        }
        added += 1
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=120), encoding='utf-8')
    print(f"  metadata.yaml: добавлено записей {added}, всего {len(documents)}")


def prune_metadata(meta_path: Path, names: set) -> None:
    """Убирает из metadata.yaml записи об удалённых файлах карточек."""
    if not meta_path.exists():
        return
    data = yaml.safe_load(meta_path.read_text(encoding='utf-8')) or {}
    documents = data.get('documents', {})
    for name in names:
        documents.pop(name, None)
    meta_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=120), encoding='utf-8')


def file_kind(file_name: str) -> Optional[str]:
    """К какой базе относится файл карточек. Длинный префикс проверяется первым:
    projects_accounting_2018.md — это учётная база, а не основная база проектов."""
    for kind in sorted(SCHEMES, key=lambda k: -len(SCHEMES[k]['file_prefix'])):
        if file_name.startswith(SCHEMES[kind]['file_prefix'] + '_'):
            return kind
    return None


def write_card_meta(path: Path, entries: List[Dict[str, Any]], rebuilt_kinds: set) -> None:
    """Перезаписывает метаданные пересобранных баз, сохраняя записи остальных (для --only)."""
    kept = []
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                item = json.loads(line)
                if file_kind(item['file_name']) not in rebuilt_kinds:
                    kept.append(item)
    with open(path, 'w', encoding='utf-8') as f:
        for item in kept + entries:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  {path.name}: {len(kept) + len(entries)} карточек с метаданными")


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка карточек проектов из CSV")
    parser.add_argument('--src', default='data/raw_csv', help='папка с CSV')
    parser.add_argument('--out', default='data/raw_cards', help='куда класть карточки')
    parser.add_argument('--only', choices=sorted(SCHEMES), help='собрать только одну из баз')
    parser.add_argument('--include-checked', action='store_true',
                        help='собрать и карточки листа checked (по умолчанию исключён из базы)')
    parser.add_argument('--anonymized', action='store_true',
                        help='без реальных названий клиентов, фамилий, сумм и номеров договоров')
    parser.add_argument('--no-metadata', action='store_true',
                        help='не дописывать metadata.yaml в папке назначения')
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.exists():
        raise SystemExit(f"Не найдена папка {src}. Сначала: python tools/xlsb_to_csv.py")

    if args.only:
        kinds = [args.only]
    else:
        kinds = [k for k in SCHEMES if k != 'checked' or args.include_checked]
    excluded = set() if (args.include_checked or args.only == 'checked') else {'checked'}
    entries, card_meta, skipped = [], [], 0
    for kind in kinds:
        print(f"{SCHEMES[kind]['label']}:")
        e, m, s = build_documents(src, out, kind, args.anonymized)
        entries += e
        card_meta += m
        skipped += s

    # карточки исключённого листа checked убираем вместе с метаданными, даже если их собирали раньше
    removed = [f for f in out.glob('*.md') if file_kind(f.name) in excluded] if out.exists() else []
    for f in removed:
        f.unlink()
    if removed:
        print(f"  удалено карточек листа checked: {len(removed)} файлов (вернуть — --include-checked)")
        prune_metadata(out / 'metadata.yaml', {f.name for f in removed})

    if entries or removed:
        # старые файлы пересобранных баз удаляем, иначе устаревшие карточки попадут в индекс
        fresh = {e['file_name'] for e in entries}
        for old in out.glob('*.md'):
            if file_kind(old.name) in kinds and old.name not in fresh:
                old.unlink()
        write_card_meta(out / CARD_META_FILE, card_meta, set(kinds) | excluded)
        if not args.no_metadata and entries:
            update_metadata(out / 'metadata.yaml', entries)

    total = sum(e['count'] for e in entries)
    print(f"\nГотово: {len(entries)} файлов, {total} карточек в {out}/ (пустых пропущено: {skipped})")
    print("Дальше: python src/build_index.py --profile full")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
