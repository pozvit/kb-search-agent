"""
Сборка текстовых карточек проектов и пропозалов из CSV в базу знаний.

Зачем нужен отдельный шаг. loader.py умеет читать .csv, но класть таблицы
в базу знаний напрямую бесполезно: чанкер режет текст по разделам и предложениям,
а строка таблицы — это не предложение. Вдобавок теги, команда и детальные подтеги
лежат в отдельных файлах, и по одной строке projects.csv нельзя ответить,
кто делал проект и какие у него подтеги.

Что делает: собирает по одному связному описанию на каждый проект и пропозал,
подтягивая теги, команду и обезличенное описание клиента из связанных CSV,
и группирует описания по годам в markdown-файлы. Каждый проект оформляется
разделом с номером-ID, поэтому при стратегии чанкинга `section` одна карточка
становится ровно одним чанком и не смешивается с соседями.

Запуск:
    python tools/csv_to_cards.py                     # data/raw_csv -> data/raw
    python tools/csv_to_cards.py --out data/raw_kb   # в отдельную папку
    python tools/csv_to_cards.py --anonymized        # без реальных названий и фамилий
    python tools/csv_to_cards.py --only projects     # только проекты
"""

import argparse
import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Как называются поля в двух базах. Они близки, но не совпадают: у пропозалов
# нет дат начала и окончания, а роль партнёра названа иначе.
SCHEMES = {
    'projects': {
        'files': {
            'main': 'projects.csv',
            'tags_service': 'project_tags_service.csv',
            'tags_industry': 'project_tags_industry.csv',
            'team': 'project_team.csv',
        },
        'id': 'ID Проекта',
        'title': 'Наименование проекта',
        'title_anon': 'Обезличенное наименование проекта',
        'client': 'Реальный клиент (краткое наименование)',
        'client_full': 'Клиент по договору (полное наименование)',
        'client_anon': 'Обезличенное наименование клиента',
        'year': 'Год проекта',
        'label': 'Проекты',
        'category': 'internal/projects',
        'noun': 'проект',
    },
    'proposals': {
        'files': {
            'main': 'proposals.csv',
            'tags_service': 'proposal_tags_service.csv',
            'tags_industry': 'proposal_tags_industry.csv',
            'team': 'proposal_team.csv',
        },
        'id': 'ID',
        'title': 'Наименование пропозала',
        'title_anon': 'Обезличенное наименование проекта',
        'client': 'Реальный клиент (краткое наименование)',
        'client_full': 'Клиент по договору (полное наименование)',
        'client_anon': 'Обезличенное наименование клиента',
        'year': 'Год пропозала',
        'label': 'Пропозалы',
        'category': 'internal/proposals',
        'noun': 'пропозал',
    },
}

# Поля основной таблицы, которые выводятся в карточку как есть.
# Ключ — заголовок в CSV, значение — подпись в карточке. Порядок сохраняется.
PLAIN_FIELDS = [
    ('Внутренний код AC', 'Внутренний код'),
    ('Где реализован (ADL | AC | consulting | industry)', 'Где реализован'),
    ('Компания, где реализован проект', 'Компания-исполнитель'),
    ('Наименование (юр.) проекта по Договору', 'Наименование по договору'),
    ('Команда проекта (из числа действующих сотрудников)', 'Команда'),
    ('Партнер/директор (из числа действующих сотрудников)', 'Партнёр / директор'),
    ('Примечания, Теги проекта в свободной форме', 'Примечания'),
    ('Примечания, Теги в свободной форме', 'Примечания'),
]

# Поля, которые в обезличенном режиме не выводятся вовсе
SENSITIVE_FIELDS = {
    'Внутренний код AC', 'Компания, где реализован проект',
    'Наименование (юр.) проекта по Договору',
    'Команда проекта (из числа действующих сотрудников)',
    'Партнер/директор (из числа действующих сотрудников)',
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=';'))


def group_by_id(rows: List[Dict[str, str]], key: str, value: str) -> Dict[str, List[str]]:
    """Сворачивает длинную таблицу «ID -> значение» в словарь списков без дублей."""
    result: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        ident, item = (row.get(key) or '').strip(), (row.get(value) or '').strip()
        if ident and item and item not in result[ident]:
            result[ident].append(item)
    return result


def detail_tags(tag_rows: List[Dict[str, str]], ident: str) -> List[str]:
    """
    Возвращает только детальные теги, без верхнего уровня.

    Теги верхнего уровня дублируют колонку «Тип услуги» основной таблицы,
    поэтому в карточке они были бы повторены дважды.
    """
    return [r['Тег'] for r in tag_rows
            if r.get('ID проекта') == ident and r.get('Категория') != 'Верхний уровень']


def build_card(row: Dict[str, str], scheme: Dict[str, Any], number: int,
               tags_service: List[Dict[str, str]], tags_industry: List[Dict[str, str]],
               team: Dict[str, List[str]], anonymized: bool) -> str:
    ident = (row.get(scheme['id']) or '').strip()
    title = (row.get(scheme['title_anon'] if anonymized else scheme['title'])
             or row.get(scheme['title']) or 'Без названия').strip()
    client = (row.get(scheme['client_anon'] if anonymized else scheme['client'])
              or '').strip()
    year = (row.get(scheme['year']) or '').strip()

    head_parts = [title]
    if client:
        head_parts.append(f"— {client}")
    if year:
        head_parts.append(f"({year})")
    lines = [f"{number}. {' '.join(head_parts)}"]

    if not anonymized:
        full = (row.get(scheme['client_full']) or '').strip()
        anon = (row.get(scheme['client_anon']) or '').strip()
        if client:
            lines.append(f"Клиент: {client}" + (f" ({full})" if full else ""))
        if anon:
            lines.append(f"Обезличенное описание клиента: {anon}")
    else:
        if client:
            lines.append(f"Клиент: {client}")

    start = (row.get('Дата начала') or '').strip()
    end = (row.get('Дата окончания - факт') or '').strip()
    if start or end:
        lines.append(f"Период: {start or '?'} — {end or '?'}")
    elif year:
        lines.append(f"Год: {year}")

    for field, caption in PLAIN_FIELDS:
        if field not in row:
            continue
        if anonymized and field in SENSITIVE_FIELDS:
            continue
        value = (row.get(field) or '').strip()
        if value:
            lines.append(f"{caption}: {value}")

    service = (row.get('Тип услуги') or '').strip()
    if service:
        lines.append(f"Тип услуги: {service}")
    detailed = detail_tags(tags_service, ident)
    if detailed:
        lines.append(f"Детальные теги услуг: {'; '.join(detailed)}")

    industry = (row.get('Отрасль') or '').strip()
    if industry:
        lines.append(f"Отрасль: {industry}")
    detailed_ind = detail_tags(tags_industry, ident)
    if detailed_ind:
        lines.append(f"Детальные отраслевые теги: {'; '.join(detailed_ind)}")

    if not anonymized and team.get(ident):
        lines.append(f"Участники: {'; '.join(team[ident])}")

    return '\n'.join(lines)


def build_documents(src: Path, out: Path, kind: str, anonymized: bool) -> List[Dict[str, str]]:
    scheme = SCHEMES[kind]
    files = scheme['files']
    rows = read_csv(src / files['main'])
    if not rows:
        print(f"  {files['main']}: не найден или пуст, пропущено")
        return []

    tags_service = read_csv(src / files['tags_service'])
    tags_industry = read_csv(src / files['tags_industry'])
    team = group_by_id(read_csv(src / files['team']), 'ID проекта', 'Сотрудник')

    by_year: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        year = (row.get(scheme['year']) or '').strip() or 'без_года'
        by_year[year].append(row)

    written: List[Dict[str, str]] = []
    out.mkdir(parents=True, exist_ok=True)
    for year in sorted(by_year):
        cards = []
        # нумерация сквозная внутри файла: чанкер опознаёт заголовок по номеру
        # в начале строки, а ID проекта может доходить до трёх знаков
        for number, row in enumerate(by_year[year], start=1):
            cards.append(build_card(row, scheme, number, tags_service,
                                    tags_industry, team, anonymized))

        suffix = '_anon' if anonymized else ''
        name = f"{kind}_{year}{suffix}.md"
        title = f"{scheme['label']} {year}"
        body = f"{title}\nВсего {scheme['noun']}ов в разделе: {len(cards)}\n\n" + '\n\n'.join(cards) + '\n'
        (out / name).write_text(body, encoding='utf-8')
        written.append({
            'file_name': name,
            'document_name': title,
            'category': scheme['category'],
            'version': year,
            'count': len(cards),
        })
        print(f"  {name:28} {len(cards):4} карточек, {len(body):7} символов")
    return written


def update_metadata(meta_path: Path, entries: List[Dict[str, str]]) -> None:
    """
    Дописывает записи в data/raw/metadata.yaml, не трогая существующие.

    Без sidecar structurer.py проставит document_name = имя файла,
    category = uncategorized, и метаданные перестанут доходить до payload Qdrant.
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
    meta_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=120),
        encoding='utf-8')
    print(f"  metadata.yaml: добавлено записей {added}, всего {len(documents)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка карточек проектов из CSV")
    parser.add_argument('--src', default='data/raw_csv', help='папка с CSV')
    # по умолчанию отдельная папка: data/raw занята базой знаний из 5 PDF,
    # под которую написаны config/questions.yaml и все прогоны в results/
    parser.add_argument('--out', default='data/raw_cards', help='куда класть карточки')
    parser.add_argument('--only', choices=['projects', 'proposals'],
                        help='собрать только одну из баз')
    parser.add_argument('--anonymized', action='store_true',
                        help='без реальных названий клиентов, фамилий и кодов договоров')
    parser.add_argument('--no-metadata', action='store_true',
                        help='не дописывать metadata.yaml в папке назначения')
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.exists():
        raise SystemExit(f"Не найдена папка {src}. Сначала: python tools/xlsb_to_csv.py")

    kinds = [args.only] if args.only else ['projects', 'proposals']
    entries: List[Dict[str, str]] = []
    for kind in kinds:
        print(f"{SCHEMES[kind]['label']}:")
        entries.extend(build_documents(src, out, kind, args.anonymized))

    if entries and not args.no_metadata:
        update_metadata(out / 'metadata.yaml', entries)

    total = sum(e['count'] for e in entries)
    print(f"\nГотово: {len(entries)} файлов, {total} карточек в {out}/")
    print("Чтобы проиндексировать: укажите эту папку в config/config.yaml -> paths.raw_dir,")
    print("поставьте merge_short_sections: false в config/chunking.yaml,")
    print("удалите data/qdrant_storage/ и прогоните pipeline -> chunk -> embed -> vector.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
