"""
Конвертация презентаций (.pptx) и документов (.docx) в обезличенный текст для базы знаний.

Цепочка целиком:
    .pptx / .docx  ->  tools/office_to_kb.py  ->  .md  ->  tools/md_to_pdf.py  ->  .pdf

То есть на выходе получается ровно тот же вид, что у пяти продуктовых документов
в data/raw/: плоский текст с нумерованными разделами, без графики, контактов
и финансовых данных.

Что делает:
  1. Извлекает текст. Из .pptx — заголовки и содержимое слайдов, таблицы, заметки
     докладчика. Из .docx — абзацы с учётом уровня заголовков, таблицы, колонтитулы.
  2. Восстанавливает структуру: слайд или раздел документа становится нумерованным
     разделом «N. Заголовок», что нужно для стратегии чанкинга `section`.
  3. Вычищает контакты и финансовые данные по набору правил для русского
     и английского языков.
  4. Помечает подозрительные строки, которые правилами не ловятся, и выносит их
     в отчёт для ручной проверки.

ВАЖНО про обезличивание. Автоматическая вычистка ненадёжна по своей природе:
регулярные выражения находят телефоны, почту, суммы с валютой и реквизиты,
но не находят «стоимость составила три миллиона», фамилию в середине абзаца
или название клиента в заголовке. Поэтому инструмент не заявляет, что результат
чист, а формирует отчёт _redaction_report.md со списком того, что удалено,
и того, что требует глазами. Публиковать выгрузку без просмотра этого отчёта нельзя.

Запуск:
    python tools/office_to_kb.py                        # data/raw_office -> data/source_md_office
    python tools/office_to_kb.py --names data/raw_csv/project_employees.csv
    python tools/office_to_kb.py --keep-notes           # включать заметки докладчика
    python tools/office_to_kb.py --strict               # удалять и подозрительные строки

Требуется: pip install python-docx python-pptx
"""

import argparse
import csv
import datetime as dt
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

try:
    from docx import Document
except ImportError:
    Document = None
try:
    from pptx import Presentation
except ImportError:
    Presentation = None

REDACTED = '[удалено]'

# Нумерованный пункт в начале строки: «1.», «2.1.», «10.» — ровно то, что
# chunker.split_sections считает заголовком раздела. Внутри слайда такие строки
# обычные буллеты, и без нейтрализации каждый пункт становится отдельным чанком:
# на реальных презентациях это давало 188 фрагментов на документ вместо одного
# на слайд. Заменяем точку на скобку — смысл сохраняется, шаблон заголовка ломается.
BODY_NUMBER_RE = re.compile(r'^(\d{1,3}(?:\.\d{1,3})*)\.(?=\s)')

# ---------------------------------------------------------------- правила вычистки
# Каждое правило: (метка для отчёта, регулярное выражение).
# Порядок важен: более специфичные идут раньше, иначе широкое правило
# съест кусок строки и специфичное уже не сработает.

CONTACT_RULES: List[Tuple[str, re.Pattern]] = [
    ('email', re.compile(r'[\w.+-]+@[\w-]+\.[\w.]{2,}', re.I)),
    ('ссылка', re.compile(r'\b(?:https?://|www\.)\S+', re.I)),
    # телефоны: +7 (495) 123-45-67, 8-800-555-35-35, +44 20 7123 4567
    ('телефон', re.compile(
        r'(?<!\d)(?:\+?\d{1,3}[\s\-(]*)?(?:\d{2,5}[\s\-)]*){2,4}\d{2,4}(?!\d)')),
    ('мессенджер', re.compile(
        r'\b(?:telegram|whatsapp|skype|linkedin|телеграм|вотсап|скайп)\s*:?\s*@?\S+', re.I)),
]

FINANCE_RULES: List[Tuple[str, re.Pattern]] = [
    # реквизиты — до денежных правил, иначе номер счёта распознается как сумма
    ('банковские реквизиты', re.compile(
        r'\b(?:р/?с|к/?с|расч[её]тный\s+сч[её]т|корр\.?\s*сч[её]т|БИК|ИНН|КПП|ОГРН|ОКПО'
        r'|IBAN|SWIFT|BIC|account\s+no)\b[\s:№]*[\w\-/]*', re.I)),
    ('идентификатор (ИИН/БИН)', re.compile(r'\b(?:ИИН|БИН)\b[\s:№]*\d{0,12}', re.I)),
    ('номер карты', re.compile(r'(?<!\d)(?:\d{4}[\s-]?){3}\d{4}(?!\d)')),
    # суммы с валютой: 1 200 000 руб., $1.5M, EUR 300k, 45 млн ₽
    ('сумма', re.compile(
        r'(?<![\w])(?:[$€£₽]\s?)?\d[\d\s\u00a0.,]*\s*'
        r'(?:тыс\.?|млн\.?|млрд\.?|тысяч\w*|миллион\w*|миллиард\w*|k|m|bn|mln)?\s*'
        r'(?:руб\w*|₽|RUB|USD|EUR|KZT|тенге|долл\w*|евро|\$|€|£)\.?', re.I)),
    ('сумма (валюта впереди)', re.compile(
        r'(?:[$€£₽]|USD|EUR|RUB|KZT)\s?\d[\d\s\u00a0.,]*\s*(?:k|m|bn|mln|млн|тыс\.?)?', re.I)),
    ('ставка/тариф', re.compile(
        r'\b(?:man-?day|человеко-?день|day\s*rate|ставка)\b[^.\n]{0,40}', re.I)),
]

# Даты и проценты защищаются от телефонного правила: «2018-10-30» подходит под
# шаблон «группы цифр через разделитель» и без защиты вычищается как номер.
PROTECT_RE = re.compile(
    r'\b\d{4}-\d{2}-\d{2}\b'
    r'|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b'
    r'|\b\d{1,2}/\d{1,2}/\d{2,4}\b'
    r'|\b(?:Q[1-4]\s?)?(?:19|20)\d{2}\b')
PROTECT_TOKEN = '\x00P{}\x00'

# Строки целиком выбрасываются, если начинаются с такого маркера контактного блока
CONTACT_LINE_MARKERS = re.compile(
    r'^\s*(?:тел\.?|телефон|моб\.?|факс|e-?mail|почта|адрес|контакт\w*'
    r'|phone|mobile|tel\.?|fax|address|contacts?)\b', re.I)

# Подозрительные строки: правилами не ловятся, но требуют глаза
SUSPECT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ('возможна сумма прописью', re.compile(
        r'\b(?:стоимость|бюджет|цена|вознаграждение|гонорар|оплата|сумма|аванс'
        r'|cost|price|budget|fee|payment|invoice)\b', re.I)),
    ('упоминание договора', re.compile(
        r'\b(?:договор\w*|контракт\w*|счёт|счет-фактур\w*|agreement|contract)\b', re.I)),
    ('возможные ФИО', re.compile(
        r'\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ]\.\s?[А-ЯЁ]?\.?|\b[А-ЯЁ]\.\s?[А-ЯЁ]?\.\s?[А-ЯЁ][а-яё]{2,}')),
    ('должность рядом с именем', re.compile(
        r'\b(?:партн[её]р|директор|руководител\w*|менеджер|managing\s+partner'
        r'|director|partner|head\s+of)\b', re.I)),
]


def cyrillic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if 'CYRILLIC' in unicodedata.name(c, ''))
    return cyr / len(letters)


def detect_language(text: str) -> str:
    share = cyrillic_share(text)
    if share > 0.6:
        return 'ru'
    if share < 0.1:
        return 'en'
    return 'mixed'


class Sanitizer:
    """
    Вычистка контактов и финансовых данных.

    Считает статистику по типам, чтобы отчёт показывал, чего и сколько удалено:
    нулевое число удалённых сумм в коммерческом предложении — сигнал, что правила
    не сработали, а не что документ чист.
    """

    def __init__(self, names: Optional[Set[str]] = None, strict: bool = False):
        self.names = names or set()
        self.strict = strict
        self.stats: Dict[str, int] = defaultdict(int)
        self.suspects: List[Tuple[str, str]] = []
        self._name_re = self._build_name_regex()

    def _build_name_regex(self) -> Optional[re.Pattern]:
        """
        Регулярка по списку известных фамилий. Список берётся из выгрузки реестра
        (project_employees.csv), где фамилии записаны как «Иванов И.».
        Ищем именно фамилию, чтобы поймать и «Сотрудник Артур», и «А. Сотрудник».
        """
        surnames = set()
        for raw in self.names:
            token = raw.split()[0].strip(' .,;') if raw.split() else ''
            if len(token) > 2:
                surnames.add(re.escape(token))
        if not surnames:
            return None
        return re.compile(r'\b(?:' + '|'.join(sorted(surnames)) + r')\w*\b')

    def scan_suspects(self, line: str, source: str) -> None:
        for label, pattern in SUSPECT_PATTERNS:
            if pattern.search(line):
                self.suspects.append((f'{source} | {label}', line.strip()[:200]))
                break

    def clean_line(self, line: str, source: str) -> Optional[str]:
        if CONTACT_LINE_MARKERS.match(line):
            self.stats['контактная строка целиком'] += 1
            return None

        self.scan_suspects(line, source)

        # прячем даты и годы под непечатаемые метки на время работы правил
        protected: List[str] = []

        def _stash(match: re.Match) -> str:
            protected.append(match.group(0))
            return PROTECT_TOKEN.format(len(protected) - 1)

        line = PROTECT_RE.sub(_stash, line)

        for label, pattern in FINANCE_RULES + CONTACT_RULES:
            line, count = pattern.subn(REDACTED + ' ', line)
            if count:
                self.stats[label] += count

        if self._name_re:
            line, count = self._name_re.subn(REDACTED + ' ', line)
            if count:
                self.stats['фамилия из справочника'] += count

        if self.strict:
            for label, pattern in SUSPECT_PATTERNS:
                if pattern.search(line):
                    self.stats[f'строгий режим: {label}'] += 1
                    return None

        for index, value in enumerate(protected):
            line = line.replace(PROTECT_TOKEN.format(index), value)

        # схлопываем подряд идущие маркеры: «[удалено] [удалено]» читается плохо
        line = re.sub(rf'(?:{re.escape(REDACTED)}[\s,;:–-]*){{2,}}', REDACTED + ' ', line)
        cleaned = re.sub(r'\s{2,}', ' ', line).strip(' \t;,')
        # строка, от которой остались одни маркеры, смысла не несёт
        if not cleaned or cleaned.replace(REDACTED, '').strip(' .,:;–-') == '':
            return None
        return cleaned


# ------------------------------------------------------------------- извлечение

def table_lines(table) -> List[str]:
    """Таблицу разворачиваем построчно: «ячейка | ячейка», без выравнивания."""
    lines = []
    for row in table.rows:
        cells = [c.text.strip().replace('\n', ' ') for c in row.cells]
        cells = [c for c in cells if c]
        if cells:
            lines.append(' | '.join(cells))
    return lines


def extract_pptx(path: Path, keep_notes: bool) -> List[Dict[str, Any]]:
    if Presentation is None:
        raise SystemExit("Нужен python-pptx: pip install python-pptx")
    prs = Presentation(str(path))
    sections: List[Dict[str, Any]] = []

    for index, slide in enumerate(prs.slides, start=1):
        title = ''
        body: List[str] = []

        # заголовок слайда берём из placeholder'а, если он есть; иначе — первая
        # непустая строка текста, иначе слайд останется без осмысленного имени
        try:
            if slide.shapes.title and slide.shapes.title.text.strip():
                title = slide.shapes.title.text.strip().replace('\n', ' ')
        except (AttributeError, ValueError):
            pass

        for shape in slide.shapes:
            if shape.has_table:
                body.extend(table_lines(shape.table))
                continue
            if not shape.has_text_frame:
                continue
            text = shape.text_frame.text.strip()
            if not text or text.replace('\n', ' ') == title:
                continue
            body.extend(t.strip() for t in text.split('\n') if t.strip())

        if keep_notes and slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                body.append('Заметки докладчика: ' + notes.replace('\n', ' '))

        if not title:
            title = body[0][:80] if body else f'Слайд {index}'
            if body and body[0][:80] == title:
                body = body[1:]

        sections.append({'title': title, 'body': body, 'origin': f'слайд {index}'})
    return sections


def extract_docx(path: Path) -> List[Dict[str, Any]]:
    if Document is None:
        raise SystemExit("Нужен python-docx: pip install python-docx")
    doc = Document(str(path))
    sections: List[Dict[str, Any]] = []
    current = {'title': '', 'body': [], 'origin': 'начало документа'}

    def push() -> None:
        if current['title'] or current['body']:
            sections.append(dict(current))

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or '').lower()
        is_heading = style.startswith('heading') or style.startswith('заголовок') or style == 'title'
        if is_heading:
            push()
            current = {'title': text.replace('\n', ' '), 'body': [],
                       'origin': f'раздел «{text[:40]}»'}
        else:
            current['body'].append(text)
    push()

    # таблицы в python-docx лежат отдельно от абзацев и порядок между ними
    # восстановить без разбора XML нельзя, поэтому выносим их в конец
    table_body: List[str] = []
    for table in doc.tables:
        table_body.extend(table_lines(table))
    if table_body:
        sections.append({'title': 'Таблицы документа', 'body': table_body,
                         'origin': 'таблицы'})

    if not sections:
        sections = [{'title': path.stem, 'body': [], 'origin': 'пустой документ'}]
    return sections


# --------------------------------------------------------------------- сборка

def build_markdown(name: str, sections: List[Dict[str, Any]],
                   sanitizer: Sanitizer, source_file: str) -> Tuple[str, int]:
    """
    Собирает markdown с нумерованными разделами.

    Нумерация сквозная: «1. Заголовок». Именно этот формат распознаёт
    chunker.split_sections, поэтому раздел презентации станет отдельным чанком.
    """
    out: List[str] = [name, '']
    kept = 0
    number = 0

    for section in sections:
        title = sanitizer.clean_line(section['title'], f"{source_file} / {section['origin']}")
        body: List[str] = []
        for line in section['body']:
            cleaned = sanitizer.clean_line(line, f"{source_file} / {section['origin']}")
            if cleaned:
                body.append(BODY_NUMBER_RE.sub(r'\1)', cleaned))
        if not title and not body:
            continue
        number += 1
        kept += 1
        out.append(f"{number}. {title or 'Без заголовка'}")
        out.extend(body)
        out.append('')

    return '\n'.join(out).rstrip() + '\n', kept


def load_names(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    file = Path(path)
    if not file.exists():
        print(f"  список фамилий не найден: {file}, пропущено")
        return set()
    with open(file, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    key = 'Сотрудник' if rows and 'Сотрудник' in rows[0] else None
    names = {(r.get(key) or '').strip() for r in rows} if key else set()
    names.discard('')
    print(f"  справочник фамилий: {len(names)} записей")
    return names


def write_report(out: Path, per_file: List[Dict[str, Any]],
                 sanitizer: Sanitizer) -> None:
    lines = [
        '# Отчёт по обезличиванию',
        '',
        f'Дата: {dt.datetime.now().isoformat(timespec="seconds")}',
        '',
        'Автоматическая вычистка не гарантирует полноты. Правила находят телефоны,',
        'почту, ссылки, суммы с валютой, банковские реквизиты и фамилии из справочника,',
        'но не находят суммы прописью, названия клиентов и имена, записанные не по шаблону.',
        'Перед публикацией просмотрите раздел «Требует проверки».',
        '',
        '## Обработанные файлы',
        '',
        '| Файл | Язык | Разделов | Символов |',
        '|---|---|---|---|',
    ]
    for item in per_file:
        lines.append(f"| `{item['source']}` | {item['language']} | "
                     f"{item['sections']} | {item['chars']} |")

    lines.extend(['', '## Что удалено', '', '| Тип | Срабатываний |', '|---|---|'])
    if sanitizer.stats:
        for label, count in sorted(sanitizer.stats.items(), key=lambda kv: -kv[1]):
            lines.append(f'| {label} | {count} |')
    else:
        lines.append('| — | 0 |')
        lines.extend(['', 'Ни одно правило не сработало. Для коммерческих предложений это',
                      'почти наверняка означает, что текст извлёкся не полностью '
                      '(например, лежит в картинках), а не что документы чисты.'])

    lines.extend(['', f'## Требует проверки ({len(sanitizer.suspects)} строк)', ''])
    if sanitizer.suspects:
        lines.append('Строки оставлены в выгрузке, но содержат маркеры, по которым')
        lines.append('правила не отрабатывают автоматически.')
        lines.append('')
        for source, text in sanitizer.suspects[:400]:
            lines.append(f'- **{source}** — {text}')
        if len(sanitizer.suspects) > 400:
            lines.append(f'- … ещё {len(sanitizer.suspects) - 400} строк')
    else:
        lines.append('Подозрительных строк не обнаружено.')

    (out / '_redaction_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Конвертация .pptx/.docx в обезличенный текст для базы знаний")
    parser.add_argument('--src', default='data/raw_office', help='папка с .pptx/.docx')
    parser.add_argument('--out', default='data/source_md_office', help='куда класть .md')
    parser.add_argument('--names', default=None,
                        help='CSV со списком фамилий (например data/raw_csv/project_employees.csv)')
    parser.add_argument('--keep-notes', action='store_true',
                        help='включать заметки докладчика из .pptx')
    parser.add_argument('--strict', action='store_true',
                        help='удалять и подозрительные строки, а не только помечать')
    parser.add_argument('--no-metadata', action='store_true')
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.exists():
        raise SystemExit(f"Не найдена папка {src}")
    files = sorted([f for f in src.iterdir()
                    if f.suffix.lower() in ('.pptx', '.docx') and not f.name.startswith('~$')])
    if not files:
        raise SystemExit(f"В {src} нет .pptx или .docx")

    out.mkdir(parents=True, exist_ok=True)
    sanitizer = Sanitizer(load_names(args.names), strict=args.strict)
    per_file: List[Dict[str, Any]] = []
    entries: List[Dict[str, str]] = []

    for file in files:
        try:
            sections = (extract_pptx(file, args.keep_notes) if file.suffix.lower() == '.pptx'
                        else extract_docx(file))
        except Exception as exc:                       # noqa: BLE001
            print(f"  {file.name}: не прочитан ({exc.__class__.__name__}: {exc})")
            continue

        raw_text = ' '.join(s['title'] + ' ' + ' '.join(s['body']) for s in sections)
        language = detect_language(raw_text)
        title = file.stem.replace('_', ' ').strip()
        markdown, kept = build_markdown(title, sections, sanitizer, file.name)

        target = out / (re.sub(r'[^\w\-.]+', '_', file.stem) + '.md')
        target.write_text(markdown, encoding='utf-8')
        per_file.append({'source': file.name, 'language': language,
                         'sections': kept, 'chars': len(markdown)})
        entries.append({'file_name': target.stem + '.pdf', 'document_name': title,
                        'language': language})
        print(f"  {file.name:44} -> {target.name:36} {kept:3} разделов, {language}")

    if not per_file:
        raise SystemExit("Ни один файл не обработан")

    write_report(out, per_file, sanitizer)

    if not args.no_metadata:
        meta = {'documents': {
            e['file_name']: {'document_name': e['document_name'],
                             'category': 'internal/commercial_proposals',
                             'version': dt.date.today().isoformat(),
                             'source_type': 'converted_presentation',
                             'language': e['language']}
            for e in entries}}
        (out / 'metadata_template.yaml').write_text(
            yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=120),
            encoding='utf-8')

    total_removed = sum(sanitizer.stats.values())
    print(f"\nГотово: {len(per_file)} файлов в {out}/")
    print(f"Удалено фрагментов: {total_removed}. Требует проверки: {len(sanitizer.suspects)} строк.")
    print(f"ПРОСМОТРИТЕ {out}/_redaction_report.md до того, как индексировать.")
    print("Затем: python tools/md_to_pdf.py " + str(out) + " data/raw_office_pdf")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
