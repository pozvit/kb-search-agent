import yaml
from loader import load_documents
from cleaner import clean_text
from normalizer import normalize_text
from structurer import structure_document
from exporter import export_documents

def run_pipeline(config_path: str):
    # Загрузка конфигурации
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Смешанная база: paths.raw_dirs — список папок, paths.raw_dir — одна папка.
    # Список имеет приоритет; одиночный ключ оставлен для совместимости
    # с прежними конфигами и с config/rag_baseline.yaml.
    raw_dirs = config['paths'].get('raw_dirs') or [config['paths']['raw_dir']]
    if isinstance(raw_dirs, str):
        raw_dirs = [raw_dirs]
    prepared_dir = config['paths']['prepared_dir']

    print("Загрузка документов...")
    raw_docs = []
    for directory in raw_dirs:
        docs = load_documents(directory)
        print(f"   {directory}: {len(docs)}")
        raw_docs.extend(docs)
    print(f"   Загружено {len(raw_docs)} документов из {len(raw_dirs)} папок.")
    
    print("Очистка текста...")
    cleaned_docs = []
    for doc in raw_docs:
        doc['text'] = clean_text(doc['text'], config.get('cleaning', {}))
        if doc['text'].strip():  # пропускаем пустые
            cleaned_docs.append(doc)
    print(f"   Осталось {len(cleaned_docs)} документов после очистки.")
    
    print("Нормализация текста...")
    for doc in cleaned_docs:
        doc['text'] = normalize_text(doc['text'], config.get('normalization', {}))
    
    print("Структурирование документов...")
    structured_docs = [structure_document(doc, config) for doc in cleaned_docs]
    
    print("Экспорт результатов...")
    output_file = export_documents(structured_docs, prepared_dir, config.get('export', {}))
    print(f"Готово! Результат сохранён в {output_file}")
    
    # Краткая статистика
    print("Итоговая статистика:")
    print(f"   - Всего документов: {len(structured_docs)}")
    print(f"   - Уникальных источников: {len(set(doc['metadata']['source'] for doc in structured_docs))}")
    print(f"   - Форматы: {set(doc['metadata']['file_type'] for doc in structured_docs)}")

if __name__ == "__main__":
    # Для прямого запуска
    import sys
    run_pipeline(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")