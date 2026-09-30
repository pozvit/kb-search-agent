import yaml
import uuid
from chunk_loader import load_prepared_documents
from chunker import chunk_document
from chunk_exporter import export_chunks
from card_metadata import card_meta_for_chunk, load_card_metadata

# Определяем пространство имён для стабильных UUID (можно использовать любой константный UUID)
NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # UUID для DNS

def run_chunk_pipeline(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    input_file = config['paths']['input_jsonl']
    output_file = config['paths']['output_jsonl']
    chunk_config = config['chunking']
    strategy = chunk_config['strategy']
    chunk_size = chunk_config['chunk_size']
    chunk_overlap = chunk_config['chunk_overlap']
    split_long = chunk_config.get('split_long_sentences', True)
    merge_short = chunk_config.get('merge_short_sections', True)
    card_meta_paths = config['paths'].get('card_metadata') or []
    if isinstance(card_meta_paths, str):
        card_meta_paths = [card_meta_paths]
    card_index = load_card_metadata(card_meta_paths)
    if card_meta_paths:
        print(f"   метаданные карточек реестров: {len(card_index)}")
    enriched = 0
    card_files = {name for name, _ in card_index}
    card_chunks_without_meta = 0
    
    print(f"Загрузка подготовленных документов из {input_file}")
    documents = load_prepared_documents(input_file)
    print(f"   Загружено {len(documents)} документов.")
    
    all_chunks = []
    total_chunks = 0
    
    for doc in documents:
        doc_id = doc['metadata'].get('document_id', 'unknown')
        text = doc['text']
        # Пробрасываем ВСЕ метаданные документа, а не фиксированный список полей:
        # прикладные поля (document_name, category, version) должны дойти
        # до payload Qdrant и попасть в ответ RAG. Чанк-специфичные ключи
        # ниже перезаписывают одноимённые поля документа.
        original_meta = dict(doc['metadata'])
        chunks_data = chunk_document(text, strategy, chunk_size, chunk_overlap, split_long,
                                      merge_short)
        
        for chunk_info in chunks_data:
            chunk_text = chunk_info['text']
            chunk_idx = chunk_info['chunk_index']
            actual_size = chunk_info['chunk_size']
            
            # Генерируем стабильный UUID на основе document_id и индекса
            chunk_uuid = uuid.uuid5(NAMESPACE, f"{doc_id}_{chunk_idx}")
            chunk_id = str(chunk_uuid)  # строка в формате UUID
            
            chunk_meta = dict(original_meta)
            card_meta = card_meta_for_chunk(card_index, original_meta.get('file_name'), chunk_text)
            if card_meta:
                chunk_meta.update(card_meta)
                enriched += 1
            elif original_meta.get('file_name') in card_files:
                card_chunks_without_meta += 1
            chunk_meta.update({
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "chunk_size": actual_size,
                "chunk_overlap": chunk_overlap,
                "strategy": strategy,
            })

            chunk_obj = {
                "text": chunk_text,
                "metadata": chunk_meta
            }
            all_chunks.append(chunk_obj)
            total_chunks += 1
    
    print(f"Создано {total_chunks} чанков.")
    if card_index:
        print(f"   с метаданными карточек: {enriched}; чанков карточек без метаданных: {card_chunks_without_meta}")
    print(f"Сохранение результатов в {output_file}")
    export_chunks(all_chunks, output_file)
    print("Готово!")

if __name__ == "__main__":
    import sys
    run_chunk_pipeline(sys.argv[1] if len(sys.argv) > 1 else "config/chunking.yaml")