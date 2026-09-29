"""
Сборка индекса одной командой: подготовка -> чанкинг -> embeddings -> Qdrant.

    python src/build_index.py --profile public              # 5 обезличенных PDF из data/raw (тесты и сдача)
    python src/build_index.py --profile full                # полная база по config/*.yaml (только локально)
    python src/build_index.py --profile public --no-embed   # только чанки: для --retriever lexical

Профили отличаются только набором конфигов; код этапов общий.
После изменения параметров чанкинга запускайте сборку БЕЗ --no-embed: иначе чанки
обновятся, а эмбеддинги и Qdrant останутся от предыдущей сборки.

Для смоук-теста без скачивания модели: RAG_FAKE_EMBEDDINGS=1 (качество поиска нерелевантно).
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROFILES = {
    "public": {
        "prepare": "config/public/config.yaml",
        "chunk": "config/public/chunking.yaml",
        "embed": "config/public/embeddings.yaml",
        "index": "config/public/vector_store.yaml",
    },
    "full": {
        "prepare": "config/config.yaml",
        "chunk": "config/chunking.yaml",
        "embed": "config/embeddings.yaml",
        "index": "config/vector_store.yaml",
    },
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Сборка индекса базы знаний")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="public")
    parser.add_argument("--skip-prepare", action="store_true", help="не пересобирать prepared/chunks")
    parser.add_argument("--no-embed", action="store_true",
                        help="только подготовка и чанкинг (достаточно для --retriever lexical, без torch и модели)")
    args = parser.parse_args(argv)

    os.chdir(PROJECT_ROOT)
    cfg = PROFILES[args.profile]

    from pipeline import run_pipeline
    from chunk_pipeline import run_chunk_pipeline

    stages = []
    if not args.skip_prepare:
        stages += [("1/4 подготовка", run_pipeline, cfg["prepare"]),
                   ("2/4 чанкинг", run_chunk_pipeline, cfg["chunk"])]
    if not args.no_embed:
        from embed_pipeline import run_embed_pipeline      # импорт тянет torch/sentence-transformers
        from vector_pipeline import run_vector_pipeline
        stages += [("3/4 embeddings", run_embed_pipeline, cfg["embed"]),
                   ("4/4 индексация Qdrant", run_vector_pipeline, cfg["index"])]

    started = time.time()
    result = None
    for title, func, config_path in stages:
        print(f"\n===== {title} ({config_path}) =====")
        t0 = time.time()
        result = func(config_path)
        print(f"----- {title}: {time.time() - t0:.1f} c")

    elapsed = time.time() - started
    if args.no_embed:
        print(f"\nПрофиль '{args.profile}': чанки обновлены за {elapsed:.1f} c. "
              f"Эмбеддинги и Qdrant НЕ обновлялись — годится только для --retriever lexical.\n"
              f"Для семантического поиска выполните: python src/build_index.py --profile {args.profile}")
        return 0
    ok = bool(result)
    print(f"\nПрофиль '{args.profile}': индекс {'валиден' if ok else 'НЕ валиден'}, всего {elapsed:.1f} c")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
