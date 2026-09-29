import gc
import shutil
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Iterator

DISTANCE_MAP = {
    'cosine': models.Distance.COSINE,
    'dot': models.Distance.DOT,
    'euclid': models.Distance.EUCLID,
}


class QdrantIndexer:
    def __init__(self, config: dict):
        self.config = config
        # Путь к хранилищу Qdrant берём из config['paths']
        storage_path = config.get('paths', {}).get('qdrant_storage', 'data/qdrant_storage')
        self.client = QdrantClient(path=storage_path)
        self.collection_name = config['vector_store']['collection_name']
        self.vector_size = config['vector_store']['vector_size']
        self.distance = config['vector_store'].get('distance', 'Cosine')
        self.batch_size = config['vector_store'].get('batch_size', 64)
        self.recreate = config['vector_store'].get('recreate_collection', False)

        # Метрика валидируется сразу: молчаливая подмена на COSINE недопустима,
        # иначе конфиг и фактическое поведение расходятся.
        key = str(self.distance).strip().lower()
        if key not in DISTANCE_MAP:
            raise ValueError(
                f"Недопустимая метрика distance='{self.distance}' в config/vector_store.yaml. "
                f"Допустимые значения: Cosine, Dot, Euclid."
            )
        self.distance_enum = DISTANCE_MAP[key]

    def create_collection(self):
        """
        Создаёт коллекцию, пересоздаёт её при recreate_collection: true,
        а при recreate_collection: false строго проверяет параметры существующей коллекции.
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and self.recreate:
            self._recreate_safely()
            print(f"Коллекция '{self.collection_name}' удалена (recreate_collection: true).")
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance_enum
                )
            )
            print(f"Коллекция '{self.collection_name}' создана "
                  f"(размерность {self.vector_size}, метрика {self.distance_enum.name}).")
            leftover_points = self.client.count(self.collection_name).count
            if leftover_points:
                raise RuntimeError(
                    f"Только что созданная коллекция '{self.collection_name}' уже содержит "
                    f"{leftover_points} точек — старое хранилище не удалено. Закройте процессы, "
                    f"использующие Qdrant, удалите папку "
                    f"'{self.config.get('paths', {}).get('qdrant_storage', 'data/qdrant_storage')}' и повторите."
                )
            return

        # Коллекция существует и пересоздание отключено — сверяем параметры и падаем при расхождении.
        info = self.client.get_collection(self.collection_name)
        actual_size = info.config.params.vectors.size
        actual_distance = info.config.params.vectors.distance
        problems = []
        if actual_size != self.vector_size:
            problems.append(f"размерность коллекции {actual_size}, в конфиге {self.vector_size}")
        if actual_distance != self.distance_enum:
            problems.append(f"метрика коллекции {actual_distance.name}, в конфиге {self.distance_enum.name}")
        if problems:
            raise ValueError(
                f"Существующая коллекция '{self.collection_name}' несовместима с конфигом: "
                + "; ".join(problems)
                + ". Установите recreate_collection: true или удалите каталог "
                  f"'{self.config.get('paths', {}).get('qdrant_storage', 'data/qdrant_storage')}'."
            )
        print(f"Коллекция '{self.collection_name}' уже существует, параметры совпадают с конфигом "
              f"(размерность {actual_size}, метрика {actual_distance.name}).")

    # ------------------------------------------------------------------
    # Надёжное пересоздание коллекции в локальном режиме (особенно Windows)
    # ------------------------------------------------------------------

    def _local_collection_dir(self) -> Path:
        storage_path = self.config.get('paths', {}).get('qdrant_storage', 'data/qdrant_storage')
        return Path(storage_path) / 'collection' / self.collection_name

    def _recreate_safely(self) -> None:
        """
        qdrant-client в локальном режиме удаляет папку коллекции через
        shutil.rmtree(..., ignore_errors=True), не закрыв перед этим SQLite.
        На Windows открытый storage.sqlite удалить нельзя, ошибка глотается, и новая
        коллекция подхватывает старые точки (пример: 49 новых чанков -> 112 точек).

        Поэтому: 1) явно закрываем хранилище коллекции; 2) удаляем коллекцию;
        3) если папка всё ещё на месте — закрываем клиент, чистим сами (с ошибкой,
        а не молча) и открываем клиент заново.
        """
        local = getattr(self.client, '_client', None)
        collection = getattr(local, 'collections', {}).get(self.collection_name)
        if collection is not None and hasattr(collection, 'close'):
            try:
                collection.close()
            except Exception:
                pass
        del collection
        gc.collect()

        self.client.delete_collection(self.collection_name)

        leftover = self._local_collection_dir()
        if not leftover.exists():
            return

        print(f"   папка коллекции не удалена клиентом Qdrant ({leftover}) — удаляю вручную")
        self.client.close()
        gc.collect()
        last_error = None
        for _ in range(5):
            try:
                shutil.rmtree(leftover)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.5)
        storage_path = self.config.get('paths', {}).get('qdrant_storage', 'data/qdrant_storage')
        self.client = QdrantClient(path=storage_path)
        if last_error is not None or leftover.exists():
            raise RuntimeError(
                f"Не удалось удалить старую коллекцию {leftover}: {last_error}. "
                f"Закройте все процессы, использующие хранилище (агент, другой запуск), "
                f"удалите папку '{storage_path}' вручную и повторите сборку."
            )

    def _iter_batches(self, records: List[Dict[str, Any]]) -> Iterator[List[models.PointStruct]]:
        """
        Готовит PointStruct порциями по batch_size: полный список точек
        никогда не находится в памяти целиком.
        """
        batch: List[models.PointStruct] = []
        for rec in records:
            chunk_id = rec['metadata'].get('chunk_id')
            if not chunk_id:
                # сюда попасть нельзя: chunk_id проверяется в load_embeddings до индексации
                raise ValueError("Запись без chunk_id дошла до индексации — проверьте vector_loader.load_embeddings()")
            payload = dict(rec['metadata'])
            payload['text'] = rec['text']
            batch.append(models.PointStruct(id=chunk_id, vector=rec['embedding'], payload=payload))
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def index_records(self, records: List[Dict[str, Any]]) -> int:
        """
        Загружает векторы и payload в коллекцию батчами.
        Идентификатор точки = chunk_id из metadata.
        """
        total = len(records)
        if total == 0:
            print("Нет точек для загрузки.")
            return 0

        uploaded = 0
        for batch in self._iter_batches(records):
            self.client.upsert(collection_name=self.collection_name, points=batch)
            uploaded += len(batch)
            print(f"Загружено {uploaded} из {total} векторов.")
        return uploaded

    def get_collection_info(self) -> dict:
        info = self.client.get_collection(self.collection_name)
        return {
            'points_count': info.points_count,
            'status': info.status
        }
