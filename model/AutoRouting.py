from model.__init__ import *
import time as t
from model import logger
LOGGER = logger.Logger().get_logger(logger_type="embedding_")

class Qdrant:
    def __init__(self, model_name="hiieu/halong_embedding"):
        self.model_name = model_name
        self.client = QdrantClient(
            url="https://5de6f86f-9666-4d5e-9b64-0718dbad2e4d.us-east4-0.gcp.cloud.qdrant.io",
            api_key=QDRANT_API_KEY,
            timeout=1000
        )
        LOGGER.info("Initializing AutoRouting...")
    
    def turn_on_optimization_mode(self,collection_name):
        self.client.update_collection(
            collection_name=collection_name,
            optimizer_config=OptimizersConfigDiff(),
        )

    def check_status(self, collection_name):
        response = self.client.get_collection(collection_name=collection_name)
        status = response.status

        if status == 'green':
            LOGGER.info("Status Green")
        elif status == 'yellow':
            LOGGER.info("Status Yellow, Qdrant is optimizing. Waiting for 30 seconds before retrying ...")
            t.sleep(30)
            return self.check_status(collection_name) 
        elif status == 'grey':
            self.turn_on_optimization_mode(collection_name=collection_name)
            LOGGER.info("Status Grey, turn_on_optimization_mode done, Qdrant is optimizing ... Sleeping 60s")
            t.sleep(60)
            return self.check_status(collection_name) 
        else:
            LOGGER.info("Server down, please restart")
        return status

    def create_or_recreate_collection(self, collection_name, vector_size=768, distance_metric=Distance.COSINE):
        vectors_config = VectorParams(
            size=vector_size,  # Kích thước vector
            distance=distance_metric,  # Khoảng cách sử dụng trong tìm kiếm vector
        )

        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            optimizers_config=OptimizersConfigDiff(default_segment_number=16),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=False,
                ),
            ),
        )
        LOGGER.info(f"Collection {collection_name} has been recreated.")

    def create_payload_index(self, collection_name, field_name, field_schema=PayloadSchemaType.TEXT):

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
        LOGGER.info(f"Payload index for field {field_name} has been created in collection {collection_name}.")

    def chunk_list(self,input_list, chunk_size):
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]

    def retry_upsert(self, collection_name, points, retries=5):
        self.check_status(collection_name)
        attempt = 0
        while attempt < retries:
            try:
                self.client.upsert(collection_name=collection_name, points=points)
                LOGGER.info(f'Successfully inserted batch')
                return True
            except Exception as e:
                attempt += 1
                LOGGER.info(f'Error inserting: {e}, waiting for 30 seconds and retrying ({attempt}/{retries})')
                t.sleep(30)
        
    @retry(stop=stop_after_attempt(3),wait=wait_fixed(30))
    def delete_points(self, key, collection_name, delete_list, batch_size=100):
        for batch in delete_list:
            LOGGER.info(f'Delete driver {batch}')
            try:
                must_conditions = [
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=batch)
                    )
                ]
                filter_selector = FilterSelector(
                    filter=Filter(must=must_conditions)
                )
                result = self.client.delete(
                    collection_name=collection_name,
                    points_selector=filter_selector,
                )
                deleted_points = len(batch)
                if hasattr(result, 'status') and result.status == 'completed':
                    LOGGER.info(f"Đã xóa {deleted_points} points.")
                    deleted_driver = pd.DataFrame([{'driver_name': batch, 'deleted_at': pd.Timestamp.now()}])
                    self.toBigquery(deleted_driver, table_id='deleted_drivers')
                else:
                    LOGGER.info(f"Không thể xóa các points.")
            except httpx.ReadTimeout:
                LOGGER.info(f"Timeout when deleting batch of {len(batch)} points. Retrying...")
                t.sleep(2)  # Sleep for 2 seconds before retrying
            except Exception as e:
                LOGGER.info(f"An error occurred while deleting points: {str(e)}")

class TextEmbedder:
    def __init__(self, model_name="hiieu/halong_embedding"):
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = SentenceTransformer(model_name, device=device)
        LOGGER.info("Using device: %s", device)

    def encode(self, address: str) -> np.ndarray:
        try:
            # Adjusted to pass the entire address, not individual characters
            vector = self.model.encode(address, show_progress_bar=False)
        except Exception as e:
            LOGGER.info(f"Error processing row with address {address}, error: {e}")
        return vector.reshape(1, -1)

class AutoRouting(TextEmbedder):
    
    def __init__(self, model_name="hiieu/halong_embedding"):
        super().__init__(model_name=model_name)
        self.client = QdrantClient(
            url="https://5de6f86f-9666-4d5e-9b64-0718dbad2e4d.us-east4-0.gcp.cloud.qdrant.io",
            api_key=QDRANT_API_KEY,
            timeout=1000
        )
        self.Qdrant = Qdrant()
        # Kiểm tra trạng thái kết nối với Qdrant collection
        status = self.client.get_collection(collection_name=COLLECTION_NAME).status
        LOGGER.info("Initializing AutoRouting...")
        LOGGER.info(f"Loading model: {model_name}")
        LOGGER.info(f"Connecting qdrant collection status: {status}")

    def info(self):
        self.Database.info()

    @staticmethod
    @contextmanager
    def time_process(task_name="Task"):
        start_time = datetime.now()
        yield
        end_time = datetime.now()
        LOGGER.info(f"{task_name} completed in {(end_time - start_time).total_seconds():.4f} seconds")

    def process_row(self, row, col_name="address_concat"):
        address = row[col_name]
        try:
            embedding = self.encode(address)
        except Exception as e:
            logger.error(f"Error processing row with address {address}, error: {e}")
            embedding = np.zeros(self.model.get_sentence_embedding_dimension())

        return embedding
    
    def process_embedding(self, row, col_name="address_concat"):
        dict_result = {}
        for column in row.index:
            dict_result[column] = row[column]

        if col_name not in row:
            logger.error(f"Column {col_name} does not exist in the row!")
            return None  

        address = row[col_name]
        try:
            embedding = self.encode(address)
            dict_result["embedding"] = embedding
        except Exception as e:
            logger.error(f"Error processing row with address {address}, error: {e}")
            dict_result["embedding"] = np.zeros(self.model.get_sentence_embedding_dimension())
        return dict_result

    async def batch_process(self, df, col_name, num_workers=None):
        print("Batch processing")
        with self.time_process("Batch processing"):
            num_workers = (
                num_workers
                if num_workers is not None
                else max(multiprocessing.cpu_count() // 2, 1)
            )
            results = await self._process_chunk_serial(df, col_name, type = 'embedding')

        return results
    
    @staticmethod
    def clear_cuda_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            LOGGER.info("CUDA cache cleared.")

    async def embedding(self,ref_database,chunk_size=20000):
        counter = 0

        def generate_uuid_from_string(input_string): # Hàm để tạo UUID từ address_concat (hash-based UUID)
            hash_object = hashlib.sha256(input_string.encode('utf-8')) # Tạo hash từ input_string (address_concat)
            return uuid.UUID(hash_object.hexdigest()[:32]) # Lấy 16 byte đầu tiên của hash và chuyển thành UUID

        results = await self.batch_process(ref_database, col_name="address_concat")
        results = pd.DataFrame(results)
        LOGGER.info(f'Process embedding done: {results.shape}')

        results["photo_latitude"] = results["photo_latitude"].astype(np.float32)
        results["photo_longitude"] = results["photo_longitude"].astype(np.float32)
        data_chunks = range(0, len(results), chunk_size)

        # Chạy qua từng chunk dữ liệu và thực hiện upsert
        for start in data_chunks:
            end = min(start + chunk_size, len(results))
            points = [
                PointStruct(
                    id=str(generate_uuid_from_string(address_concat)), 
                    vector=vector.tolist(),
                    payload={
                        "short_name": short_name,
                        "address_concat": address_concat,
                        "display_name_top": display_name_top,
                        "photo_latitude": photo_latitude,
                        "photo_longitude": photo_longitude,
                        "hub_id": hub_id,
                        "hub_name": hub_name
                    },
                )
                for index, (
                    vector,
                    short_name,
                    address_concat,
                    display_name_top,
                    photo_latitude,
                    photo_longitude,
                    hub_id,
                    hub_name
                ) in enumerate(
                    zip(
                        np.vstack(results["embedding"][start:end].to_list()),
                        results["short_name"][start:end].to_list(),
                        results["address_concat"][start:end].to_list(),
                        results["display_name_top"][start:end].to_list(),
                        results["photo_latitude"][start:end].to_list(),
                        results["photo_longitude"][start:end].to_list(),
                        results["hub_id"][start:end].to_list(),
                        results["hub_name"][start:end].to_list(),
                    ),
                    start,
                )
            ]
            success = self.Qdrant.retry_upsert(collection_name=COLLECTION_NAME, points=points)
            
            if success:
                LOGGER.info(f'Successfully inserted batch!')
            counter += len(points)

    async def _process_chunk_serial(self, df, col_name="address_concat", type ='search'):
            # with self.time_process("Processing batch serially"):
        results = []
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            position=0,
            leave=True,
        ):
            if type == 'search':
                results.append(self.process_row(row, col_name))
            else:
                results.append(self.process_embedding(row, col_name))
        return results

    async def _process_chunk_parallel(self, df):
        results = []
        for df_chunk in np.array_split(df,20):  # Chia DataFrame thành 100 phần
            try:
                res = await self._process_chunk_serial(df = df_chunk)
                results.extend(res if isinstance(res, list) else [res])

            except Exception as e:
                LOGGER.error(f"Error processing chunk: {str(e)}")
        return results

    async def search(self, search_queries, df, batch_size):
        start_time = datetime.now()
        results = []
        
        # Chia df thành các batch trước khi xử lý
        batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

        for i, batch_df in enumerate(batches):
            batch_queries = search_queries[i * batch_size:(i + 1) * batch_size]
            retry_count = 0
            success = False
            
            while retry_count < 5 and not success:
                try:
                    LOGGER.info(f"Processing batch {i + 1} with size {len(batch_df)} (attempt {retry_count + 1})")
                    
                    responses = await asyncio.to_thread(
                        self.client.query_batch_points,
                        collection_name=COLLECTION_NAME,
                        requests=batch_queries,
                        timeout=1000
                    )
                    success = True
                    
                    # Ánh xạ đúng order_id cho từng batch
                    for idx, response in enumerate(responses):
                        order_id = batch_df.iloc[idx]['order_id']  # Ánh xạ chính xác từ batch_df
                        
                        temp_results = [
                            {
                                "related_address": r.payload.get("address_concat"),
                                "similarity_score": float(r.score),
                                "latitude": r.payload.get("photo_latitude"),
                                "longitude": r.payload.get("photo_longitude"),
                                "display_name_top": r.payload.get("display_name_top"),
                            }
                            for r in response.points
                        ]
                        
                        driver_name_dict = {}
                        for result in temp_results:
                            driver_name = result["display_name_top"]
                            if driver_name not in driver_name_dict or driver_name_dict[driver_name]["similarity_score"] < result["similarity_score"]:
                                driver_name_dict[driver_name] = result
                        
                        sorted_results = sorted(driver_name_dict.values(), key=lambda x: -x["similarity_score"])
                        for seq_no, result in enumerate(sorted_results[:5], start=1):
                            result["order_id"] = order_id
                            result["seq_no"] = seq_no
                            results.append(result)
                
                except Exception as e:
                    retry_count += 1
                    LOGGER.warning(f"Batch {i + 1} failed on attempt {retry_count}. Error: {e}")
                    
                    if retry_count == 5 and batch_size > 60:
                        batch_size = max(batch_size - 1000, 100)
                        LOGGER.info(f"Reducing batch size to {batch_size} and retrying.")
                    elif retry_count == 5:
                        LOGGER.error("Maximum retries reached. Skipping batch.")
                    await asyncio.sleep(1)

            LOGGER.info(f"Batch {i + 1} completed.")

        result_df = pd.DataFrame(results)
        LOGGER.info(f"Total completed in {(datetime.now() - start_time).total_seconds() / 60:.2f} minutes.")
        return result_df

    async def run(self, df, batch_size):
        batch_embeddings = await self._process_chunk_parallel(df)
        search_queries = [
            QueryRequest(
                query=embedding[0], 
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="hub_name",
                            match=MatchText(text=hub_name),
                        )
                    ]
                ),
                limit=30,
                with_payload=True,
            )
            for embedding, hub_name in zip(batch_embeddings, df["hub_name"])
        ]
        result_df = await self.search(search_queries, df, batch_size=batch_size)
        return result_df
