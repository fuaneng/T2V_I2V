from pymilvus import MilvusClient
import config

class MilvusManager:
    def __init__(self, db_path=config.DB_FILE_PATH, collection_name=config.COLLECTION_NAME):
        # 统一读取 config，彻底杜绝前后端读错 db 的致命 bug
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self._create_collection(dim=config.DIMENSION)

    def _create_collection(self, dim):
        if self.client.has_collection(self.collection_name):
            return
        
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dim,
            primary_field_name="id",
            id_type="int",
            auto_id=True,
            vector_field_name="vector",
            metric_type="COSINE" 
        )

    def insert_video_data(self, vector, video_id, video_path, caption):
        """插入单条数据"""
        data = [{
            "vector": vector.tolist() if hasattr(vector, 'tolist') else vector,
            "video_id": video_id,
            "video_path": video_path,
            "caption": caption
        }]
        self.client.insert(collection_name=self.collection_name, data=data)

    def insert_batch(self, batch_data):
        """执行批处理入库"""
        if batch_data:
            self.client.insert(collection_name=self.collection_name, data=batch_data)

    def search(self, query_vector, top_k=6):
        """执行向量检索"""
        query_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_list],
            limit=top_k,
            output_fields=["video_id", "video_path", "caption"],
            search_params={"metric_type": "COSINE"}
        )
        return results[0] if results and len(results) > 0 else []

    def get_count(self):
        """获取当前入库向量记录总数"""
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return int(stats.get('row_count', 0))
        except Exception:
            return 0
            
    def drop_collection(self):
        """危险操作：清空当前集合"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)