import os
import json
from tqdm import tqdm
from model_wrapper import InternVideo2Wrapper
from vector_db import MilvusManager
import config

def build_index():
    print("====================================")
    print("🚀 开始构建本地视频向量库 (V2)...")
    print(f"配置维度: {config.DIMENSION}, 目标数据库: {config.DB_FILE_PATH}")
    print("====================================")
    
    # 1. 初始化模型和 V2 重构后的数据库接口
    engine = InternVideo2Wrapper(config.MODEL_PATH)
    db = MilvusManager()

    # 重置并清空已有的集合数据，确保索引干净重新构建
    db.drop_collection()
    db._create_collection(dim=config.DIMENSION)

    # 2. 读取 MSR-VTT 元数据文件
    if not os.path.exists(config.JSON_DATA_PATH):
        print(f"❌ 找不到 JSON 数据描述文件: {config.JSON_DATA_PATH}")
        return

    with open(config.JSON_DATA_PATH, 'r') as f:
        video_list = json.load(f)

    print(f"即将处理并提取 {len(video_list)} 个视频...")

    # 3. 循环抽帧和批次插入
    batch_data = []
    for item in tqdm(video_list):
        video_filename = item['video']
        video_full_path = os.path.join(config.VIDEO_DIR, video_filename)

        if not os.path.exists(video_full_path):
            continue

        feat = engine.encode_video(video_full_path)
        
        if feat is not None:
            # 兼容 MSR-VTT 字典结构
            record = {
                "vector": feat.tolist(),
                "video_id": item['video_id'],
                "video_path": video_full_path,
                "caption": item.get('caption', 'None')
            }
            batch_data.append(record)

        # 批处理批量写入，防止长期挂载单次写入阻塞
        if len(batch_data) >= 20:
            db.insert_batch(batch_data)
            batch_data = []

    # 剩余没满20的尾盘数据写入
    if batch_data:
        db.insert_batch(batch_data)

    print(f"\n✅ 向量库构建完成！")
    print(f"当前全量数据库已就绪，总计向量数: {db.get_count()}")

if __name__ == "__main__":
    import torch
    build_index()