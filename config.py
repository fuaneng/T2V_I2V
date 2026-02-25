# config.py
import os

MODEL_PATH = '/mnt/video/model/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt'
VIDEO_DIR = '/mnt/video/data/MSR-VTT/video'
JSON_DATA_PATH = '/mnt/video/data/MSR-VTT/video/msrvtt_test_1k.json'

DB_FILE_PATH = "./internvideo_v2_local.db"
COLLECTION_NAME = "msrvtt_collection"
DIMENSION = 512  # 修复: InternVideo2 Stage2 的 embed_dim 最终投影维度是 512 而不是 768
