# InternVideo2 本地多模态视频检索系统 (V2)

这是一个基于 `InternVideo2` 强大的特征空间对齐能力构建的**本地纯离线**视频检索引擎。系统支持通过**自然语言（Text-to-Video）**和**参考图片（Image-to-Video）**来高速检索本地视频片段库。

本系统在底层使用 [Milvus Lite](https://milvus.io/) 作为统一的高维向量数据库资源管理器。

---

## 🏗️ 项目架构与文件说明

| 文件/目录 | 功能描述 |
| --- | --- |
| `config.py` | **全局基准配置池**。所有的模型路径、视频数据集路径、Milvus数据库落盘途径及特征维度等在此统一定义，确保前后端不分裂。|
| `model_wrapper.py` | **InternVideo2 模型胶水层**。内部直接接驳官方代码库的初始化能力，提供标准的 `encode_video()`, `encode_image()`, `encode_text()` 方法。|
| `vector_db.py` | **Milvus 检索驱动器**。封装了本地 `.db` 的增删改查逻辑、批处理支持及 `Cosine` 余弦相似度近邻算法。|
| `extract_and_index.py` | **特征提取与向量建库引擎**。后台静默遍历 JSON 设定的视频列表，利用模型提特征并持久化到本地 Milvus。|
| `streamlit_ui.py` | **可视化前端面板**。提供友好的 Web UI 界面、进度状态条以及动态展示检索结果。|
| `InternVideo2/` | **官方核心依赖目录**。需保证包含官方仓库的 `multi_modality` 等核心源码库，系统才能推断正确。|

---

## ⚡ 核心能力与升级修复 (V2 更新点)

1. **真实多模态支持**：完全移除了旧版的占位空壳实现，现在 `model_wrapper.py` 内部原生调用 InternVideo2 接口，完成了图片、文本、视频共享一个同维映射池的最佳实践。
2. **严谨的特征维度对齐修复**：修正了原系统以 768/1024 混合使用导致的崩溃 Bug。当前确认 **InternVideo2 1B Stage2 模型最终对齐向量维度为 512**。
3. **安全内存与数据库同步**：不再直接越权操纵 pymilvus 底层 API。所有存储通过 `MilvusManager` 的批量（Batch）接口平稳写入，防止大批量特征抽取时爆内存。

---

## 🚀 快速开始

### 1. 环境准备与依赖
推荐硬件配置：需要 `>=16GB` 显存的 GPU 支撑 1B 参数视觉大模型流转。
```bash
pip install torch torchvision numpy Pillow decord pymilvus streamlit
# 以及 InternVideo2 要求的其余核心依赖
```

### 2. 相关数据准备配置
打开并编辑项目根目录中的 **`config.py`**：
```python
# 修改为你具体的模型权重路径
MODEL_PATH = '/mnt/video/model/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt'
# 修改为本地 MSR-VTT 或其他视频文件的存储库
VIDEO_DIR = '/mnt/video/data/MSR-VTT/video'
# 修改为定义了视频ID、路径和原始标题的 JSON 清单文件
JSON_DATA_PATH = '/mnt/video/data/MSR-VTT/video/msrvtt_test_1k.json'
```

### 3. 构建本地离线向量库
执行入库脚本。这会自动清空旧库重建，抽取 `JSON_DATA_PATH` 中标明的视频进行推理运算。这可能需要一定时间，运算结果将直接落盘到 `internvideo_v2_local.db`中：
```bash
python extract_and_index.py
```

### 4. 启动 Streamlit UI 及交互搜索
当提示 "向量库构建完成" 后，启动前端可视化面板：
```bash
streamlit run streamlit_ui.py
```

- **文搜视频 (T2V)**：在输入框内使用英文（或者如果模型支持中文的话）描绘场景，模型编码提问即可返回 Top 6 视频切片。
- **图搜视频 (I2V)**：上传包含参考特征的静帧 / 照片，利用图-视频潜空间的相通直接匹配相关视频。
