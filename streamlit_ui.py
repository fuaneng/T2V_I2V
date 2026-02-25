import streamlit as st
import json
import os
from PIL import Image
from model_wrapper import InternVideo2Wrapper
from vector_db import MilvusManager
import config

# --- 页面 UI 基础配置 ---
st.set_page_config(page_title="InternVideo2 Pro V2", layout="wide")

@st.cache_resource
def load_essentials():
    model = InternVideo2Wrapper(config.MODEL_PATH)
    db = MilvusManager()
    return model, db

# 后台静默加载模型环境
with st.spinner("正在初始化 InternVideo2 模型与向量引擎，请稍候..."):
    model_engine, db_manager = load_essentials()

# --- 侧边栏：状态面板与调试测试工具 ---
with st.sidebar:
    st.title("⚙️ 系统状态管理")
    st.info(f"📊 当前数据库量级: **{db_manager.get_count()}** 帧/片段")
    st.divider()
    
    st.subheader("🛠️ 演示用快速入库工具")
    st.caption("注: 前端触发强制重新初始化并灌入前50条数据。")
    if st.button("一键初始化并写入数据"):
        if not os.path.exists(config.JSON_DATA_PATH):
            st.error("JSON 预置文件路径不存在，请检查挂载配置！")
        else:
            db_manager.drop_collection()
            db_manager._create_collection(config.DIMENSION)
            
            with open(config.JSON_DATA_PATH, 'r') as f:
                videos_meta = json.load(f)
            
            pbar = st.progress(0)
            status_text = st.empty()
            
            for i, item in enumerate(videos_meta[:50]):
                v_path = os.path.join(config.VIDEO_DIR, item['video'])
                if os.path.exists(v_path):
                    feat = model_engine.encode_video(v_path)
                    if feat is not None:
                        db_manager.insert_video_data(feat, item['video_id'], v_path, item.get('caption', 'None'))
                
                pbar.progress((i + 1) / 50)
                status_text.text(f"入库进度: {item['video']}")
                
            st.success("入库成功！右上方可以刷新页面。")
            st.rerun()

# --- 主核心交互区域 ---
st.title("🎥 InternVideo2: 跨模态文本/图像搜视频 V2")
st.markdown("该演示运用 InternVideo2 的同一潜空间 (Latent Space) 对齐能力，支持自然语言检索与给定参考图检索高度相关的长短视频切片。")

tab1, tab2 = st.tabs(["🔍 文本寻找视频 (T2V)", "🖼️ 图像寻找视频 (I2V)"])

def display_results(hits):
    """渲染召回卡片区域集"""
    if not hits:
        st.warning("暂无召回结果，可能库为空，请先在侧边栏做数据入库操作。")
        return

    cols = st.columns(3)
    for i, hit in enumerate(hits):
        entity = hit['entity']
        score = hit['distance'] 
        
        with cols[i % 3]:
            try:
                # 尝试渲染本地路径
                st.video(entity['video_path'])
            except Exception:
                st.error("🎥 视频文件不存在或无权访问")
                
            st.metric(label="相似性分数 (Cosine)", value=f"{score:.4f}")
            st.caption(f"**资产 ID:** `{entity['video_id']}`")
            st.markdown(f"**原标:** *{entity['caption']}*")
            st.divider()

with tab1:
    col_t1, col_t2 = st.columns([4, 1])
    with col_t1:
        query_text = st.text_input("请用自然语言描述视频：", placeholder="示例: A dog is catching a frisbee on the beach")
    with col_t2:
        st.write("")
        st.write("")
        trigger_txt = st.button("启动联合语义检索", use_container_width=True)
        
    if trigger_txt and query_text:
        with st.spinner("InternTextEncoder 深度编码中..."):
            text_feat = model_engine.encode_text(query_text)
            if text_feat is not None:
                results = db_manager.search(text_feat, top_k=6)
                display_results(results)
            else:
                st.error("文本编码失败，Model 层未正常返回特征向量。")

with tab2:
    uploaded_file = st.file_uploader("请选定一张参照图：", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="用于锚定特征序列的参考图", width=360)
        
        if st.button("启动关联视觉比对", use_container_width=True):
            with st.spinner("InternVisionEncoder 抽取高维多维视觉表征中..."):
                img_feat = model_engine.encode_image(img)
                if img_feat is not None:
                    results = db_manager.search(img_feat, top_k=6)
                    display_results(results)
                else:
                    st.error("图片编码失败，Model 层未能转化图片数据。")