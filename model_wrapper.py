import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# 将 InternVideo2 的 multi_modality 目录加入 Python 搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
INTERNVIDEO2_PATH = os.path.join(current_dir, "InternVideo2", "multi_modality")
if INTERNVIDEO2_PATH not in sys.path:
    sys.path.insert(0, INTERNVIDEO2_PATH)

try:
    from utils.config import Config, eval_dict_leaf
    from demo.utils import setup_internvideo2
except ImportError as e:
    print(f"Warning: InternVideo2 imports failed, please check submodules: {e}")

class InternVideo2Wrapper:
    def __init__(self, model_path, device="cuda"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_dim = 768  # InternVideo2-1B Stage2 默认维度
        
        print(f"正在准备加载真实的 InternVideo2 模型: {model_path} ...")
        
        # InternVideo2 配置文件路径
        config_path = os.path.join(INTERNVIDEO2_PATH, "demo", "internvideo2_stage2_config.py")
        
        # 加载配置
        config = Config.from_file(config_path)
        config = eval_dict_leaf(config)
        
        # 覆写预训练模型地址到本地真实路径
        config.pretrained_path = model_path
        config.model.vision_encoder.pretrained = model_path
        # 确保与本地环境匹配
        config.model.text_encoder.pretrained = "bert-large-uncased" if "bert" in config.model.text_encoder.name else config.model.text_encoder.pretrained
        
        # 构建模型和 Tokenizer
        self.model, self.tokenizer = setup_internvideo2(config)
        self.model = self.model.to(self.device).eval()
        self.config = config
        
        # InternVideo2 标准图片预处理管道
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _get_video_frames(self, video_path, num_frames=8):
        """解码视频并均匀采样"""
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        # 均匀采样 8 帧 （注意此处需与 config.num_frames 匹配）
        num_frames = getattr(self.config, 'num_frames', num_frames)
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        frames = vr.get_batch(indices).asnumpy()
        
        processed_frames = []
        for f in frames:
            img = Image.fromarray(f)
            processed_frames.append(self.transform(img))
        
        # 转为 [B=1, T, C, H, W]  注：后续传入模型将会重置形状适配
        video_tensor = torch.stack(processed_frames).unsqueeze(0) 
        return video_tensor.to(self.device)

    # ==========================
    # 多模态特征提取接口 (真实 API)
    # ==========================
    
    @torch.no_grad()
    def encode_video(self, video_path):
        """提取视频特征向量并归一化"""
        try:
            video_input = self._get_video_frames(video_path)
            with torch.cuda.amp.autocast():
                # InternVideo2 内部接口获取视频表征
                feat = self.model.get_vid_feat(video_input)
                
            feat = F.normalize(feat, dim=-1)
            return feat.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            print(f"处理视频 {video_path} 失败: {e}")
            return None

    @torch.no_grad()
    def encode_text(self, text):
        """提取文本特征向量供检索使用"""
        try:
            with torch.cuda.amp.autocast():
                # InternVideo2 提供封装好的接口：tokenize + encode
                feat = self.model.get_txt_feat(text)
                
            feat = F.normalize(feat, dim=-1)
            return feat.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            print(f"处理文本特征失败: {e}")
            return None

    @torch.no_grad()
    def encode_image(self, image: Image.Image):
        """提取图片特征向量供检索使用"""
        try:
            # 将图处理为单帧伪装视频形状 [B=1, T=1, C=3, H=224, W=224]
            processed_img = self.transform(image).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.cuda.amp.autocast():
                # 图特征通过 Vision Encoder 一样提取，T=1 自动适配为 True 的 use_image 标识
                feat = self.model.get_vid_feat(processed_img)
            
            feat = F.normalize(feat, dim=-1)
            return feat.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            print(f"处理图片特征失败: {e}")
            return None