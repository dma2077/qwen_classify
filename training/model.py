import yaml
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.qwen2_5_vl_classify import Qwen2_5_VLForImageClassification

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def build_model(cfg):
    return Qwen2_5_VLForImageClassification(
        pretrained_model_name=cfg["model"]["pretrained_name"],
        num_labels=cfg["model"]["num_labels"]
    )
