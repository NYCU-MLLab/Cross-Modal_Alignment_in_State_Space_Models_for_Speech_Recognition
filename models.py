import torch
import torch.nn as nn
from models.mambaCTC import MambaCTCModel, BiMambaCTCModel 

def get_model(config):
    # 根據 config 檔案的設定來回傳指定的模型
    model = BiMambaCTCModel(
        input_dim=config.INPUT_DIM,
        vocab_size=config.VOCAB_SIZE,
        d_model=config.D_MODEL,
        n_head=config.N_HEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
    )
    return model