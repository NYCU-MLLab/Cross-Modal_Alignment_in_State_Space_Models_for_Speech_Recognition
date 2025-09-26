import torch
from torch.optim import AdamW
from torch.optim.lrs_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler

import config  # 導入設定
from data_utils import get_dataloaders
from tokenizer import SentencePieceTransform
from models import get_model
from engine import Trainer
from utils import plot_loss, plot_wer # 引入繪圖工具

def main():
    # --- 1. 初始化 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 2. 載入分詞器 ---
    # train_sentencepiece(...) 
    tokenizer = SentencePieceTransform(f"{config.MODEL_PREFIX_TED}.model")

    # --- 3. 載入資料 ---
    train_loader, dev_loader = get_dataloaders(tokenizer, config)

    # --- 4. 建立模型 ---
    model = get_model(config).to(device)
    
    # --- 5. 設定優化器與排程器 ---
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scaler = GradScaler()
    
    # --- 6. 建立 Trainer ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=dev_loader,
        tokenizer=tokenizer,
        device=device,
        config=config
    )
    
    # --- 7. 開始訓練 ---
    trainer.fit(
        epochs=config.NUM_EPOCHS,
        model_path=config.MODEL_SAVE_PATH,
        log_path=config.LOG_PATH,
        pred_path=config.PREDICTIONS_PATH,
        scaler=scaler
    )
    
    # --- 8. (可選) 訓練後評估與繪圖 ---
    # trainer.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    # test_wer = trainer.test(test_loader)
    # print(f"Final Test WER: {test_wer}")
    
    # 繪製 loss 和 WER 曲線
    plot_loss(trainer.train_loss_history, trainer.valid_loss_history)
    plot_wer(trainer.valid_ctc_wer_history, trainer.valid_aed_wer_history)


if __name__ == "__main__":
    main()