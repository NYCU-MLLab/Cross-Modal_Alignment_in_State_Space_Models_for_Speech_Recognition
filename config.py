# --- 資料路徑 ---
DATASET_DIR = "/home/fanche/thesis/work/dataset"
MODEL_PREFIX_LIBRISPEECH = "/home/fanche/thesis/work/tokenizer/spm_librispeech-960-5000"
MODEL_PREFIX_TED = "/home/fanche/thesis/work/tokenizer/spm_ted-5000"

# --- 模型參數 ---
D_MODEL = 512
N_HEAD = 4
NUM_ENCODER_LAYERS = 18
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
VOCAB_SIZE = 5000
INPUT_DIM = 80  # Mel-spectrogram dimension

# --- 訓練參數 ---
BATCH_SIZE = 32
NUM_EPOCHS = 120
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8
CTC_LOSS_WEIGHT = 0.3
CE_LOSS_WEIGHT = 0.7  # Or 0 if not used
DURATION_LOSS_WEIGHT = 0.0
GRADIENT_ACCUMULATION_TARGET = 128
MAX_GRAD_NORM = 1.0

# --- 檔案儲存路徑 ---
MODEL_SAVE_PATH = "attn_ctc_ted.pth"
LOG_PATH = "attn_ctc_ted_log.txt"
PREDICTIONS_PATH = "attn_ctc_ted_preds.txt"