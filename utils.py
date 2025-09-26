import matplotlib.pyplot as plt

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def plot_train_loss(self):
    """繪製CTC loss與CE loss 曲線"""
    plt.figure(figsize=(8, 5))
    plt.plot(self.train_ctc_loss_history, label='Train CTC Loss')
    plt.plot(self.train_ce_loss_history, label='Train CE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training CTC and CE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss(self):
    """繪製訓練與驗證 loss 曲線"""
    plt.figure(figsize=(8, 5))
    plt.plot(self.train_loss_history, label='Train Loss')
    plt.plot(self.valid_loss_history, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_wer(self):
    """繪製驗證 WER 曲線"""
    # 忽略前 30 個 epoch 的 WER
    plt.figure(figsize=(8, 5))
    epochs = range(120, len(self.valid_ctc_wer_history))  # 設定 X 軸對應的 epoch 數值
    plt.plot(epochs, self.valid_ctc_wer_history[120:], label='CTC WER')
    plt.plot(epochs, self.valid_aed_wer_history[120:], label='AED WER')
    plt.xlabel('Epoch')
    plt.ylabel('WER (%)')
    plt.title('Validation WER')
    plt.legend()
    plt.grid(True)
    plt.show()

def read_log(self, log_path):
    """讀取訓練過程的 loss 與 WER 紀錄"""
    self.train_ctc_loss_history = []
    self.train_ce_loss_history = []
    self.train_loss_history = []
    self.valid_loss_history = []
    self.valid_ctc_loss_history = []
    self.valid_ce_loss_history = []
    self.valid_ctc_wer_history = []
    self.valid_aed_wer_history = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(', ')
        if len(parts) < 6:
            continue  # 避免讀取到格式不正確的行

        # 解析數值
        try:
            # Epoch: 1, Train: 0.1234, Valid: 0.5678, Train CTC: 0.3456, Train CE: 0.4567, Valid CTC: 0.1233, Valid CE: 0.2589, CTC WER: 12.34, AED WER: 23.45
            epoch = int(parts[0].split(': ')[1])
            train_loss = float(parts[1].split(': ')[1])
            valid_loss = float(parts[2].split(': ')[1])
            train_ctc_loss = float(parts[3].split(': ')[1])
            train_ce_loss = float(parts[4].split(': ')[1])
            valid_ctc_loss = float(parts[5].split(': ')[1])
            valid_ce_loss = float(parts[6].split(': ')[1])
            ctc_wer = float(parts[7].split(': ')[1])
            aed_wer = float(parts[8].split(': ')[1])

            self.train_loss_history.append(train_loss)
            self.train_ce_loss_history.append(train_ce_loss)
            self.train_ctc_loss_history.append(train_ctc_loss)
            self.valid_loss_history.append(valid_loss)
            self.valid_ctc_loss_history.append(valid_ctc_loss)
            self.valid_ce_loss_history.append(valid_ce_loss)
            self.valid_ctc_wer_history.append(ctc_wer)
            self.valid_aed_wer_history.append(aed_wer)
            
        except (IndexError, ValueError):
            print(f"格式錯誤，無法解析: {line}")

    epoch = len(self.train_loss_history)
    return epoch

def save_log(self, epoch, save_path):
    """儲存訓練過程的 loss 與 WER 紀錄"""
    with open(save_path, 'a') as f:
        f.write(f"Epoch: {epoch}, Train: {self.train_loss_history[-1]:.4f}, Valid: {self.valid_loss_history[-1]:.4f}, Train CTC: {self.train_ctc_loss_history[-1]:.4f}, Train CE: {self.train_ce_loss_history[-1]:.4f}, Valid CTC: {self.valid_ctc_loss_history[-1]:.4f}, Valid CE: {self.valid_ce_loss_history[-1]:.4f}, CTC WER: {self.valid_ctc_wer_history[-1]:.2f}, AED WER: {self.valid_aed_wer_history[-1]:.2f}\n")