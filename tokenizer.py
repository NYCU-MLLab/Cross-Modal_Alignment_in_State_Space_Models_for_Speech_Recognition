import sentencepiece as spm
import re
from tqdm import tqdm

def normalize_text(text):
    """
    正規化文本：轉為小寫、去除多餘的空格和特殊字符。
    """

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?\'\`]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def extract_text_and_save(dataset, output_file):
    """
    從 LibriSpeech 數據集提取文本，並保存到文件。
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in tqdm(enumerate(dataset)):
            transcript = item[2]  # 只提取轉錄文本
            # normalized_transcript = normalize_text(transcript)
            f.write(transcript.lower() + "\n")
    print(f"文本已保存至 {output_file}")

def train_sentencepiece(corpus_file, model_prefix, vocab_size, special_tokens=[]):
    """
    使用 SentencePiece 訓練分片模型。
    """
    spm.SentencePieceTrainer.Train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,  # 支援完整字符集
        model_type="bpe",  # 或選擇 'unigram', 'char', 'word'
        user_defined_symbols=special_tokens
    )
    print(f"SentencePiece 模型訓練完成，模型前綴為: {model_prefix}")

class SentencePieceTransform:
    """Maps subwords to integers and vice versa using SentencePiece"""
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def text_to_int(self, text):
        """ Use the SentencePiece tokenizer to convert text to an integer sequence """
        subwords = self.sp.EncodeAsPieces(text.lower())
        return [self.sp.PieceToId(subword) for subword in subwords]

    def int_to_text(self, labels):
        """ Use the SentencePiece tokenizer to convert integer labels to a text sequence """
        return self.sp.decode(labels)