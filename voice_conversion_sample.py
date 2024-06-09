
import csv
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import get_path_config
from style_bert_vits2.logging import logger
from style_bert_vits2.tts_model import TTSModel
import sounddevice as sd

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write

warnings.filterwarnings("ignore")

path_config = get_path_config()

#"model_assets"フォルダ内の学習済み重みが格納されているフォルダ名を指定
model_name = "sample"
#GPUで動かす場合は"cuda"を指定
device = "cpu"
#net_gの重みを読み込む。
weight_name = "G_xxxxx.pth"
#変換前の入力音声
audio_file = "inputs.wav"

model_path = path_config.assets_root / model_name
model_file = model_path / weight_name

def get_model(model_file: Path):
    return TTSModel(
        model_path=model_file,
        config_path=model_path/ "config.json",
        style_vec_path=model_path/ "style_vectors.npy",
        device=device,
    )

model = get_model(model_file)

# ファイルパスを指定
file_path = model_path / audio_file
# wavファイルを読み込み
sample_rate, record_audio = wavfile.read(file_path)

# 推論
sr, audio = model.infer_audio(record_audio,sample_rate = sample_rate)

#生成された音声の再生
sd.play(audio, sr)
sd.wait()

# numpyの音声データをWAVファイルとして保存
output_filename = model_path / 'vc_output.wav'
write(output_filename, sr, audio)
