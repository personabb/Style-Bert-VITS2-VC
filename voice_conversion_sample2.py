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
#from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.tts_model_nog import TTSModel
import sounddevice as sd

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
import librosa

#録音用のクラス
class Recorder:
    def __init__(self,):
                 
            self.fs = 44100
            self.silence_threshold = 1.0
            self.min_duration = 0.1
            self.amplitude_threshold = 0.03
            self.start_threshold = 0.3

    def speech2audio(self):
        record_Flag = False

        non_recorded_data = []
        recorded_audio = []
        silent_time = 0
        input_time = 0
        start_threshold = 0.3
        all_time = 0
        
        with sd.InputStream(samplerate=self.fs, channels=1) as stream:
            while True:
                data, overflowed = stream.read(int(self.fs * self.min_duration))
                all_time += 1
                if all_time == 10:
                    print("stand by ready OK")
                elif all_time >=10:
                    if np.max(np.abs(data) > self.amplitude_threshold) and not record_Flag:
                        input_time += self.min_duration
                        if input_time >= start_threshold:
                            record_Flag = True
                            print("recording...")
                            recorded_audio=non_recorded_data[int(-1*start_threshold*10)-2:]  

                    else:
                        input_time = 0

                    if overflowed:
                        print("Overflow occurred. Some samples might have been lost.")
                    if record_Flag:
                        recorded_audio.append(data)

                    else:
                        non_recorded_data.append(data)

                    if np.all(np.abs(data) < self.amplitude_threshold):
                        silent_time += self.min_duration
                        if (silent_time >= self.silence_threshold) and record_Flag:
                            print("finished")
                            record_Flag = False
                            break
                    else:
                        silent_time = 0

        audio_data = np.concatenate(recorded_audio, axis=0)

        return audio_data

warnings.filterwarnings("ignore")


path_config = get_path_config()

# リサンプリングフラグ(入力音声が44100Hzでない場合にTrueにする)
resample_Flag = True
target_sr = 44100

# "model_assets"フォルダ内の学習済み重みが格納されているフォルダ名を指定
model_name = "sample"
# GPUで動かす場合は"cuda"を指定
device = "cpu"
# VC_FlagがTrueの場合は音声変換を行う。Falseの場合はテキストから音声合成を行う
VC_Flag = True
# recoed_audio_FlagがTrueの場合は録音を行う。Falseの場合はファイルから音声を読み込む
recoed_audio_Flag = False
# 音声変換の場合のみ有効。変換元の話者IDを指定
reference_speraker_id = 0
#変換先の話者IDを指定
target_speaker_id = 1
# 重みファイル名を指定
weight_name = "G_xxxx.pth"
# テキストを指定。VC_FlagがFalseの場合のみ有効
text = "新しいPCが欲しいです"
# 音声ファイル名を指定。VC_FlagがTrueかつrecoed_audio_FlagがFalseの場合のみ有効
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

if VC_Flag:
    if recoed_audio_Flag:
        recorder = Recorder()
        # 録音(44100Hz) -> 音声データ取得 -> numpy配列に変換 
        record_audio = recorder.speech2audio()
        record_audio = record_audio.flatten().astype(np.float32)*32768
        sample_rate = 44100
        record_audio = np.array(record_audio, dtype=np.int16)
        # numpyの音声データをWAVファイルとして保存
        output_filename = model_path / 'user_input.wav'
        write(output_filename, sample_rate, record_audio)
    else:
        # ファイルパスを指定
        file_path = model_path / audio_file
        # wavファイルを読み込み
        sample_rate, record_audio = wavfile.read(file_path)

    
        if resample_Flag:
            # リサンプリング
            # librosaのリサンプリング関数はデータをfloat32型に変換するため、int16型からの変換を行います
            record_audio = record_audio.astype(float) / 32768.0  # int16をfloat32にスケーリング
            record_audio = librosa.resample(record_audio, orig_sr=sample_rate, target_sr=target_sr)

            # リサンプリング後の音声を保存
            # 出力する前にデータを再びint16に変換
            record_audio = (record_audio * 32768).astype('int16')
        else:
            pass

    # 推論
    sr, audio = model.infer_audio(record_audio,ref_speaker_id=reference_speraker_id, tar_speaker_id=target_speaker_id)
else:
    # 推論
    sr, audio = model.infer(text,speaker_id=target_speaker_id)

# 生成された音声の再生
sd.play(audio, sr)
sd.wait()

# numpyの音声データをWAVファイルとして保存
output_filename = model_path / 'vc_output.wav'
write(output_filename, sr, audio)


