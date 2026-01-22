import io
import os
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from opencc import OpenCC
import av

@st.cache_resource
def load_model(size: str):
    return WhisperModel(size, device="auto", compute_type="auto")

def decode_to_np(file_bytes: bytes) -> tuple[np.ndarray, int]:
    bio = io.BytesIO(file_bytes)
    container = av.open(bio)
    stream = next(s for s in container.streams if s.type == "audio")
    samples = []
    sr = None
    for frame in container.decode(stream):
        arr = frame.to_ndarray()
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
            arr /= 32768.0
        samples.append(arr)
        sr = frame.sample_rate
    if not samples:
        return np.zeros((0,), dtype=np.float32), 16000
    audio = np.concatenate(samples).astype(np.float32)
    return audio, sr or 16000

def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x
    n = int(round(len(x) * 16000 / sr))
    xp = np.linspace(0, len(x) - 1, num=len(x), dtype=np.float32)
    fp = x
    x_new = np.linspace(0, len(x) - 1, num=n, dtype=np.float32)
    y = np.interp(x_new, xp, fp).astype(np.float32)
    return y

def to_traditional(text: str) -> str:
    cc = OpenCC("s2t")
    return cc.convert(text)

def transcribe_bytes(model: WhisperModel, data: bytes) -> str:
    audio, sr = decode_to_np(data)
    audio = resample_to_16k(audio, sr)
    initial_prompt = "請用繁體中文並偏向粵語口語用字，例如：嘅、喺、佢、唔、咩、冇。"
    segments, info = model.transcribe(audio, language="zh", task="transcribe", vad_filter=True, beam_size=5, temperature=0.0, initial_prompt=initial_prompt, without_timestamps=True, condition_on_previous_text=True)
    parts = []
    for s in segments:
        parts.append(s.text.strip())
    return "".join(parts)

st.title("粵語語音轉繁體中文")
model_size = st.selectbox("模型", ["small", "medium", "large-v3"], index=1)
uploaded = st.file_uploader("選擇語音檔（可多選）", type=["ogg", "opus", "m4a", "mp3", "wav"], accept_multiple_files=True)
go = st.button("開始轉換")
if go and uploaded:
    model = load_model(model_size)
    texts = []
    for f in uploaded:
        try:
            text = transcribe_bytes(model, f.getvalue())
            text_trad = to_traditional(text)
            texts.append(text_trad)
        except Exception as e:
            texts.append(f"錯誤：{e}")
    st.text("\n\n".join(texts))
