import io
import os
import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
from opencc import OpenCC
try:
    import av  # may be unavailable in some cloud environments
except Exception:
    av = None
import wave
import re

@st.cache_resource
def load_model(size: str):
    return WhisperModel(size, device="auto", compute_type="auto")

def decode_to_np(file_bytes: bytes, filename: str | None = None) -> tuple[np.ndarray, int]:
    bio = io.BytesIO(file_bytes)
    if av is not None:
        try:
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
        except Exception:
            pass
    # Fallback: decode WAV with python stdlib
    try:
        bio.seek(0)
        with wave.open(bio, "rb") as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if sw == 1:
            x = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            x = (x - 128.0) / 128.0
        elif sw == 2:
            x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            x = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError("Unsupported WAV sample width")
        if nch > 1:
            x = x.reshape(-1, nch).mean(axis=1)
        return x.astype(np.float32), sr
    except Exception as e:
        raise RuntimeError("音訊解碼失敗。請上傳 WAV 檔，或在雲端啟用 AV/FFmpeg。") from e

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

def to_mandarin_traditional(text: str) -> str:
    reps = [
        ("佢哋", "他們"),
        ("佢", "他"),
        ("你哋", "你們"),
        ("噉", "這樣"),
        ("咁", "這麼"),
        ("睇", "看"),
        ("曬", "全部"),
        ("啲", "些"),
        ("嗰啲", "那些"),
        ("嗰個", "那個"),
        ("嗰度", "那裡"),
        ("嗰邊", "那邊"),
        ("呢度", "這裡"),
        ("呢啲", "這些"),
        ("喺", "在"),
        ("嚟", "來"),
        ("返嚟", "回來"),
        ("出嚟", "出來"),
        ("食", "吃"),
        ("飲", "喝"),
        ("攞", "拿"),
        ("畀", "給"),
        ("俾", "給"),
        ("講", "說"),
        ("話", "說"),
        ("唔", "不"),
        ("冇", "沒有"),
        ("係", "是"),
        ("唔係", "不是"),
        ("點樣", "如何"),
        ("點", "怎麼"),
        ("一陣", "一會兒"),
        ("依家", "現在"),
        ("嗰陣", "當時"),
        ("啱", "合適"),
        ("啩", "吧"),
        ("㗎", "啊"),
        ("喎", "啊"),
        ("啦", ""),
        ("呀", ""),
        ("啊", ""),
        ("蚊", "元"),
    ]
    for a, b in reps:
        text = text.replace(a, b)
    text = re.sub(r"([的是了呢吧嗎啊呀啦喎㗎]+)$", "", text)
    text = re.sub(r"\b[Ss]erver\b", "伺服器", text)
    text = re.sub(r"\b[Dd]eploy\b", "部署", text)
    text = re.sub(r"\bupdate\b", "更新", text, flags=re.IGNORECASE)
    return text

def transcribe_bytes(model: WhisperModel, data: bytes, mandarin: bool = False, filename: str | None = None) -> str:
    audio, sr = decode_to_np(data, filename=filename)
    audio = resample_to_16k(audio, sr)
    if mandarin:
        initial_prompt = "請以繁體中文輸出，使用普通話書面用語，避免粵語口語詞彙。"
    else:
        initial_prompt = "請用繁體中文並偏向粵語口語用字，例如：嘅、喺、佢、唔、咩、冇。"
    segments, info = model.transcribe(audio, language="zh", task="transcribe", vad_filter=True, beam_size=5, temperature=0.0, initial_prompt=initial_prompt, without_timestamps=True, condition_on_previous_text=True)
    parts = []
    for s in segments:
        parts.append(s.text.strip())
    return "".join(parts)

st.title("粵語語音轉繁體中文")
col1, col2 = st.columns(2)
with col1:
    model_size = st.selectbox("模型", ["small", "medium", "large-v3"], index=1)
with col2:
    precise = st.checkbox("精準模式（使用 large-v3）", value=False)
if precise:
    model_size = "large-v3"
uploaded = st.file_uploader("選擇語音檔（可多選）", type=["ogg", "opus", "m4a", "mp3", "wav"], accept_multiple_files=True)
style = st.radio("輸出風格", ["粵語（繁中）", "普通話（繁中）"], index=1)
if av is None:
    st.info("雲端可能未安裝音訊編解碼器。若上傳非 WAV 檔可能解碼失敗，建議先轉為 WAV。")
go = st.button("開始轉換")
if go and uploaded:
    try:
        model = load_model(model_size)
        texts = []
        for f in uploaded:
            try:
                text = transcribe_bytes(model, f.getvalue(), mandarin=(style == "普通話（繁中）"), filename=f.name)
                text_trad = to_traditional(text)
                if style == "普通話（繁中）":
                    text_trad = to_mandarin_traditional(text_trad)
                texts.append(text_trad)
            except Exception as e:
                texts.append(f"{f.name}：{e}")
        st.text_area("結果", "\n\n".join(texts), height=300)
    except Exception as e:
        st.error("執行時發生錯誤。")
        st.exception(e)
