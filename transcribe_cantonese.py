import argparse
import os
import sys
import shutil
import subprocess
import tempfile
from typing import List, Tuple

def ensure_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def resolve_inputs(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    exts = {".ogg", ".opus", ".m4a", ".mp3", ".wav"}
    if os.path.isdir(path):
        items = []
        for root, _, files in os.walk(path):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    items.append(os.path.join(root, f))
        return sorted(items)
    if os.path.splitext(path)[1].lower() in exts:
        return [path]
    raise ValueError("Unsupported input type")

def get_output_path(in_path: str, out_dir_or_file: str | None) -> str:
    if out_dir_or_file is None:
        base, _ = os.path.splitext(in_path)
        return base + ".txt"
    if os.path.isdir(out_dir_or_file):
        base = os.path.splitext(os.path.basename(in_path))[0]
        return os.path.join(out_dir_or_file, base + ".txt")
    if len(resolve_inputs(in_path)) == 1 and not os.path.isdir(out_dir_or_file):
        return out_dir_or_file
    raise ValueError("When processing multiple files, --output must be a directory")

def _convert_to_wav16k(src_path: str) -> str:
    dst_path = src_path + ".wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-f", "wav", dst_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst_path
    except Exception:
        return src_path

def transcribe_file(model_size: str, audio_path: str, language: str, timestamps: bool) -> Tuple[str, List[Tuple[float, float, str]]]:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="auto", compute_type="auto")
    proc_path = _convert_to_wav16k(audio_path)
    initial_prompt = "請用繁體中文並偏向粵語口語用字，例如：嘅、喺、佢、唔、咩、冇。"
    segments, info = model.transcribe(proc_path, language=language, task="transcribe", vad_filter=True, beam_size=5, temperature=0.0, initial_prompt=initial_prompt, condition_on_previous_text=True)
    parts = []
    segs = []
    for s in segments:
        text = s.text.strip()
        parts.append(text)
        if timestamps:
            segs.append((s.start, s.end, text))
    return "".join(parts), segs

def to_traditional(text: str) -> str:
    from opencc import OpenCC
    cc = OpenCC("s2t")
    return cc.convert(text)

def format_segments(segs: List[Tuple[float, float, str]]) -> str:
    rows = []
    for start, end, text in segs:
        rows.append(f"[{start:6.2f} → {end:6.2f}] {text}")
    return "\n".join(rows)

def main():
    parser = argparse.ArgumentParser(prog="transcribe_cantonese", description="將 WhatsApp 粵語語音轉為繁體中文文字")
    parser.add_argument("input", help="音訊檔或資料夾路徑（支援 .ogg/.opus/.m4a/.mp3/.wav）")
    parser.add_argument("-o", "--output", help="輸出檔或資料夾；未指定則在同目錄產生 .txt")
    parser.add_argument("--model", default="small", help="Whisper 模型大小，例如: tiny, base, small, medium, large-v3")
    parser.add_argument("--language", default="zh", help="語言代碼，預設 zh")
    parser.add_argument("--timestamps", action="store_true", help="輸出時間戳內容到檔案")
    args = parser.parse_args()

    if not ensure_ffmpeg():
        print("未偵測到 ffmpeg，請先安裝後再執行。", file=sys.stderr)
        sys.exit(2)

    targets = resolve_inputs(args.input)
    if not targets:
        print("未找到可處理的音訊檔。", file=sys.stderr)
        sys.exit(1)

    if args.output and not os.path.exists(args.output):
        parent = os.path.dirname(args.output)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        if len(targets) > 1:
            os.makedirs(args.output, exist_ok=True)

    for audio in targets:
        try:
            text, segs = transcribe_file(args.model, audio, args.language, args.timestamps)
            text_trad = to_traditional(text)
            out_path = get_output_path(audio, args.output)
            payload = text_trad
            if args.timestamps and segs:
                payload = payload + "\n\n" + format_segments(segs)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(payload)
            print(f"完成：{audio} → {out_path}")
        except Exception as e:
            print(f"失敗：{audio} → {e}", file=sys.stderr)
            continue

if __name__ == "__main__":
    main()
