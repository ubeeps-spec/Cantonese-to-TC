import os
import tempfile
import subprocess
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

def _convert_to_wav16k(src_path):
    dst_path = src_path + ".wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", "-f", "wav", dst_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst_path
    except Exception:
        return src_path

def transcribe_file(model_size, audio_path, language, timestamps):
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

def to_traditional(text):
    from opencc import OpenCC
    cc = OpenCC("s2t")
    return cc.convert(text)

def format_segments(segs):
    rows = []
    for start, end, text in segs:
        rows.append(f"[{start:6.2f} → {end:6.2f}] {text}")
    return "\n".join(rows)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    files = request.files.getlist("files")
    model = request.form.get("model", "small")
    language = request.form.get("language", "zh")
    timestamps = request.form.get("timestamps") == "on"
    results = []
    tmpdir = tempfile.mkdtemp()
    for f in files:
        if not f or not f.filename:
            continue
        name = secure_filename(f.filename)
        path = os.path.join(tmpdir, name)
        f.save(path)
        try:
            text, segs = transcribe_file(model, path, language, timestamps)
            text_trad = to_traditional(text)
            payload = text_trad
            if timestamps and segs:
                payload = payload + "\n\n" + format_segments(segs)
            results.append({"filename": name, "text": payload})
        except Exception as e:
            results.append({"filename": name, "text": f"錯誤：{e}"})
    return render_template("result.html", results=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
