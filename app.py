
import hashlib, os, tempfile
import streamlit as st
import censor as C
from trimmer import prepare_preview, section_trimmer
from pipeline import run, render

PREVIEW_SECONDS = 30
WHISPER_MODEL = "turbo"   # "base" = faster, "small" = better, "turbo"/"medium" = best but slow on CPU

st.set_page_config(page_title="Song Profanity Censor")
st.title("AI Profanity Detection & Censoring in Music")


def fmt(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"


def fmt_dur(sec):
    sec = int(round(sec))
    return f"{sec // 60} min {sec % 60} s" if sec >= 60 else f"{sec} s"


def words_text(n):
    return "No explicit words found" if n == 0 else f"{n} explicit word{'s' if n != 1 else ''} censored"


def get_audio(res, fmt="wav"):
    """Re-render only when padding/method changed - no need to re-run Demucs or Whisper."""
    sig = (method, round(pad, 3), fmt)
    cache = res.setdefault("cache", {})
    if sig not in cache:
        cache.clear()
        y = render(res["source"], res["flagged"], method, pad)
        cache[sig] = C.to_mp3_bytes(y, sr) if fmt == "mp3" else C.to_wav_bytes(y, sr)
    return cache[sig]


def show_details(t):
    with st.expander("Processing details"):
        if "separation_s" in t:
            st.write(f"Vocal separation: {fmt_dur(t['separation_s'])}")
        st.write(f"Speech recognition: {fmt_dur(t['transcription_s'])}")


f = st.file_uploader("Upload a song", type=["mp3", "wav", "flac", "m4a", "ogg"])
if f is None:
    st.info("Upload a song to begin.")
    st.stop()

# ---- Load the song once per upload and keep it in session state ----
data = f.getvalue()
key = hashlib.md5(data).hexdigest()
if st.session_state.get("track_key") != key:
    workdir = tempfile.mkdtemp()
    src = os.path.join(workdir, "source" + os.path.splitext(f.name)[1])
    with open(src, "wb") as fh:
        fh.write(data)
    with st.spinner("Loading song..."):
        audio = C.load(src)
    st.session_state.update(track_key=key, workdir=workdir, src=src,
                            audio=audio, preview=None, full=None)

workdir, src, audio = (st.session_state[k] for k in ("workdir", "src", "audio"))
sr = C.SR
total = audio.shape[-1] / sr
st.caption(f"{f.name} - length {fmt(total)}")

# ---- Settings ----
with st.expander("⚙️ Censor Settings", expanded=False):
    method = st.selectbox("Censor method", ["mute", "beep"])

    pad = st.slider(
        "Padding around words (s)",
        0.0, 0.4, 0.1, 0.01
    )

    use_sep = st.checkbox(
        "Separate vocals first (Demucs) - slower, better quality",
        True
    )

# ---- 1. Choose and preview a section ----
st.subheader("1. Choose a 30-sec preview section")
st.caption("Select a section of the song to preview the censoring. You can drag the box in the waveform or use the slider below. "
           "The preview will play.")
section_len = min(PREVIEW_SECONDS, total)
preview_file = prepare_preview(audio, sr, key)
start = min(section_trimmer(preview_file, section_len, key=f"trim_{key}"),
            max(0.0, total - section_len))
segment = audio[:, int(start * sr): int((start + section_len) * sr)]

# ---- 2. Censor the selected section ----
st.subheader("2. Censor this section")
if st.button("Censor this section", type="primary"):
    seg_path = os.path.join(workdir, "segment.wav")
    out_path = os.path.join(workdir, "segment_censored.wav")
    C.save(seg_path, segment, sr)
    with st.spinner("Processing section... this can take a few minutes on CPU"):
        try:
            rep = run(seg_path, out_path, method, WHISPER_MODEL, pad, use_sep)
        except Exception as e:
            st.error(str(e)); st.stop()
    st.session_state.preview = {"start": start, "flagged": rep["flagged"],
                                "transcript": rep["transcript"],
                                "timings": rep["timings"], "source": rep["source"]}

pv = st.session_state.get("preview")
if pv and pv["start"] == start:
    st.success(f"Done! {words_text(len(pv['flagged']))} in this section.")
    show_details(pv["timings"])
    if pv["flagged"]:
        with st.expander(f"Censored words ({len(pv['flagged'])})"):
            st.table([{"word": w["word"], "start (s)": round(pv["start"] + w["start"], 2),
                       "end (s)": round(pv["start"] + w["end"], 2)} for w in pv["flagged"]])
    with st.expander("Transcript of this section"):
        st.write(" ".join(w["word"] for w in pv["transcript"]))
    st.caption("Censored section - changing padding or method above updates this instantly.")
    st.audio(get_audio(pv), format="audio/wav")
elif pv:
    st.caption("You moved the slider - press 'Censor this section' to preview the new section.")

# ---- 3. Full track ----
st.subheader("3. Censor and download the full track")
st.caption("Happy with the preview? This processes the whole song and can take several minutes.")
if st.button("Censor full track"):
    out_path = os.path.join(workdir, "full_censored.wav")
    with st.spinner("Processing full track... please keep this tab open"):
        try:
            rep = run(src, out_path, method, WHISPER_MODEL, pad, use_sep)
        except Exception as e:
            st.error(str(e)); st.stop()
    st.session_state.full = {"flagged": rep["flagged"], "n": len(rep["flagged"]),
                             "timings": rep["timings"], "source": rep["source"]}

full = st.session_state.get("full")
if full:
    st.success(f"Your full track is ready. {words_text(full['n'])}.")
    show_details(full["timings"])
    st.download_button("Download full censored track (MP3)", get_audio(full, "mp3"),
                       os.path.splitext(f.name)[0] + "-censored.mp3", "audio/mpeg")
