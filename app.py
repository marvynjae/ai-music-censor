import hashlib, os, tempfile
import streamlit as st
import censor as C
from trimmer import prepare_preview, section_trimmer
from pipeline import run, render

PREVIEW_SECONDS = 30

# Whisper model: "base" (fast on CPU).
# Accuracy on sung or fast vocals is limited, so word timestamps can be imprecise.
# A larger model ("small, medium, turbo") improves this at the cost of speed.
# Full-track processing is slow on CPU and fast on a GPU.
WHISPER_MODEL = "base"

st.set_page_config(page_title="Song Profanity Censor", layout="centered", initial_sidebar_state="collapsed")

def load_css(file_name="styles.css"):
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as css_file:
            st.markdown(f"<style>{' '.join(css_file.read().split())}</style>", unsafe_allow_html=True)

load_css()

st.title("AI Profanity Detection & Censor")
st.caption("Upload a track, test a 30-second clip, and export the clean full song.")


def fmt(sec):
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"

def words_text(n):
    return "No explicit words found" if n == 0 else f"{n} explicit word{'s' if n != 1 else ''} censored"

def get_audio(res, fmt_type, method, pad):
    sig = (method, round(pad, 3), fmt_type)
    cache = res.setdefault("cache", {})
    if sig not in cache:
        cache.clear()
        y = render(res["source"], res["flagged"], method, pad)
        cache[sig] = C.to_mp3_bytes(y, C.SR) if fmt_type == "mp3" else C.to_wav_bytes(y, C.SR)
    return cache[sig]

f = st.file_uploader("Upload a song", type=["mp3", "wav", "flac", "m4a", "ogg"])
if f is None:
    st.stop()

data = f.getvalue()
key = hashlib.md5(data).hexdigest()

if st.session_state.get("track_key") != key:
    workdir = tempfile.mkdtemp()
    src = os.path.join(workdir, "source" + os.path.splitext(f.name)[1])

    with open(src, "wb") as fh:
        fh.write(data)

    with st.spinner("Loading song..."):
        audio = C.load(src)

    st.session_state.update(
        track_key=key,
        workdir=workdir,
        src=src,
        audio=audio,
        preview=None,
        full=None,
    )

workdir, src, audio = (st.session_state[k] for k in ("workdir", "src", "audio"))
sr = C.SR
total = audio.shape[-1] / sr

st.markdown(f"**{f.name}** · `{fmt(total)}`")

with st.expander("Censor settings", expanded=False):

    c1, c2, c3 = st.columns([1, 1.3, 1.2])

    with c1:
        method = st.selectbox(
            "Censor style",
            ["mute", "beep"],
            help="Choose how explicit words are covered: 'mute' silences the vocal cleanly, while 'beep' plays a classic censor tone over the word.",
        )

    with c2:
        pad = st.slider(
            "Padding (s)",
            0.0,
            0.4,
            0.1,
            0.01,
            help="Adds extra time before and after each explicit word so the start or end of the word doesn't leak through. Increase this if you still hear syllables cutting through.",
        )

    with c3:
        st.write("")
        use_sep = st.checkbox(
            "Preserve instrumental",
            value=True,
            help=(
                "Separates the vocals from the instrumental "
                "before censoring. This allows explicit words to be muted or "
                "beeped while the background music continues playing. "
                "Processing takes longer when enabled. Turn it off for faster "
                "processing, but censoring may also affect the instrumental."
            ),
        )

# =========================================
# 1. PREVIEW TRACK
# =========================================

st.divider()
st.subheader("1. Preview Track (30s)")

section_len = min(PREVIEW_SECONDS, total)
preview_file = prepare_preview(audio, sr, key)

start = min(
    section_trimmer(preview_file, section_len, key=f"trim_{key}"),
    max(0.0, total - section_len),
)

segment = audio[:, int(start * sr): int((start + section_len) * sr)]

if st.button("Scan & Censor Selected Portion", type="secondary", use_container_width=True):
    seg_path = os.path.join(workdir, "segment.wav")
    out_path = os.path.join(workdir, "segment_censored.wav")

    C.save(seg_path, segment, sr)

    with st.spinner("Processing your preview..."):
        try:
            rep = run(
                seg_path,
                method=method,
                model_name=WHISPER_MODEL,
                pad=pad,
                use_separation=use_sep
            )
        except Exception as e:
            st.error(str(e))
            st.stop()

    st.session_state.preview = {
        "start": start,
        "use_sep": use_sep,
        "segment": segment,
        "flagged": rep["flagged"],
        "transcript": rep["transcript"],
        "timings": rep["timings"],
        "source": rep["source"],
    }


pv = st.session_state.get("preview")
pv_valid = pv is not None and pv["start"] == start and pv.get("use_sep") == use_sep

if pv_valid:

    tab_c, tab_o = st.tabs(["Censored Preview", "Original Uncensored"])

    with tab_c:
        st.audio(get_audio(pv, "wav", method, pad), format="audio/wav")

    with tab_o:
        st.audio(C.to_wav_bytes(pv["segment"], sr), format="audio/wav")

    with st.expander(f"Censored words ({len(pv['flagged'])})", expanded=False):
        if pv["flagged"]:
            st.table([
                {
                    "Word": w["word"],
                    "Start": f"{round(pv['start'] + w['start'], 2)}s",
                    "End": f"{round(pv['start'] + w['end'], 2)}s",
                }
                for w in pv["flagged"]
            ])
        else:
            st.caption("No explicit words detected in this section.")

elif pv:
    st.info("You moved the selection or changed a setting. Press **Scan & Censor Selected Portion** to update.")


# =========================================
# 2. EXPORT CLEAN FULL TRACK
# =========================================


st.divider()
st.subheader("2. Export Clean Full Track")

if pv_valid:
    st.caption(
        "Your preview is complete. Export the entire song using the selected censor settings."
    )
else:
    st.caption(
        "Complete the preview scan above to unlock full-track export."
    )

if st.button(
    "Censor Full Track",
    type="primary",
    use_container_width=True,
    disabled=not pv_valid
):
    st.session_state.full = None

    progress = st.progress(0, text="Preparing your track...")

    def update_progress(percent, message):
        progress.progress(percent, text=message)

    try:
        rep = run(
            src,
            method=method,
            model_name=WHISPER_MODEL,
            pad=pad,
            use_separation=use_sep,
            progress_callback=update_progress,
        )

        st.session_state.full = {
            "flagged": rep["flagged"],
            "n": len(rep["flagged"]),
            "timings": rep["timings"],
            "source": rep["source"],
        }

        progress.progress(93, text="Preparing clean MP3...")

        get_audio(st.session_state.full, "mp3", method, pad)

        progress.progress(100, text="Censoring complete!")

    except Exception as e:
        st.session_state.full = None
        st.error(str(e))
        st.stop()

    finally:
        progress.empty()

full = st.session_state.get("full")

if full:
    st.success(
        f"Your clean track is ready · {words_text(full['n'])}"
    )

    st.download_button(
        "Download Clean Track",
        get_audio(full, "mp3", method, pad),
        os.path.splitext(f.name)[0] + "-censored.mp3",
        "audio/mpeg",
        type="secondary",
        use_container_width=True,
    )
