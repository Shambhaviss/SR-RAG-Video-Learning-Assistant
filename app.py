import json
import html
import uuid
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import sr_rag_backend as rag


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SR-RAG | Video Learning Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODERN DARK THEME & STYLING
# ============================================================

st.markdown(
    """
    <style>
    /* ---------------- Global App Canvas ---------------- */
    .stApp {
        background-color: #0b1120;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none;
    }
    [data-testid="stHeader"] > * {
        pointer-events: auto;
    }
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDeployButton"] {
        visibility: hidden !important;
        height: 0px !important;
    }
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1400px;
    }

    /* ---------------- Sidebar ---------------- */
    [data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] > label {
        display: block;
        font-size: 0.72rem !important;
        font-weight: 800 !important;
        letter-spacing: 1.8px !important;
        color: #64748b !important;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: #131e33;
        border: 1px solid #1e2e4a;
        border-radius: 10px;
        padding: 9px 14px !important;
        margin: 0 !important;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex !important;
        align-items: center;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #1a2944 !important;
        border-color: #3b82f6 !important;
        transform: translateX(3px);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"] {
        display: none !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        border-color: #60a5fa !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p,
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 750 !important;
        letter-spacing: -0.3px;
    }
    p, label, .stMarkdown {
        color: #94a3b8;
    }

    /* ---------------- Cards & Panels ---------------- */
    .hero {
        background: linear-gradient(135deg, #13233f 0%, #162a4d 50%, #17203b 100%);
        border: 1px solid #1e3a63;
        border-radius: 16px;
        padding: 30px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .hero-label {
        color: #60a5fa;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .hero-title {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 8px;
        line-height: 1.25;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 10px;
        line-height: 1.5;
        max-width: 800px;
    }
    .metric-card {
        background: #111e36;
        border: 1px solid #1e3458;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 105px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    .metric-value {
        color: #f1f5f9;
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 8px;
        font-variant-numeric: tabular-nums;
        overflow-wrap: anywhere;
    }
    .panel {
        background: #0f1c30;
        border: 1px solid #1e3559;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 16px;
    }
    .panel-title {
        color: #f8fafc;
        font-size: 1.08rem;
        font-weight: 750;
        margin-bottom: 8px;
    }

    /* Duration Guide Box */
    .duration-guide {
        background: #0d1a2d;
        border: 1px solid #1e3a63;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }

    .badge-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 750;
        margin-right: 8px;
    }
    .badge-fast {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-slow {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    div.stButton > button {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 9px 22px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.45) !important;
        transform: translateY(-1px);
    }
    div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input {
        background: #0d1829 !important;
        color: #f1f5f9 !important;
        border: 1px solid #223755 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stExpander"] {
        background: #0e1b2f !important;
        border: 1px solid #1b2f4d !important;
        border-radius: 12px !important;
        margin-bottom: 10px;
    }
    hr {
        border-color: #1e293b !important;
    }
    .translation-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_timestamp(value):
    try:
        seconds = float(value)
        if seconds < 0:
            return str(value)
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
    except (TypeError, ValueError):
        return str(value)


def show_status(status):
    value = str(status or "NOT_AVAILABLE").upper()
    if value in {"SUPPORTED", "ANSWERED", "NO_HALLUCINATION_DETECTED", "ACCEPTABLE", "LOADED", "AVAILABLE"}:
        st.success(value.replace("_", " ").title())
    elif value in {"ABSTAINED", "PARTIALLY_SUPPORTED", "NEEDS_REVIEW", "NOT_RUN", "NOT_AVAILABLE", "UNKNOWN"}:
        st.warning(value.replace("_", " ").title())
    else:
        st.error(value.replace("_", " ").title())


def get_history():
    history_path = Path(getattr(rag, "OUTPUT_DIR", Path.cwd() / "outputs")) / "decision_trace.json"
    if not history_path.exists():
        return [], history_path
    try:
        with history_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return (data if isinstance(data, list) else []), history_path
    except (json.JSONDecodeError, OSError):
        return None, history_path


def artifact_status(path):
    path = Path(path)
    if not path.exists():
        return "Not found"
    if path.is_dir():
        try:
            return f"Present · {sum(1 for _ in path.iterdir())} item(s)"
        except OSError:
            return "Present"
    return f"Present · {path.stat().st_size:,} bytes"


def save_note(question, answer, video_name=""):
    notes = st.session_state.setdefault("saved_notes", [])
    note_id = str(uuid.uuid4())
    notes.append({
        "id": note_id,
        "question": question,
        "answer": answer,
        "video": video_name,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })


# ============================================================
# PERSISTENT ACTIVE VIDEO
# ============================================================

OUTPUT_DIR = Path(getattr(rag, "OUTPUT_DIR", Path.cwd() / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_VIDEO_FILE = OUTPUT_DIR / "active_video.json"


def save_active_video_record(video_path, processing_result):
    video_path = Path(video_path).resolve()
    payload = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "processing_result": processing_result,
    }
    temp_file = ACTIVE_VIDEO_FILE.with_suffix(".tmp")
    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
    temp_file.replace(ACTIVE_VIDEO_FILE)


def load_active_video_record():
    if not ACTIVE_VIDEO_FILE.exists():
        return None
    try:
        with ACTIVE_VIDEO_FILE.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        video_path = Path(payload.get("video_path", ""))
        if video_path.is_file():
            return payload
    except Exception:
        return None
    return None


# ============================================================
# SESSION DEFAULTS
# ============================================================

st.session_state.setdefault("saved_notes", [])
st.session_state.setdefault("qa_history", [])
st.session_state.setdefault("latest_result", None)
st.session_state.setdefault("latest_question", "")
st.session_state.setdefault("summary_result", None)

if not st.session_state.get("active_video"):
    saved = load_active_video_record()
    if saved:
        st.session_state["active_video"] = saved.get("video_name")
        st.session_state["video_path"] = saved.get("video_path")
        st.session_state["video_processing_result"] = saved.get("processing_result")


# ============================================================
# SLEEK SIDEBAR NAVIGATION
# ============================================================

pages = [
    "🏠  Home",
    "📥  Upload Video",
    "📑  Understand Video",
    "💬  Ask Questions",
    "📝  My Notes",
    "⚙️  Advanced Details",
]

page_name_map = {
    "🏠  Home": "Home",
    "📥  Upload Video": "Upload Video",
    "📑  Understand Video": "Understand Video",
    "💬  Ask Questions": "Ask Questions",
    "📝  My Notes": "My Notes",
    "⚙️  Advanced Details": "Advanced Details",
}

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; margin-bottom: 20px;">
            <div style="font-size: 28px; background: #1e3a8a; border-radius: 12px; padding: 6px 10px; border: 1px solid #3b82f6;">🧠</div>
            <div>
                <div style="font-size: 19px; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">SR-RAG</div>
                <div style="font-size: 11px; color: #60a5fa; font-weight: 700; letter-spacing: 0.8px;">VIDEO LEARNING ASSISTANT</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_target = st.session_state.pop("navigation_target", "Home")
    default_idx = 0
    for i, p_label in enumerate(pages):
        if page_name_map[p_label] == nav_target:
            default_idx = i
            break

    selected_raw = st.radio("NAVIGATION", pages, index=default_idx)
    page = page_name_map[selected_raw]

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 0.72rem; font-weight: 800; letter-spacing: 1.8px; color: #64748b; margin-bottom: 10px; text-transform: uppercase;">
            SYSTEM STATUS
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background: #111e33; border: 1px solid #1e3556; border-radius: 12px; padding: 14px; font-size: 0.82rem;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom: 8px;">
                <span style="height: 9px; width: 9px; background-color: #10b981; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #10b981;"></span>
                <span style="color: #f1f5f9; font-weight: 700;">Backend Connected</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.76rem; line-height: 1.6;">
                <div>• Embeddings: <b style="color:#cbd5e1;">BGE-M3 (1024d)</b></div>
                <div>• LLM: <b style="color:#cbd5e1;">{getattr(rag, 'LLM_MODEL', 'Groq Llama/GPT')}</b></div>
                <div>• Script Engine: <b style="color:#34d399;">Devanagari Normalizer</b></div>
                <div>• Adaptive Retrieval: <b style="color:#38bdf8;">Active</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.get("active_video"):
        st.markdown(
            f"""
            <div style="margin-top: 10px; background: #0c1a2e; border: 1px solid #1e3a63; border-radius: 10px; padding: 10px 12px; font-size: 0.78rem;">
                <div style="color: #60a5fa; font-weight: 750;">ACTIVE VIDEO</div>
                <div style="color: #f8fafc; font-weight: 600; margin-top: 3px; word-break: break-all;">
                    {st.session_state['active_video']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.caption("SR-RAG Research · Multilingual Video Q&A")


# ============================================================
# PAGE 1: HOME
# ============================================================

if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">LEARN FROM EDUCATIONAL VIDEOS</div>
            <div class="hero-title">Understand more. Ask better questions.</div>
            <div class="hero-subtitle">
                Upload educational videos, explore transcripts in authentic Hindi Devanagari or English,
                and ask complex questions. Every answer is cross-checked against transcript evidence with self-reflective verification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chunks = getattr(rag, "KNOWLEDGE_BASE", [])
    history, _ = get_history()
    history_count = len(history) if isinstance(history, list) else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Transcript passages", len(chunks))
    with c2:
        metric_card("Saved questions", len(st.session_state["qa_history"]))
    with c3:
        metric_card("Processed decisions", history_count)

    st.write("")
    st.markdown("### Get started")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">📹 1. Upload a video</div>
                <p>Upload educational videos in MP4, AVI, MOV, or MKV. SR-RAG extracts audio,
                auto-detects Hindi/Hinglish speech, prevents false Urdu Arabic script, and builds searchable BGE-M3 passages.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to video upload", key="home_upload"):
            st.session_state["navigation_target"] = "Upload Video"
            st.rerun()
    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">💬 2. Ask questions</div>
                <p>Ask technical questions about the video. Review evidence-grounded answers
                in English, Hindi, Hinglish, or Urdu with rigorous claim verification.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Ask a question", key="home_ask"):
            st.session_state["navigation_target"] = "Ask Questions"
            st.rerun()

    st.markdown("### Key Improvements & Fixes in this Release")
    st.markdown(
        """
        - **Hindi Auto-Detection & Devanagari Script Enforcement**: Disambiguates Hindi from Urdu phonetics so spoken Hindi is accurately transcribed in Devanagari script instead of Urdu Arabic script.
        - **Real-time Progress Bar & ETA Notification**: Live percentage, elapsed time, and countdown timer showing remaining wait time.
        - **Video Duration Performance Guidelines**: Fast processing recommendations (<= 10 mins) with time estimates.
        - **Interactive Translation Toggles**: Instantly switch between Original, English, Hindi, Urdu, and Hinglish.
        - **Adaptive Retrieval**: Expands query context and neighbor windows whenever initial retrieval confidence is below 0.35.
        """
    )


# ============================================================
# PAGE 2: UPLOAD VIDEO (WITH PROGRESS BAR, REMAINING TIME & DURATION GUIDELINES)
# ============================================================

elif page == "Upload Video":
    st.markdown("## Upload a video")
    st.caption("Create a searchable transcript and prepare the video for question answering.")

    # -------------------------------------------------------------
    # DURATION & PERFORMANCE GUIDELINES NOTICE
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="duration-guide">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#60a5fa; font-weight:750; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px;">
                    ⚡ Video Duration & Performance Guidelines
                </span>
                <span class="badge-tag badge-fast">Optimal: ≤ 10 min</span>
            </div>
            <div style="font-size:0.85rem; color:#cbd5e1; line-height:1.5;">
                • <b>Fast Processing Range (≤ 10 minutes)</b>: Processes rapidly in <b>~30–60 seconds</b> with quick transcription and BGE-M3 embedding.<br>
                • <b>Standard Range (10–30 minutes)</b>: Takes approximately <b>1–3 minutes</b> depending on audio length.<br>
                • <b>Extended Range (> 30 minutes)</b>: Will take <b>3–6+ minutes</b> due to dense chunking and multi-pass embedding.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an educational video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV and MKV. Videos under 10 minutes process fastest.",
    )

    if uploaded_file is not None:
        st.video(uploaded_file)
        file_size_mb = uploaded_file.size / (1024 * 1024)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.write(f"**Selected file:** `{uploaded_file.name}`")
            st.write(f"**File size:** `{file_size_mb:.2f} MB`")
        with col_f2:
            language_option = st.selectbox(
                "Spoken Language Setting",
                [
                    "Auto-Detect (Smart Hindi/English/Urdu Disambiguation)",
                    "Force Hindi (हिंदी · Devanagari Script)",
                    "Force English",
                    "Force Urdu (اردو)",
                ],
                index=0,
                help="Auto-Detect intelligently checks audio and title to prevent Whisper from mistakenly outputting Urdu/Arabic script for spoken Hindi.",
            )

        lang_code_map = {
            "Auto-Detect (Smart Hindi/English/Urdu Disambiguation)": "auto",
            "Force Hindi (हिंदी · Devanagari Script)": "hi",
            "Force English": "en",
            "Force Urdu (اردو)": "ur",
        }
        chosen_lang = lang_code_map[language_option]

        if st.button("Process video", type="primary"):
            video_dir = Path(getattr(rag, "VIDEO_DIR", Path(rag.PROJECT_DIR) / "videos"))
            video_dir.mkdir(parents=True, exist_ok=True)
            video_extension = Path(uploaded_file.name).suffix.lower()
            video_path = video_dir / f"{uuid.uuid4().hex}{video_extension}"

            with open(video_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            # -------------------------------------------------------------
            # REAL-TIME PROGRESS BAR & REMAINING TIME NOTIFICATION
            # -------------------------------------------------------------
            progress_container = st.container()
            with progress_container:
                st.markdown("#### Processing Progress")
                p_bar = st.progress(0)
                status_text = st.empty()
                time_metric_col1, time_metric_col2, time_metric_col3 = st.columns(3)
                elapsed_box = time_metric_col1.empty()
                eta_box = time_metric_col2.empty()
                stage_box = time_metric_col3.empty()

                start_wall_clock = time.time()

                def update_progress_ui(percent, message, step, total_steps, eta_seconds, phase):
                    p_bar.progress(min(100, max(0, int(percent))))
                    status_text.markdown(f"**{phase}**: {message}")
                    elapsed_sec = int(time.time() - start_wall_clock)
                    elapsed_box.metric("⏱ Elapsed Time", f"{elapsed_sec // 60:02d}:{elapsed_sec % 60:02d}")
                    rem_sec = int(eta_seconds)
                    eta_box.metric("⏳ Estimated Remaining", f"{rem_sec // 60:02d}:{rem_sec % 60:02d}")
                    stage_box.metric("📌 Step", f"{step} of {total_steps}")

                try:
                    result = rag.process_video(
                        video_path,
                        forced_language=chosen_lang,
                        progress_callback=update_progress_ui
                    )

                    st.session_state["active_video"] = result.get("video_name", uploaded_file.name)
                    st.session_state["video_path"] = str(video_path.resolve())
                    st.session_state["video_processing_result"] = result
                    save_active_video_record(video_path, result)
                    st.session_state["latest_result"] = None
                    st.session_state["latest_question"] = ""
                    st.session_state["summary_result"] = None

                    st.success("🎉 Video processed successfully! Transcript normalized to Devanagari Hindi / English.")
                    st.rerun()

                except Exception as error:
                    st.error(f"Video processing failed: {error}")

    processing = st.session_state.get("video_processing_result")
    if processing:
        st.divider()
        st.markdown("### Current active video")
        st.write(f"**Video:** {processing.get('video_name', 'N/A')}")
        
        detected = processing.get("detected_language", {})
        if detected:
            st.markdown(
                f"""
                <div style="background:#0c1a2e; border:1px solid #1e3a63; border-radius:10px; padding:10px 14px; margin-bottom:12px;">
                    <span style="color:#60a5fa; font-weight:700;">Spoken Language:</span>
                    <b style="color:#f8fafc; margin-left:6px;">{detected.get('display_name', 'Hindi')}</b>
                    <span style="margin-left:12px; color:#34d399; font-size:0.8rem;">(Script: {detected.get('script_enforced', 'Devanagari')})</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        a, b, c = st.columns(3)
        with a:
            metric_card("Transcript segments", processing.get("segment_count", "N/A"))
        with b:
            metric_card("Transcript passages", processing.get("chunk_count", "N/A"))
        with c:
            duration = safe_float(processing.get("duration_seconds"))
            metric_card("Video Duration", format_timestamp(duration) if duration else "N/A")


# ============================================================
# PAGE 3: UNDERSTAND VIDEO (WITH HINDI AUTO-DETECTION & TRANSLATION TOGGLES)
# ============================================================

elif page == "Understand Video":
    st.markdown("## Understand your video")
    st.caption("Review the transcript, translate passages, and generate a structured summary.")

    chunks = getattr(rag, "KNOWLEDGE_BASE", [])
    if not chunks:
        st.warning("No video transcript is loaded yet. Upload and process a video first.")
    else:
        if st.session_state.get("active_video"):
            st.info(f"Current active video: **{st.session_state['active_video']}**")

        st.markdown("### Video overview")
        a, b = st.columns(2)
        with a:
            metric_card("Transcript passages", len(chunks))
        with b:
            metric_card("Embedding model", "BGE-M3 (1024 dims)")

        # Summary Generation Section
        summary_function = getattr(rag, "generate_video_summary", None)
        st.markdown("### Video summary")
        if callable(summary_function):
            summary_col1, summary_col2 = st.columns([1, 3])
            with summary_col1:
                summary_language = st.selectbox(
                    "Summary language",
                    ["English", "Hindi", "Hinglish", "Urdu"],
                    key="summary_language",
                )
            with summary_col2:
                st.write("")
                st.write("")
                generate_summary_clicked = st.button("Generate video summary", type="primary")

            if generate_summary_clicked:
                try:
                    with st.spinner("Generating structured summary from transcript..."):
                        summary = summary_function(response_language=summary_language)
                    st.session_state["summary_result"] = summary
                except Exception as error:
                    st.error(f"Summary generation failed: {error}")

            if st.session_state.get("summary_result"):
                summary = st.session_state["summary_result"]
                if isinstance(summary, dict):
                    st.markdown("#### Overview")
                    st.write(summary.get("summary", ""))
                    takeaways = summary.get("key_takeaways", [])
                    if takeaways:
                        st.markdown("#### Key takeaways")
                        for item in takeaways:
                            st.markdown(f"- {item}")
                    topics = summary.get("topics", [])
                    if topics:
                        st.markdown("#### Topics covered")
                        for topic in topics:
                            title = topic.get("title", "Topic")
                            start = topic.get("start_time", 0)
                            end = topic.get("end_time", 0)
                            with st.expander(f"{title} · {format_timestamp(start)}–{format_timestamp(end)}"):
                                st.write(topic.get("description", ""))

        st.divider()

        # -------------------------------------------------------------
        # TRANSCRIPT TRANSLATION TOGGLE (ORIGINAL, ENGLISH, HINDI, URDU, HINGLISH)
        # -------------------------------------------------------------
        st.markdown("### Search & explore transcript")
        
        col_search, col_lang = st.columns([3, 2])
        with col_search:
            search_text = st.text_input(
                "Find a word or phrase in the video transcript",
                placeholder="For example: HTML, web, Tim Berners-Lee, structure, CSS",
                key="transcript_search",
            )
        with col_lang:
            transcript_target_lang = st.selectbox(
                "Transcript passage translation",
                ["Original", "English", "Hindi", "Hinglish", "Urdu"],
                index=0,
                key="transcript_translation_selector",
                help="Original shows the speech transcription in standard Devanagari Hindi. You can also translate into English, Hinglish, or Urdu.",
            )

        matches = []
        for index, chunk in enumerate(chunks):
            text_value = str(chunk.get("text", ""))
            if not search_text or search_text.lower() in text_value.lower():
                cid = chunk.get("chunk_id", index + 1)
                start = chunk.get("start_time", 0)
                end = chunk.get("end_time", 0)
                matches.append((cid, start, end, text_value))

        st.caption(
            f"Showing {min(len(matches), 60)} matching passage(s) out of {len(chunks)}. "
            f"Active view: **{transcript_target_lang}**."
        )

        for cid, start, end, text_value in matches[:60]:
            title = f"Passage {cid} · {format_timestamp(start)}–{format_timestamp(end)}"
            with st.expander(title):
                if transcript_target_lang != "Original":
                    with st.spinner(f"Translating passage {cid} to {transcript_target_lang}..."):
                        translated = rag.translate_text(text_value, transcript_target_lang)
                    st.markdown(f'<span class="translation-badge">{transcript_target_lang} translation</span>', unsafe_allow_html=True)
                    st.write(translated)
                    with st.expander("Show original transcript text", expanded=False):
                        st.caption("Original speech transcript:")
                        st.write(text_value)
                else:
                    st.write(text_value)

        if len(matches) > 60:
            st.caption("Only the first 60 matching passages are displayed.")


# ============================================================
# PAGE 4: ASK QUESTIONS (WITH MULTILINGUAL SUPPORT & EVIDENCE TRANSLATION)
# ============================================================

elif page == "Ask Questions":
    st.markdown("## Ask questions about your video")
    st.caption("Ask questions in your own words. Answers are generated and verified against video evidence.")

    if st.session_state.get("active_video"):
        st.success(f"Active video: **{st.session_state['active_video']}**")
    elif not getattr(rag, "KNOWLEDGE_BASE", []):
        st.warning("Upload and process a video before asking questions.")
    else:
        st.info("Using the transcript currently loaded in the knowledge base.")

    col_ans_lang, _ = st.columns([2, 3])
    with col_ans_lang:
        response_language = st.selectbox(
            "Answer language",
            ["English", "Hindi", "Hinglish", "Urdu"],
            index=0,
            key="response_language",
        )

    question = st.text_area(
        "What would you like to understand?",
        placeholder="For example: What is HTML and who invented it according to the video?",
        height=100,
        key="question_input",
    )

    if st.button("Generate answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not getattr(rag, "KNOWLEDGE_BASE", []):
            st.warning("There is no loaded transcript. Please process a video first.")
        else:
            with st.spinner("Retrieving video passages and checking answer claims..."):
                try:
                    result = rag.ask_sr_rag(
                        question.strip(),
                        response_language=response_language,
                    )
                    st.session_state["latest_result"] = result
                    st.session_state["latest_question"] = question.strip()
                    st.session_state["qa_history"].append({
                        "question": question.strip(),
                        "answer": result.get("answer", ""),
                        "language": response_language,
                        "video": st.session_state.get("active_video", ""),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    })
                except Exception as error:
                    st.error(f"Unable to process question: {error}")

    result = st.session_state.get("latest_result")
    if result:
        st.divider()
        st.markdown("### Answer")
        answer = result.get("answer", "No answer was returned.")
        st.markdown(
            f"""
            <div style="background: #11223e; border-left: 4px solid #3b82f6; border-radius: 10px; padding: 18px 20px; color: #f1f5f9; line-height: 1.6; margin-bottom: 16px;">
                {html.escape(answer).replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )

        verification = result.get("verification", {}) or {}
        outcome = str(result.get("answer_outcome", "UNKNOWN")).upper()
        if outcome == "ABSTAINED":
            st.warning("The video transcript did not provide enough support for a reliable answer.")
        elif verification.get("overall_status") == "PARTIALLY_SUPPORTED":
            st.warning("Some parts of this answer have only partial transcript support.")
        elif verification.get("overall_status") == "UNSUPPORTED":
            st.warning("The verification stage could not support the answer from retrieved passages.")

        left, right, third = st.columns(3)
        with left:
            st.markdown("**Answer verification**")
            show_status(verification.get("overall_status", "N/A"))
        with right:
            st.markdown("**Final decision**")
            show_status(outcome)
        with third:
            st.markdown("**Adaptive expansion**")
            st.write("Expanded" if result.get("retry_performed") else "Standard")

        # Save to Notes button directly below answer
        btn_col, _ = st.columns([1, 3])
        with btn_col:
            if st.button("＋ Save answer to My Notes", key="save_current_answer_btn"):
                save_note(
                    st.session_state.get("latest_question", result.get("question", "")),
                    answer,
                    st.session_state.get("active_video", "Video"),
                )
                st.success("Answer saved to My Notes!")

        # Evidence Translation Section
        with st.expander("View video passages supporting this answer", expanded=True):
            col_ev_hdr, col_ev_trans = st.columns([3, 2])
            with col_ev_hdr:
                st.caption("Retrieved transcript passages used as ground truth for this answer.")
            with col_ev_trans:
                evidence_lang = st.selectbox(
                    "Evidence translation",
                    ["Original", "English", "Hindi", "Hinglish", "Urdu"],
                    index=0,
                    key="evidence_translation_selector",
                    help="Read supporting passages in your preferred language while preserving original verification evidence."
                )

            evidence = result.get("evidence", [])
            for item in evidence:
                start = item.get("start_time", 0)
                end = item.get("end_time", 0)
                cid = item.get("chunk_id", "N/A")
                sim = item.get("similarity", 0.0)
                orig_text = item.get("text", "")

                with st.expander(f"Chunk {cid} · {format_timestamp(start)}–{format_timestamp(end)} · Similarity {sim:.3f}"):
                    if evidence_lang != "Original":
                        with st.spinner(f"Translating passage to {evidence_lang}..."):
                            disp = rag.translate_text(orig_text, evidence_lang)
                        st.markdown(f'<span class="translation-badge">{evidence_lang} translation</span>', unsafe_allow_html=True)
                        st.write(disp)
                        with st.expander("Show original transcript passage", expanded=False):
                            st.write(orig_text)
                    else:
                        st.write(orig_text)

        with st.expander("View answer verification details"):
            claims = verification.get("claims", []) or []
            for index, claim in enumerate(claims, start=1):
                status_title = str(claim.get("status", "N/A")).replace("_", " ").title()
                with st.expander(f"Claim {index} · {status_title}"):
                    st.write(claim.get("claim", ""))
                    st.write("**Reason:**", claim.get("reason", "Not provided"))
                    st.write("**Evidence passage IDs:**", claim.get("evidence_chunks", []))


# ============================================================
# PAGE 5: MY NOTES (WITH DIRECT + ADD NOTE BUTTON & FORM)
# ============================================================

elif page == "My Notes":
    st.markdown("## My Notes")
    st.caption("Keep useful answers and personal observations from your video learning sessions.")

    # Direct Add Note Form
    with st.expander("＋ Add New Note", expanded=False):
        note_title = st.text_input("Note Title / Question", placeholder="e.g. Core concepts of HTML tags or key observation", key="new_note_title")
        note_content = st.text_area("Note Content", placeholder="Write your notes, explanations, or code snippets here...", height=120, key="new_note_body")
        note_video = st.text_input("Associated Video", value=st.session_state.get("active_video", "Educational Video"), key="new_note_vid")

        if st.button("Save note", type="primary", key="save_manual_note_btn"):
            if not note_title.strip() or not note_content.strip():
                st.warning("Please provide both a title and content for your note.")
            else:
                save_note(note_title.strip(), note_content.strip(), note_video.strip())
                st.success("Note saved successfully!")
                st.rerun()

    notes = st.session_state.get("saved_notes", [])
    if not notes:
        st.info("You have not saved any notes yet. Click '＋ Add New Note' above or use '+ Save answer to My Notes' after asking a question.")
    else:
        st.write(f"**{len(notes)} saved note(s)**")
        for index, note in enumerate(notes):
            with st.expander(note.get("question", f"Note {index + 1}"), expanded=(index == len(notes) - 1)):
                st.caption(f"{note.get('video', 'Video not specified')} · {note.get('saved_at', '')}")
                st.write(note.get("answer", ""))
                if st.button("Delete note", key=f"del_note_{note.get('id', index)}"):
                    st.session_state["saved_notes"] = [n for n in st.session_state["saved_notes"] if n.get("id") != note.get("id")]
                    st.rerun()

        export_lines = ["# SR-RAG Learning Notes", "", f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
        for index, note in enumerate(notes, start=1):
            export_lines.extend([
                f"## {index}. {note.get('question', '')}",
                f"*Video: {note.get('video', 'Not specified')} · Saved: {note.get('saved_at', '')}*",
                "",
                note.get("answer", ""),
                "",
                "---",
                "",
            ])
        export_text = "\n".join(export_lines)
        st.download_button(
            "Download notes as Markdown",
            data=export_text,
            file_name="sr_rag_learning_notes.md",
            mime="text/markdown",
            type="primary",
        )


# ============================================================
# PAGE 6: ADVANCED DETAILS (RICH FORMATTED DIAGNOSTICS & TRACE UI)
# ============================================================

elif page == "Advanced Details":
    st.markdown("## Advanced details")
    st.caption("Research diagnostics, similarity scores, claim verification traces, and system status.")

    tab_kb, tab_evidence, tab_history, tab_system = st.tabs([
        "Transcript Explorer",
        "Evidence & Verification",
        "Decision History",
        "System Diagnostics",
    ])

    # Tab 1: Transcript Explorer
    with tab_kb:
        chunks = getattr(rag, "KNOWLEDGE_BASE", [])
        embeddings = getattr(rag, "KNOWLEDGE_EMBEDDINGS", [])
        shape = getattr(embeddings, "shape", [0, 0])
        a, b, c = st.columns(3)
        with a:
            metric_card("Loaded passages", len(chunks))
        with b:
            metric_card("Embedding dimensions", shape[1] if len(shape) > 1 else "N/A")
        with c:
            metric_card("Segments per chunk", getattr(rag, "SEGMENTS_PER_CHUNK", "N/A"))

        st.write("**Embedding model:** BAAI/bge-m3")
        st.write("**Retrieval method:** Cosine similarity")
        st.write("**Initial top-k:**", getattr(rag, "INITIAL_TOP_K", "N/A"))
        st.write("**Expanded top-k:**", getattr(rag, "EXPANDED_TOP_K", "N/A"))
        st.write("**Evidence threshold:**", getattr(rag, "EVIDENCE_THRESHOLD", "N/A"))

        search_text = st.text_input("Search transcript passages", key="advanced_kb_search")
        matches = []
        for index, chunk in enumerate(chunks):
            text_val = str(chunk.get("text", chunk.get("content", ""))) if isinstance(chunk, dict) else str(chunk)
            if not search_text or search_text.lower() in text_val.lower():
                cid = chunk.get("chunk_id", index + 1) if isinstance(chunk, dict) else index + 1
                start = chunk.get("start_time", "N/A") if isinstance(chunk, dict) else "N/A"
                end = chunk.get("end_time", "N/A") if isinstance(chunk, dict) else "N/A"
                matches.append((cid, start, end, text_val))

        st.caption(f"{len(matches)} matching passage(s).")
        for cid, start, end, text_val in matches[:60]:
            with st.expander(f"Chunk {cid} · {format_timestamp(start)}–{format_timestamp(end)}"):
                st.write(text_val)

    # Tab 2: Evidence & Verification
    with tab_evidence:
        result = st.session_state.get("latest_result")
        if not result:
            st.info("Ask a question first to inspect its evidence and verification trace.")
        else:
            st.markdown("### Latest question")
            st.info(st.session_state.get("latest_question", result.get("question", "N/A")))
            verification = result.get("verification", {}) or {}
            hallucination = result.get("hallucination_detection", {}) or {}
            reflection = result.get("reflection", {}) or {}

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Verification Status**")
                show_status(verification.get("overall_status", "N/A"))
            with c2:
                st.markdown("**Answer Outcome**")
                show_status(result.get("answer_outcome", "N/A"))
            with c3:
                st.markdown("**Hallucination Assessment**")
                show_status(hallucination.get("assessment", "N/A"))

            st.markdown("### Retrieved transcript passages with similarity scores")
            evidence = result.get("evidence", [])
            for item in evidence:
                start = item.get("start_time", 0)
                end = item.get("end_time", 0)
                cid = item.get("chunk_id", "N/A")
                sim = safe_float(item.get("similarity"))
                text_content = item.get("text", "")
                title = f"Chunk {cid} · {format_timestamp(start)}–{format_timestamp(end)}"
                if sim is not None:
                    title += f" · Similarity {sim:.3f}"
                with st.expander(title):
                    st.write(text_content)

            st.markdown("### Claim verification breakdown")
            claims = verification.get("claims", []) or []
            if claims:
                counts = {"SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0}
                for c in claims:
                    s = str(c.get("status", "")).upper()
                    if s in counts:
                        counts[s] += 1
                ca, cb, cc = st.columns(3)
                with ca:
                    metric_card("Supported", counts["SUPPORTED"])
                with cb:
                    metric_card("Partially supported", counts["PARTIALLY_SUPPORTED"])
                with cc:
                    metric_card("Unsupported", counts["UNSUPPORTED"])

                for index, claim in enumerate(claims, start=1):
                    with st.expander(f"Claim {index} · {str(claim.get('status', 'N/A')).replace('_', ' ').title()}"):
                        st.write(claim.get("claim", ""))
                        st.write("**Reason:**", claim.get("reason", "N/A"))
                        st.write("**Evidence passage IDs:**", claim.get("evidence_chunks", []))
            else:
                st.caption("No individual claim objects returned.")

            st.markdown("### Self-reflection diagnostics")
            st.write(reflection.get("reason", "No reflection reason recorded."))

            with st.expander("Technical JSON trace"):
                st.json(result, expanded=False)

    # Tab 3: Decision History
    with tab_history:
        records, history_path = get_history()
        if not records:
            st.info("No saved decision history is available yet.")
        else:
            outcomes = [str(r.get("answer_outcome", "UNKNOWN")).upper() for r in records]
            a, b, c, d = st.columns(4)
            with a:
                metric_card("Total queries", len(records))
            with b:
                metric_card("Answered", outcomes.count("ANSWERED"))
            with c:
                metric_card("Abstained", outcomes.count("ABSTAINED"))
            with d:
                metric_card("Needs review", outcomes.count("NEEDS_REVIEW"))

            st.caption(f"Trace file: `{history_path}`")
            col_hf, col_hs = st.columns([1, 2])
            with col_hf:
                filter_outcome = st.selectbox(
                    "Filter decisions",
                    ["All", "ANSWERED", "ABSTAINED", "NEEDS_REVIEW"],
                    key="history_filter_sel",
                )
            with col_hs:
                search_query = st.text_input("Search saved questions", key="history_query_search")

            shown = 0
            for index, record in enumerate(reversed(records), start=1):
                rec_outcome = str(record.get("answer_outcome", "UNKNOWN")).upper()
                q_text = str(record.get("question", "Unknown query"))
                if filter_outcome != "All" and rec_outcome != filter_outcome:
                    continue
                if search_query and search_query.lower() not in q_text.lower():
                    continue
                shown += 1
                with st.expander(f"{index}. {q_text[:90]} · {rec_outcome}"):
                    st.markdown("**Answer**")
                    st.write(record.get("answer", ""))
                    show_status(rec_outcome)
                    st.caption(f"Saved at: {record.get('timestamp', 'N/A')}")
                    with st.expander("Show detailed verification record"):
                        st.json(record, expanded=False)
            if shown == 0:
                st.info("No records match the selected filter.")

    # Tab 4: System Diagnostics
    with tab_system:
        st.markdown("### Runtime configuration")
        embeddings = getattr(rag, "KNOWLEDGE_EMBEDDINGS", [])
        shape = getattr(embeddings, "shape", [0, 0])
        st.write("**Embedding model:** BAAI/bge-m3")
        st.write("**Embedding dimensions:**", shape[1] if len(shape) > 1 else "N/A")
        st.write("**LLM:**", getattr(rag, "LLM_MODEL", "N/A"))
        st.write("**Language Normalizer:** Devanagari Hindi Script Enforcement Active")
        st.write("**Whisper Model:** base (with bilingual script prompting)")
        st.write("**Fast Duration Cap:** 10 minutes (600 seconds)")

        st.markdown("### Project artifacts")
        proj_dir = Path(getattr(rag, "PROJECT_DIR", Path.cwd()))
        _, history_path = get_history()
        for label, path in [
            ("Knowledge base", proj_dir / "knowledge_base.pkl"),
            ("Embeddings", proj_dir / "knowledge_embeddings.npy"),
            ("Videos directory", proj_dir / "videos"),
            ("Active video record", ACTIVE_VIDEO_FILE),
            ("Audio directory", proj_dir / "audio"),
            ("Transcripts directory", proj_dir / "transcripts"),
            ("Decision trace", history_path),
        ]:
            st.write(f"**{label}:** {artifact_status(path)}")

        st.info(
            "The weighted retrieval score is an internal relevance heuristic. "
            "It should not be interpreted as a calibrated probability of factual correctness."
        )

