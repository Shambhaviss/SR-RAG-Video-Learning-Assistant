"""
SR-RAG Backend: Self-Reflective Retrieval-Augmented Generation
with Hallucination Detection, Adaptive Retrieval, and Multilingual Support.

Key features and fixes:
1. Audio extraction (mono 16kHz WAV via FFmpeg) with fast duration probing.
2. Robust Language Auto-Detection & Disambiguation (Hindi vs Urdu vs English vs Hinglish)
   that prevents Whisper from mistakenly outputting Urdu/Arabic script for Hindi speech.
3. Script enforcement (Devanagari + Latin for coding terms) and fail-safe phonetic
   Devanagari normalization for any residual Arabic-script segments.
4. Rich progress callback reporting current step, percentage, elapsed time, and ETA.
5. Duration analysis showing Fast Processing Range (<= 10 mins) vs Extended (> 10 mins).
6. Multilingual translation pipeline (Original, English, Hindi, Urdu, Hinglish).
7. BGE-M3 sentence-transformers embeddings & cosine similarity retrieval.
8. Adaptive retrieval (threshold-triggered expansion with neighbor context windows).
9. Evidence-grounded claim verification, self-reflection, and hallucination detection.
10. Decision trace recording and structured video summary generation.
"""

import os
import re
import json
import time
import shutil
import hashlib
import subprocess
import uuid
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# =========================================================
# PROJECT CONFIGURATION & DIRECTORIES
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

AUDIO_DIR = PROJECT_DIR / "audio"
TRANSCRIPT_DIR = PROJECT_DIR / "transcripts"
OUTPUT_DIR = PROJECT_DIR / "outputs"
VIDEO_DIR = PROJECT_DIR / "videos"

for directory in (AUDIO_DIR, TRANSCRIPT_DIR, OUTPUT_DIR, VIDEO_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEGMENTS_PER_CHUNK = 5
INITIAL_TOP_K = 3
EXPANDED_TOP_K = 6
EVIDENCE_THRESHOLD = 0.35

# Optimal video duration recommendations
FAST_PROCESSING_MAX_SECONDS = 600  # 10 minutes
MODERATE_PROCESSING_MAX_SECONDS = 1800  # 30 minutes

LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024


# =========================================================
# ENVIRONMENT & GROQ CLIENT
# =========================================================

load_dotenv(PROJECT_DIR / ".env")
api_key = os.getenv("GROQ_API_KEY")

client = None
if api_key:
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        print(f"Warning: Failed to initialize Groq client: {e}")
else:
    print("Notice: GROQ_API_KEY not found in environment. Set it in .env file.")


def get_groq_client():
    global client
    if client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
        client = Groq(api_key=key)
    return client


# =========================================================
# EMBEDDING MODEL INITIALIZATION (LAZY / SAFE LOAD)
# =========================================================

_embedding_model_instance = None


def get_embedding_model():
    global _embedding_model_instance
    if _embedding_model_instance is None:
        print("Loading BGE-M3 embedding model...")
        _embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model_instance


class EmbeddingModelProxy:
    def encode(self, *args, **kwargs):
        return get_embedding_model().encode(*args, **kwargs)


embedding_model = EmbeddingModelProxy()


# =========================================================
# KNOWLEDGE BASE & EMBEDDINGS MANAGEMENT
# =========================================================

knowledge_base_path = PROJECT_DIR / "knowledge_base.pkl"
embeddings_path = PROJECT_DIR / "knowledge_embeddings.npy"

KNOWLEDGE_BASE = []
KNOWLEDGE_EMBEDDINGS = None

if knowledge_base_path.exists() and embeddings_path.exists():
    try:
        with open(knowledge_base_path, "rb") as file:
            KNOWLEDGE_BASE = pickle.load(file)
        KNOWLEDGE_EMBEDDINGS = np.load(embeddings_path)
        print(f"Knowledge base loaded: {len(KNOWLEDGE_BASE)} chunks, shape: {KNOWLEDGE_EMBEDDINGS.shape}")
    except Exception as err:
        print(f"Warning: Could not load saved knowledge base: {err}")
        KNOWLEDGE_BASE = []
        KNOWLEDGE_EMBEDDINGS = None


# =========================================================
# TRANSLATION CACHE & SERVICE
# =========================================================

TRANSLATION_CACHE_FILE = OUTPUT_DIR / "translation_cache.json"
_TRANSLATION_CACHE = {}


def _load_translation_cache():
    global _TRANSLATION_CACHE
    if TRANSLATION_CACHE_FILE.exists():
        try:
            with open(TRANSLATION_CACHE_FILE, "r", encoding="utf-8") as f:
                _TRANSLATION_CACHE = json.load(f)
        except Exception:
            _TRANSLATION_CACHE = {}


def _save_translation_cache():
    try:
        with open(TRANSLATION_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_TRANSLATION_CACHE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Could not save translation cache: {e}")


_load_translation_cache()


def translate_text(text: str, target_language: str) -> str:
    """
    Translates or transliterates text into the target language using Groq LLM.
    
    Supported target languages:
    - 'Original': returns input text unchanged.
    - 'English': natural English translation.
    - 'Hindi': Devanagari script.
    - 'Urdu': Urdu Nastaliq/Arabic script.
    - 'Hinglish': Romanized conversational Hindi (e.g., 'Yeh to hum sabhi log jaante hain').
    """
    if not text or not str(text).strip():
        return ""

    text_clean = str(text).strip()
    target = target_language.strip()

    if target.lower() in {"original", "same", ""}:
        return text_clean

    cache_key = hashlib.md5(f"{text_clean}___{target}".encode("utf-8")).hexdigest()
    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    groq_client = get_groq_client()

    target_rules = {
        "Hindi": "Translate/normalize into natural Hindi written strictly in Devanagari script (देवनागरी). Technical terms and programming keywords should remain in English/Latin or accurate Devanagari transliteration.",
        "English": "Translate into natural, clear, idiomatic English, preserving all technical and programming concepts.",
        "Urdu": "Translate into natural Urdu written in standard Urdu/Nastaliq script (اردو).",
        "Hinglish": "Transliterate into natural conversational Hinglish (Hindi written in Roman/English alphabet, e.g., 'Yeh to hum sabhi log jaante hain ki HTML web ki language hai'). Keep code keywords in English.",
    }.get(target, f"Translate into {target}.")

    prompt = f"""
You are an expert multilingual educational video transcript translator.

TASK:
Convert/translate the following transcript segment into {target}.

GUIDELINE:
{target_rules}

RULES:
1. Preserve all technical terms, code keywords (JavaScript, HTML, CSS, addEventListener, onclick, function, etc.), variable names, numbers, and timestamps.
2. Keep the meaning and instructional tone intact.
3. Do NOT add meta commentary, explanations, or quotes.
4. Output ONLY the translated/converted text.

TRANSCRIPT:
{text_clean}
"""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional educational subtitle translator specializing in technical videos. Translate accurately into {target}."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )

        translated = response.choices[0].message.content.strip()
        if translated.startswith('"') and translated.endswith('"'):
            translated = translated[1:-1].strip()

        _TRANSLATION_CACHE[cache_key] = translated
        _save_translation_cache()
        return translated

    except Exception as error:
        print(f"Translation error: {error}")
        return text_clean


# =========================================================
# VIDEO DURATION & PERFORMANCE PROFILE INSPECTOR
# =========================================================

def get_video_duration(video_path: Path) -> float:
    """Extracts duration in seconds using ffprobe or ffmpeg."""
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        try:
            cmd = [
                ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            cmd = [ffmpeg_path, "-i", str(video_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", result.stderr)
            if match:
                h, m, s = match.groups()
                return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            pass

    return 0.0


def check_processing_profile(duration_seconds: float) -> dict:
    """
    Evaluates whether the video duration is within the optimal fast processing range.
    Optimal fast processing duration: <= 10 minutes.
    """
    if duration_seconds <= 0:
        return {
            "tier": "UNKNOWN",
            "is_fast": True,
            "badge_color": "blue",
            "headline": "Duration Unknown",
            "description": "Duration could not be probed in advance.",
            "estimated_wait_time": "30-60s",
        }

    minutes = duration_seconds / 60.0

    if duration_seconds <= FAST_PROCESSING_MAX_SECONDS:
        return {
            "tier": "FAST",
            "is_fast": True,
            "badge_color": "emerald",
            "headline": f"⚡ Fast Processing Range ({minutes:.1f} min)",
            "description": "Recommended length: under 10 minutes. Transcribes and embeds rapidly in ~30–60 seconds.",
            "estimated_wait_time": "30–60 seconds",
            "duration_formatted": f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}",
        }
    elif duration_seconds <= MODERATE_PROCESSING_MAX_SECONDS:
        return {
            "tier": "MODERATE",
            "is_fast": False,
            "badge_color": "amber",
            "headline": f"⏱ Moderate Duration ({minutes:.1f} min)",
            "description": "Between 10 and 30 minutes. Processing will take 1–3 minutes depending on speech density.",
            "estimated_wait_time": "1–3 minutes",
            "duration_formatted": f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}",
        }
    else:
        return {
            "tier": "EXTENDED",
            "is_fast": False,
            "badge_color": "rose",
            "headline": f"⏳ Extended Processing Range ({minutes:.1f} min)",
            "description": "Over 30 minutes. Speech transcription and BGE-M3 multi-chunk embedding will take 3+ minutes.",
            "estimated_wait_time": "3–6 minutes",
            "duration_formatted": f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}",
        }


# =========================================================
# SCRIPT METRICS & LANGUAGE DISAMBIGUATION (HINDI VS URDU)
# =========================================================

def _script_counts(text: str):
    """Counts Arabic/Urdu, Devanagari and Latin letters."""
    arabic = sum(
        1 for char in text
        if ("\u0600" <= char <= "\u06FF" or "\u0750" <= char <= "\u077F" or "\u08A0" <= char <= "\u08FF")
    )
    devanagari = sum(1 for char in text if "\u0900" <= char <= "\u097F")
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    return arabic, devanagari, latin


def detect_video_language(audio_path: Path, video_title: str = "", whisper_model=None) -> dict:
    """
    Intelligently detects spoken language and resolves the Hindi vs Urdu confusion.
    """
    title_lower = video_title.lower()
    hindi_keywords = ["hindi", "हिन्दी", "हिंदी", "hinglish", "sikhe", "jaante", "kya", "hai", "kaise"]
    has_hindi_title_clue = any(k in title_lower for k in hindi_keywords)

    detected_code = "hi" if has_hindi_title_clue else "unknown"
    confidence = 0.95 if has_hindi_title_clue else 0.5
    method = "filename_clue" if has_hindi_title_clue else "acoustic_probe"

    if whisper_model is not None and not has_hindi_title_clue:
        try:
            import whisper
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
            _, probs = whisper_model.detect_language(mel)
            top_lang = max(probs, key=probs.get)
            top_prob = probs[top_lang]

            if top_lang in {"ur", "hi"}:
                detected_code = "hi"
                confidence = float(probs.get("hi", 0.0) + probs.get("ur", 0.0))
                method = "acoustic_with_hindi_priority"
            elif top_lang == "en":
                detected_code = "en"
                confidence = float(top_prob)
                method = "acoustic_whisper"
            else:
                detected_code = top_lang
                confidence = float(top_prob)
                method = "acoustic_whisper"
        except Exception as e:
            print(f"Acoustic language detection warning: {e}")
            detected_code = "hi"

    if detected_code in {"unknown", "ur"} and has_hindi_title_clue:
        detected_code = "hi"

    lang_display_names = {
        "hi": "Hindi (हिंदी · Devanagari Script)",
        "en": "English",
        "ur": "Urdu (اردو)",
    }

    return {
        "language_code": detected_code,
        "display_name": lang_display_names.get(detected_code, detected_code.title()),
        "confidence": confidence,
        "method": method,
        "script_enforced": "Devanagari (देवनागरी)" if detected_code == "hi" else "Standard",
    }


def _convert_arabic_script_hindi_to_devanagari(text: str) -> str:
    """
    Fail-safe converter: transliterates Arabic/Urdu script output into standard Devanagari Hindi.
    """
    try:
        groq_client = get_groq_client()
        prompt = f"""
The following text is spoken Hindi / Hinglish that was mistakenly transcribed in Arabic/Urdu script by speech recognition.

TASK:
Convert this text into standard Hindi in Devanagari script (देवनागरी).

STRICT RULES:
1. Preserve every English or technical word (e.g. JavaScript, HTML, CSS, DOM, variable, loop, onclick, function, code) in its original Latin script or accurate Devanagari transliteration.
2. Transliterate Hindi spoken words into standard Devanagari (e.g. "یہ تو ہم سبھی لوگ جانتے ہیں" -> "यह तो हम सभी लोग जानते हैं").
3. Do NOT add commentary, explanations, or quotes.
4. Output ONLY the converted text.

INPUT TEXT:
{text}
"""
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024
        )
        converted = response.choices[0].message.content.strip()
        arabic_cnt, dev_cnt, _ = _script_counts(converted)
        if dev_cnt > arabic_cnt:
            return converted
    except Exception as e:
        print(f"Phonetic transliteration warning: {e}")
    return text


# =========================================================
# AUDIO EXTRACTION & TRANSCRIPTION WITH PROGRESS & ETA
# =========================================================

def extract_audio(video_path: Path) -> Path:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError("FFmpeg is not installed or not available in PATH.")

    audio_path = AUDIO_DIR / f"{video_path.stem}_{uuid.uuid4().hex[:8]}.wav"
    command = [
        ffmpeg_path,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_path)
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed:\n{result.stderr[-2000:]}")

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError("Extracted audio file is empty.")

    return audio_path


WHISPER_MODEL_INSTANCE = None


def transcribe_audio(
    audio_path: Path,
    video_title: str = "",
    forced_language: str = "auto",
    progress_callback=None
):
    global WHISPER_MODEL_INSTANCE
    start_time_all = time.time()

    def report_progress(percent, msg, step=3, total_steps=6, phase="Transcribing"):
        if progress_callback:
            elapsed = time.time() - start_time_all
            if percent > 0:
                total_estimated = elapsed / (percent / 100.0)
                eta = max(0, total_estimated - elapsed)
            else:
                eta = 45.0
            progress_callback(
                percent=percent,
                message=msg,
                step=step,
                total_steps=total_steps,
                eta_seconds=eta,
                phase=phase
            )

    if WHISPER_MODEL_INSTANCE is None:
        report_progress(15, "Loading Whisper speech model...", step=2, phase="Model Init")
        import whisper
        WHISPER_MODEL_INSTANCE = whisper.load_model("base")

    report_progress(25, "Detecting spoken language and script profile...", step=2, phase="Language Detection")
    lang_info = detect_video_language(audio_path, video_title=video_title, whisper_model=WHISPER_MODEL_INSTANCE)

    target_lang = lang_info["language_code"]
    if forced_language and forced_language != "auto":
        target_lang = forced_language

    report_progress(35, f"Spoken language identified: {lang_info['display_name']}. Enforcing script...", step=3, phase="Transcription")

    initial_prompt = (
        "नमस्ते, this educational video explains programming concepts in Hindi and English. "
        "Keywords: code, JavaScript, functions, variables, HTML, CSS, DOM, syntax, web. "
        "यह वीडियो हिंदी और English में है। देवनागरी लिपि का प्रयोग करें।"
    )

    transcribe_kwargs = {
        "fp16": False,
        "verbose": False,
        "task": "transcribe",
    }

    if target_lang == "hi":
        transcribe_kwargs["language"] = "hi"
        transcribe_kwargs["initial_prompt"] = initial_prompt
    elif target_lang in {"en", "ur"}:
        transcribe_kwargs["language"] = target_lang

    report_progress(45, "Running speech-to-text recognition...", step=3, phase="Transcription")
    transcription = WHISPER_MODEL_INSTANCE.transcribe(str(audio_path), **transcribe_kwargs)

    raw_segments = transcription.get("segments", [])
    total_segs = len(raw_segments)
    segments = []
    corrected_count = 0

    report_progress(65, f"Normalizing transcript ({total_segs} segments)...", step=4, phase="Normalization")

    for idx, seg in enumerate(raw_segments):
        raw_text = str(seg.get("text", "")).strip()
        if not raw_text:
            continue

        arabic_cnt, devanagari_cnt, _ = _script_counts(raw_text)
        final_text = raw_text
        correction_status = "original"

        if target_lang == "hi" and arabic_cnt >= 3 and devanagari_cnt == 0:
            converted = _convert_arabic_script_hindi_to_devanagari(raw_text)
            c_arabic, c_dev, _ = _script_counts(converted)
            if c_dev > c_arabic:
                final_text = converted
                correction_status = "devanagari_normalized"
                corrected_count += 1

        segments.append({
            "segment_id": int(seg.get("id", idx)),
            "start_time": float(seg.get("start", 0.0)),
            "end_time": float(seg.get("end", 0.0)),
            "text": final_text,
            "raw_text": raw_text,
            "detected_language": target_lang,
            "transcription_correction": correction_status
        })

    if not segments:
        raise ValueError("No speech could be detected in this video file.")

    report_progress(75, f"Speech transcribed: {len(segments)} segments ({corrected_count} script-normalized).", step=4, phase="Chunking")
    return segments, lang_info


def create_transcript_chunks(segments):
    chunks = []
    for index in range(0, len(segments), SEGMENTS_PER_CHUNK):
        group = segments[index:index + SEGMENTS_PER_CHUNK]
        chunk_text = " ".join(s["text"] for s in group).strip()
        if not chunk_text:
            continue
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "start_time": float(group[0]["start_time"]),
            "end_time": float(group[-1]["end_time"]),
            "text": chunk_text
        })
    return chunks


def build_video_knowledge_base(chunks, progress_callback=None):
    global KNOWLEDGE_BASE
    global KNOWLEDGE_EMBEDDINGS

    if not chunks:
        raise ValueError("No transcript chunks to index.")

    valid_chunks = [c for c in chunks if str(c.get("text", "")).strip()]
    if not valid_chunks:
        raise ValueError("No valid transcript text was found.")

    model = get_embedding_model()
    embedding_list = []
    total = len(valid_chunks)

    for index, chunk in enumerate(valid_chunks):
        chunk_text = chunk["text"]
        emb = model.encode(
            [chunk_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embedding_list.append(emb[0])

        if progress_callback:
            percent = 80 + int((index + 1) / total * 15)
            progress_callback(
                percent=percent,
                message=f"Generating BGE-M3 embeddings: {index + 1}/{total} passages...",
                step=5,
                total_steps=6,
                eta_seconds=max(1.0, (total - index) * 0.5),
                phase="Vector Indexing"
            )

    embeddings = np.asarray(embedding_list, dtype=np.float32)

    kb_temp = PROJECT_DIR / "knowledge_base.tmp"
    emb_temp = PROJECT_DIR / "knowledge_embeddings.tmp.npy"

    with open(kb_temp, "wb") as f:
        pickle.dump(valid_chunks, f)
    np.save(emb_temp, embeddings)

    os.replace(kb_temp, knowledge_base_path)
    os.replace(emb_temp, embeddings_path)

    KNOWLEDGE_BASE = valid_chunks
    KNOWLEDGE_EMBEDDINGS = embeddings

    return len(valid_chunks), embeddings.shape


def process_video(video_path, forced_language: str = "auto", progress_callback=None):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError("Video does not exist.")

    start_time = time.time()

    def update_progress(percent, message, step=1, total_steps=6, eta_seconds=40.0, phase="Processing"):
        if progress_callback:
            progress_callback(
                percent=percent,
                message=message,
                step=step,
                total_steps=total_steps,
                eta_seconds=eta_seconds,
                phase=phase
            )

    duration_sec = get_video_duration(video_path)
    profile = check_processing_profile(duration_sec)
    update_progress(5, f"Analyzing video ({profile['headline']})...", step=1, phase="Initialization")

    update_progress(12, "Extracting audio from video (mono 16kHz WAV)...", step=1, phase="Audio Extraction")
    audio_path = extract_audio(video_path)

    segments, lang_info = transcribe_audio(
        audio_path,
        video_title=video_path.name,
        forced_language=forced_language,
        progress_callback=update_progress
    )

    update_progress(78, "Dividing transcript into searchable semantic passages...", step=4, phase="Chunking")
    chunks = create_transcript_chunks(segments)
    for c in chunks:
        c["source_video"] = video_path.name

    chunk_count, embedding_shape = build_video_knowledge_base(chunks, progress_callback=update_progress)

    transcript_path = TRANSCRIPT_DIR / f"{video_path.stem}_transcript.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=4, ensure_ascii=False)

    total_duration = time.time() - start_time
    update_progress(100, f"Complete! Indexed {chunk_count} passages in {total_duration:.1f}s.", step=6, phase="Finished")

    return {
        "video_name": video_path.name,
        "audio_path": str(audio_path),
        "transcript_path": str(transcript_path),
        "segment_count": len(segments),
        "chunk_count": chunk_count,
        "embedding_shape": embedding_shape,
        "duration_seconds": segments[-1]["end_time"] if segments else duration_sec,
        "detected_language": lang_info,
        "processing_profile": profile,
        "processing_time_seconds": total_duration,
    }


# =========================================================
# RETRIEVAL ENGINE & ADAPTIVE EXPANSION
# =========================================================

def retrieve_context(question: str, knowledge_base=None, embeddings=None, top_k=INITIAL_TOP_K):
    if knowledge_base is None:
        knowledge_base = KNOWLEDGE_BASE
    if embeddings is None:
        embeddings = KNOWLEDGE_EMBEDDINGS

    if not knowledge_base or embeddings is None:
        raise ValueError("Knowledge base is empty. Please process a video first.")

    model = get_embedding_model()
    query_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    scores = np.dot(embeddings, query_embedding[0])
    top_indices = np.argsort(scores)[::-1][:min(top_k, len(scores))]

    retrieved = []
    for index in top_indices:
        item = knowledge_base[int(index)].copy()
        item["similarity"] = float(scores[index])
        retrieved.append(item)

    return retrieved


def calculate_evidence_score(retrieved):
    if not retrieved:
        return 0.0

    raw_scores = np.array([item.get("similarity", 0.0) for item in retrieved], dtype=float)
    normalized = np.clip((raw_scores + 1.0) / 2.0, 0.0, 1.0)
    avg_sim = float(np.mean(normalized))
    max_sim = float(np.max(normalized))
    std_sim = float(np.std(normalized))

    confidence = 0.5 * avg_sim + 0.3 * max_sim + 0.2 * (1.0 - std_sim)
    return float(np.clip(confidence, 0.0, 1.0))


def format_evidence(retrieved):
    evidence_text = ""
    for item in retrieved:
        start = item.get("start_time", 0)
        end = item.get("end_time", 0)
        cid = item.get("chunk_id", "Unknown")
        sim = item.get("similarity", 0.0)
        text = item.get("text", "")
        evidence_text += (
            f"\n[Chunk {cid}]\n"
            f"Timestamp: {float(start):.2f}s to {float(end):.2f}s\n"
            f"Similarity: {sim:.4f}\n"
            f"Content: {text}\n"
        )
    return evidence_text


def adaptive_retrieval(question: str, initial_top_k=INITIAL_TOP_K, expanded_top_k=EXPANDED_TOP_K, threshold=EVIDENCE_THRESHOLD, neighbor_window=1):
    initial_results = retrieve_context(question, top_k=initial_top_k)
    initial_score = calculate_evidence_score(initial_results)

    retrieved = initial_results
    evidence_score = initial_score
    retry_performed = False
    neighbor_chunks_added = 0

    if evidence_score < threshold:
        retry_performed = True
        expanded_results = retrieve_context(question, top_k=expanded_top_k)
        all_results = retrieve_context(question, top_k=len(KNOWLEDGE_BASE))

        chunk_positions = {item["chunk_id"]: idx for idx, item in enumerate(KNOWLEDGE_BASE)}
        scored_chunks = {item["chunk_id"]: item for item in all_results}
        selected_ids = {item["chunk_id"] for item in expanded_results}

        for item in expanded_results:
            cid = item["chunk_id"]
            if cid not in chunk_positions:
                continue
            pos = chunk_positions[cid]
            s_pos = max(0, pos - neighbor_window)
            e_pos = min(len(KNOWLEDGE_BASE), pos + neighbor_window + 1)
            for p in range(s_pos, e_pos):
                selected_ids.add(KNOWLEDGE_BASE[p]["chunk_id"])

        retrieved = [scored_chunks[cid].copy() for cid in selected_ids if cid in scored_chunks]
        neighbor_chunks_added = max(0, len(retrieved) - len(expanded_results))
        retrieved.sort(key=lambda x: x["similarity"], reverse=True)
        evidence_score = calculate_evidence_score(retrieved)

    return {
        "retrieved": retrieved,
        "initial_evidence_score": initial_score,
        "evidence_score": evidence_score,
        "retry_performed": retry_performed,
        "neighbor_chunks_added": neighbor_chunks_added,
    }


# =========================================================
# MULTILINGUAL ANSWER GENERATION & VERIFICATION
# =========================================================

def generate_answer(question: str, retrieved: list, response_language: str = "English"):
    context = format_evidence(retrieved)
    groq_client = get_groq_client()

    script_rules = ""
    if response_language == "Hindi":
        script_rules = "Write the entire answer in Hindi using Devanagari script (देवनागरी). Technical terms like code and functions should remain in English/Latin script."
    elif response_language == "Urdu":
        script_rules = "Write the entire answer in Urdu using standard Urdu script (اردو)."
    elif response_language == "Hinglish":
        script_rules = "Write the answer in conversational Hinglish (Hindi written in Roman letters, e.g. 'Is video me speaker ne bataya hai...')."

    prompt = f"""
You are an educational video question-answering assistant.

Answer the user's question using ONLY the supplied transcript evidence.

RESPONSE LANGUAGE:
Write the complete answer in {response_language}.
{script_rules}

STRICT RULES:
1. Do not use outside knowledge or textbook definitions absent from the transcript.
2. Cite every factual statement using [Chunk ID].
3. If evidence is insufficient, clearly state that the transcript evidence is insufficient.
4. Keep technical and code keywords intact.
5. Do not translate chunk citations (e.g. keep [Chunk 12]).

EVIDENCE:
{context}

USER QUESTION:
{question}
"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a strict educational video Q&A assistant. Ground every claim in the supplied evidence."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def verify_claims(answer: str, retrieved: list):
    """Verify answer claims against transcript evidence; abstentions are not supported answers."""
    answer_lower = answer.lower().strip()
    abstention_patterns = [
        r"transcript.{0,100}(does not contain|doesn't contain|contains no|has no information)",
        r"(provided )?evidence.{0,100}(insufficient|does not contain|doesn't contain|not enough)",
        r"cannot (provide|give|answer).{0,120}(based on|from|using).{0,80}(evidence|transcript)",
        r"can't (provide|give|answer).{0,120}(based on|from|using).{0,80}(evidence|transcript)",
        r"insufficient (information|evidence|transcript)",
        r"not enough (information|evidence)",
        r"available (video )?evidence.{0,100}(insufficient|not enough)",
        r"पर्याप्त नहीं हैं",
        r"काफी नहीं हैं",
        r"معلومات کافی نہیں",
    ]

    if any(re.search(pattern, answer_lower, flags=re.IGNORECASE) for pattern in abstention_patterns):
        return {
            "claims": [],
            "overall_status": "ABSTAINED",
            "reason": "The answer explicitly states that the supplied transcript evidence is insufficient."
        }

    if not retrieved:
        return {"claims": [], "overall_status": "NOT_VERIFIED", "reason": "No transcript evidence was retrieved."}

    evidence_map = {item["chunk_id"]: item for item in retrieved if "chunk_id" in item}
    evidence_context = "\\n".join(
        f"ACTUAL CHUNK ID: {cid}\\nTRANSCRIPT: {item.get('text', '')}"
        for cid, item in evidence_map.items()
    )

    prompt = f"""
You are a strict evidence verification system.
Verify each factual claim in the answer using ONLY the supplied transcript evidence.
Do not treat the answer's wording or its citations as proof.

ANSWER:
{answer}

RETRIEVED EVIDENCE:
{evidence_context}

Return ONLY valid JSON with this schema:
{{
  "claims": [
    {{"claim": "claim text", "status": "SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED", "evidence_chunks": [1], "reason": "why"}}
  ],
  "overall_status": "SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED|NOT_VERIFIED"
}}
If there are no factual claims to verify, use NOT_VERIFIED. Never return SUPPORTED when evidence does not directly support the claim.
"""
    try:
        response = get_groq_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", raw, flags=re.IGNORECASE)
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
            raise ValueError("Invalid verification JSON schema.")
        allowed = {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "NOT_VERIFIED", "ABSTAINED"}
        if data.get("overall_status") not in allowed:
            raise ValueError("Unknown verification status.")
        return data
    except Exception as e:
        return {"claims": [], "overall_status": "NOT_VERIFIED", "error": str(e),
                "reason": "Claim verification could not be completed."}


def self_reflection(question: str, answer: str, retrieved: list):
    """Self-reflection must fail closed: an API/parse failure is never an automatic pass."""
    context = format_evidence(retrieved)
    prompt = f"""
Review the generated answer against the transcript evidence only.

QUESTION: {question}
ANSWER: {answer}
EVIDENCE: {context}

Return ONLY valid JSON:
{{
  "addresses_question": true,
  "evidence_consistency": "CONSISTENT|PARTIALLY_CONSISTENT|INCONSISTENT|INSUFFICIENT_EVIDENCE",
  "unsupported_claims": [],
  "further_retrieval_needed": false,
  "overall_assessment": "ACCEPTABLE|NEEDS_REVIEW|NOT_VERIFIED",
  "reason": "Brief explanation"
}}
If the transcript does not contain evidence needed to answer, use INSUFFICIENT_EVIDENCE and NOT_VERIFIED.
"""
    try:
        response = get_groq_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", raw, flags=re.IGNORECASE)
        data = json.loads(raw)
        if not isinstance(data, dict) or "overall_assessment" not in data:
            raise ValueError("Invalid self-reflection JSON schema.")
        return data
    except Exception as e:
        return {
            "addresses_question": False,
            "evidence_consistency": "INSUFFICIENT_EVIDENCE",
            "unsupported_claims": [],
            "further_retrieval_needed": False,
            "overall_assessment": "NOT_VERIFIED",
            "reason": "Self-reflection could not be completed; verification failed closed.",
            "error": str(e)
        }


def detect_hallucinations(answer, evidence_score, reflection_result, verification_result):
    reasons = []
    claims = verification_result.get("claims", [])
    unsupported = [c for c in claims if c.get("status") == "UNSUPPORTED"]
    partially = [c for c in claims if c.get("status") == "PARTIALLY_SUPPORTED"]

    if unsupported:
        reasons.append("Claim verification identified unsupported claims.")
    if partially:
        reasons.append("Some claims are only partially supported.")
    if evidence_score < EVIDENCE_THRESHOLD:
        reasons.append("Retrieval confidence is below the threshold.")

    if verification_result.get("overall_status") == "ABSTAINED":
        assessment = "ABSTAINED"
    elif unsupported:
        assessment = "POTENTIAL_HALLUCINATION"
    elif partially or evidence_score < EVIDENCE_THRESHOLD:
        assessment = "NEEDS_REVIEW"
    else:
        assessment = "NO_HALLUCINATION_DETECTED"

    return {
        "assessment": assessment,
        "evidence_score": float(evidence_score),
        "reasons": reasons,
    }


def save_decision_trace(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUTPUT_DIR / "decision_trace.json"

    records = []
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                records = json.load(file)
                if not isinstance(records, list):
                    records = []
        except Exception:
            records = []

    record = {
        "timestamp": datetime.now().isoformat(),
        "question": result.get("question"),
        "response_language": result.get("response_language", "English"),
        "answer": result.get("answer"),
        "retrieval": {
            "initial_confidence": result.get("initial_evidence_score"),
            "final_confidence": result.get("evidence_score"),
            "retry_performed": result.get("retry_performed")
        },
        "reflection": result.get("reflection"),
        "verification": result.get("verification"),
        "hallucination_detection": result.get("hallucination_detection"),
        "answer_outcome": result.get("answer_outcome"),
        "retrieved_evidence": result.get("evidence", []),
    }

    records.append(record)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=4, ensure_ascii=False, default=str)

    return record


def assess_evidence_relevance(question: str, retrieved: list):
    """
    Judge whether retrieved passages actually contain evidence relevant to answering
    this question. Similarity scores alone are not evidence of answerability.
    """
    if not retrieved:
        return {"sufficient": False, "status": "INSUFFICIENT_EVIDENCE",
                "reason": "No transcript passages were retrieved."}

    evidence = format_evidence(retrieved)
    prompt = f"""
You are an evidence-relevance gate for a video question-answering system.
Decide whether the transcript passages below contain direct, useful information
needed to answer the user's question. Topical word overlap is NOT enough.
If passages discuss a different subject, or only share generic words, mark insufficient.
Do not use outside knowledge.

QUESTION:
{question}

TRANSCRIPT PASSAGES:
{evidence}

Return ONLY JSON:
{{"sufficient": true, "status": "SUFFICIENT|INSUFFICIENT_EVIDENCE",
  "reason": "short explanation"}}
"""
    try:
        response = get_groq_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", raw, flags=re.IGNORECASE)
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("sufficient"), bool):
            raise ValueError("Invalid evidence relevance response.")
        return {
            "sufficient": data["sufficient"],
            "status": data.get("status", "SUFFICIENT" if data["sufficient"] else "INSUFFICIENT_EVIDENCE"),
            "reason": data.get("reason", "")
        }
    except Exception as e:
        # Fail closed: never answer as though relevance was verified.
        return {"sufficient": False, "status": "NOT_VERIFIED",
                "reason": "Evidence relevance check failed; answer withheld.",
                "error": str(e)}


def ask_sr_rag(question: str, top_k=INITIAL_TOP_K, expanded_top_k=EXPANDED_TOP_K, response_language="English"):
    if not KNOWLEDGE_BASE or KNOWLEDGE_EMBEDDINGS is None:
        raise ValueError("Knowledge base is empty. Please process a video first.")
    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    retrieval_info = adaptive_retrieval(
        question, initial_top_k=top_k, expanded_top_k=expanded_top_k
    )
    retrieved = retrieval_info["retrieved"]
    initial_score = retrieval_info["initial_evidence_score"]
    evidence_score = retrieval_info["evidence_score"]
    retry_performed = retrieval_info["retry_performed"]

    # Semantic relevance check is the decisive evidence gate. The numeric similarity
    # score is retained for diagnostics but is not treated as proof of relevance.
    relevance = assess_evidence_relevance(question, retrieved)

    abstention_messages = {
        "English": "The retrieved video transcript evidence is insufficient to answer this question reliably.",
        "Hindi": "क्षमा करें, उपलब्ध ट्रांसक्रिप्ट साक्ष्य इस प्रश्न का विश्वसनीय उत्तर देने के लिए पर्याप्त नहीं हैं।",
        "Urdu": "معذرت، دستیاب ٹرانسکرپٹ شواہد اس سوال کا قابلِ اعتماد جواب دینے کے لیے کافی نہیں ہیں۔",
        "Hinglish": "Transcript evidence is sawal ka jawab dene ke liye insufficient hai."
    }

    if not relevance["sufficient"]:
        answer = abstention_messages.get(response_language, abstention_messages["English"])
        verification = {
            "claims": [],
            "overall_status": "ABSTAINED" if relevance["status"] == "INSUFFICIENT_EVIDENCE" else "NOT_VERIFIED",
            "reason": relevance["reason"]
        }
        reflection = {
            "addresses_question": False,
            "evidence_consistency": "INSUFFICIENT_EVIDENCE",
            "unsupported_claims": [],
            "further_retrieval_needed": False,
            "overall_assessment": "NOT_VERIFIED",
            "reason": relevance["reason"]
        }
    else:
        answer = generate_answer(question, retrieved, response_language)
        reflection = self_reflection(question, answer, retrieved)
        verification = verify_claims(answer, retrieved)

        # If generation itself abstained, preserve abstention all the way to the UI.
        if verification.get("overall_status") == "ABSTAINED":
            relevance["sufficient"] = False

    hallucination_result = detect_hallucinations(
        answer, evidence_score, reflection, verification
    )

    verification_status = verification.get("overall_status", "NOT_VERIFIED")
    reflection_status = reflection.get("overall_assessment", "NOT_VERIFIED")

    if verification_status == "ABSTAINED":
        outcome = "ABSTAINED"
    elif (not relevance["sufficient"] or verification_status == "NOT_VERIFIED"
          or reflection_status == "NOT_VERIFIED"):
        outcome = "ABSTAINED" if verification_status == "ABSTAINED" else "NEEDS_REVIEW"
    elif (verification_status in {"UNSUPPORTED", "PARTIALLY_SUPPORTED"}
          or reflection_status == "NEEDS_REVIEW"
          or hallucination_result["assessment"] != "NO_HALLUCINATION_DETECTED"):
        outcome = "NEEDS_REVIEW"
    else:
        outcome = "ANSWERED"

    result = {
        "question": question,
        "response_language": response_language,
        "answer": answer,
        "evidence": retrieved,
        "evidence_score": evidence_score,
        "initial_evidence_score": initial_score,
        "evidence_relevance": relevance,
        "reflection": reflection,
        "verification": verification,
        "hallucination_detection": hallucination_result,
        "answer_outcome": outcome,
        "retry_performed": retry_performed,
    }
    save_decision_trace(result)
    return result


def generate_video_summary(response_language: str = "English", knowledge_base=None):
    if knowledge_base is None:
        knowledge_base = KNOWLEDGE_BASE
    if not knowledge_base:
        raise ValueError("No video transcript is currently loaded.")

    step = max(1, len(knowledge_base) // 18)
    selected = knowledge_base[::step][:18]
    digest = "\\n\\n".join(
        f"[Passage {c.get('chunk_id')} | {c.get('start_time', 0):.1f}s - "
        f"{c.get('end_time', 0):.1f}s]: {c.get('text', '')}"
        for c in selected
    )

    prompt = f"""
Analyze these video transcript excerpts and generate a structured summary in {response_language}.
Use ONLY the excerpts. Do not add facts from outside knowledge.

EXCERPTS:
{digest}

Return ONLY valid JSON:
{{
  "summary": "2-3 paragraphs summarizing what the video teaches",
  "key_takeaways": ["Takeaway 1", "Takeaway 2", "Takeaway 3", "Takeaway 4"],
  "topics": [{{"title": "Topic name", "start_time": 0.0, "end_time": 120.0,
              "description": "Brief description"}}]
}}
"""
    try:
        response = get_groq_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", raw, flags=re.IGNORECASE)
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("summary"), str):
            raise ValueError("Invalid summary response schema.")
        data.setdefault("key_takeaways", [])
        data.setdefault("topics", [])
        data["generation_status"] = "SUCCESS"
        return data
    except Exception as e:
        # Never return a fabricated, topic-specific fallback summary.
        return {
            "summary": "Summary could not be generated because the language-model request failed. Please try again.",
            "key_takeaways": [],
            "topics": [],
            "generation_status": "FAILED",
            "error": str(e)
        }
