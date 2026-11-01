from faster_whisper import WhisperModel

model = WhisperModel(
    "large-v3",
    device="cpu",
    local_files_only=True,
)

def transcribe_audio(file_path: str) -> str:
    """Transcribe speech from an audio file into text."""
    # audio in ./files
    file_path = f"./files/{file_path}"
    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        vad_filter=False,
    )

    return "\n".join(
        f"[{segment.start:.1f}s-{segment.end:.1f}s] "
        f"{segment.text.strip()}"
        for segment in segments
    )

from llama_index.core.tools import FunctionTool

transcribe_audio_tool = FunctionTool.from_defaults(
    fn=transcribe_audio,
    name="transcribe_audio",
    description=(
        "Transcribe speech from an audio file. "
        "Use when the question requires knowing what is said in audio. "
        "Input: the path to the audio file. "
        "Output: transcript with segment timestamps."
    ),
)