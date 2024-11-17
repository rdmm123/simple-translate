from pathlib import Path
from typing import Annotated
from soni_translate.soni_translate import SoniTranslate
from dataclasses import dataclass

@dataclass
class TranslateSettings:
    SPEAKER_VOICE: Annotated[str, "The voice of the speaker. One of Edge TTS voices."] = "es-CL-LorenzoNeural-Male"
    SUBTITLE_TYPE: Annotated[str, "File format for subtitles (ex: srt)."] = "disable"
    BATCH_SIZE: Annotated[int, "Translation batch size. Increase for faster but more resource heavy execution."] = 16
    FROM: Annotated[str, "Original language of video"] = "English (en)"
    TO: Annotated[str, "Language to translate video to"] = "Spanish (es)"
    AUDIO_ACCELERATION: Annotated[float, "Audio acceleration of speaker"] = 1.5
    PREVIEW: Annotated[bool, "Generate a 10s preview of the translated video."] = False

def translate_video(path: Path, settings: TranslateSettings | None = None) -> str:
    if not settings:
        settings = TranslateSettings()

    if not path.exists():
        print(f"File {path} does not exist. Aborting.")

    soni = SoniTranslate(cpu_mode=False)
    print(f"Beginning translation {path}")
    output_path = soni.multilingual_media_conversion(
        directory_input=str(path),
        origin_language=TranslateSettings.FROM,
        target_language=TranslateSettings.TO,
        tts_voice00=TranslateSettings.SPEAKER_VOICE,
        soft_subtitles_to_video=True,
        batch_size=TranslateSettings.BATCH_SIZE,
        max_accelerate_audio=TranslateSettings.AUDIO_ACCELERATION,
        volume_original_audio=0.1,
        volume_translated_audio=1.80,
        output_format_subtitle=TranslateSettings.SUBTITLE_TYPE,
        preview=TranslateSettings.PREVIEW,

    )
    print(f"Translated file created at {output_path}")
    return output_path
