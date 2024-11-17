import argparse
from pathlib import Path
from soni_translate.soni_translate import SoniTranslate
from dataclasses import dataclass

@dataclass
class TranslateSettings:
    SPEAKER_VOICE: str = "es-CL-LorenzoNeural-Male"
    SUBTITLE_TYPE: str = "disable"
    COMPUTE_TYPE: str = "float16"
    BATCH_SIZE: int = 16
    ORIGIN_LANGUAGE: str = "English (en)"
    TRANSLATE_LANGUAGE: str = "Spanish (es)"
    AUDIO_ACCELERATION: float = 1.5
    PREVIEW: bool = True

def translate_video(path: Path) -> str:
    soni = SoniTranslate(cpu_mode=False)
    print(f"Beginning translation {path}")
    output_path = soni.multilingual_media_conversion(
        directory_input=str(path),
        origin_language=TranslateSettings.ORIGIN_LANGUAGE,
        target_language=TranslateSettings.TRANSLATE_LANGUAGE,
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        help='Path to the video to translate'
    )
    args = parser.parse_args()
    translate_video(args.input)

if __name__ == '__main__':
    main()
