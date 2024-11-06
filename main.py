import os
from pathlib import Path
from soni_translate.soni_translate import SoniTranslate
from soni_translate.mdx_net import (
    UVR_MODELS,
    MDX_DOWNLOAD_LINK,
    mdxnet_models_dir,
)
from soni_translate.utils import download_manager
from dataclasses import dataclass

@dataclass
class TranslateSettings:
    SPEAKER_VOICE: str = "es-CL-LorenzoNeural-Male"
    INPUT_FOLDER: Path = Path('/home/rdmm123/random')
    SUBTITLE_TYPE: str = "disable"
    COMPUTE_TYPE: str = "float16"
    BATCH_SIZE: int = 16
    ORIGIN_LANGUAGE: str = "English (en)"
    TRANSLATE_LANGUAGE: str = "Spanish (es)"
    AUDIO_ACCELERATION: float = 1.5
    PREVIEW: bool = False

def init() -> None:
    for id_model in UVR_MODELS:
        download_manager(
            os.path.join(MDX_DOWNLOAD_LINK, id_model), mdxnet_models_dir
        )

def translate_video(soni: SoniTranslate, path: Path) -> str:
    print(f"Beginning translation {path}")
    output_path = soni.multilingual_media_conversion(
        directory_input=str(path),
        origin_language=TranslateSettings.ORIGIN_LANGUAGE,
        target_language=TranslateSettings.TRANSLATE_LANGUAGE,
        tts_voice00=TranslateSettings.SPEAKER_VOICE,
        soft_subtitles_to_video=True,
        batch_size=TranslateSettings.BATCH_SIZE,
        max_accelerate_audio=TranslateSettings.AUDIO_ACCELERATION,
        volume_original_audio=0.25,
        volume_translated_audio=1.80,
        output_format_subtitle=TranslateSettings.SUBTITLE_TYPE,
        voice_imitation=TranslateSettings.VOICE_IMITATION,
        preview=TranslateSettings.PREVIEW,

    )
    print(f"Translated file created at {output_path}")
    return output_path

def main() -> None:
    # init()
    soni = SoniTranslate(cpu_mode=False)
    video_path = TranslateSettings.INPUT_FOLDER / 'test_out.mp4'
    translate_video(soni, video_path)

if __name__ == '__main__':
    main()
