import copy
import hashlib
import os

import torch

from soni_translate.audio_segments import create_translated_audio
from soni_translate.language_configuration import (
    BARK_VOICES_LIST,
    LANGUAGES,
    OPENAI_TTS_MODELS,
    UNIDIRECTIONAL_L_LIST,
    VITS_VOICES_LIST,
)
from soni_translate.logging_setup import (
    configure_logging_libs,
    logger,
)
from soni_translate.postprocessor import media_out

from soni_translate.preprocessor import audio_video_preprocessor
from soni_translate.speech_segmentation import (
    COMPUTE_TYPE_CPU,
    align_speech,
    diarization_models,
    diarize_speech,
    transcribe_speech,
)
from soni_translate.text_multiformat_processor import (
    break_aling_segments,
    linguistic_level_segments,
    process_subtitles,
)
from soni_translate.text_to_speech import (
    accelerate_segments,
    audio_segmentation_to_voice,
    coqui_xtts_voices_list,
    edge_tts_voices_list,
    piper_tts_voices_list,
)
from soni_translate.translate_segments import (
    translate_text,
)
from soni_translate.utils import (
    is_audio_file,
    is_subtitle_file,
    remove_files,
    run_command,
)
from soni_translate.voice_main import ClassVoices

from inspect import currentframe


configure_logging_libs()  # noqa

class TTS_Info:
    def __init__(self, piper_enabled, xtts_enabled):
        self.list_edge = edge_tts_voices_list()
        self.list_bark = list(BARK_VOICES_LIST.keys())
        self.list_vits = list(VITS_VOICES_LIST.keys())
        self.list_openai_tts = OPENAI_TTS_MODELS
        self.piper_enabled = piper_enabled
        self.list_vits_onnx = piper_tts_voices_list() if self.piper_enabled else []
        self.xtts_enabled = xtts_enabled

    def tts_list(self):
        self.list_coqui_xtts = coqui_xtts_voices_list() if self.xtts_enabled else []
        list_tts = self.list_coqui_xtts + sorted(
            self.list_edge
            + self.list_bark
            + self.list_vits
            + self.list_openai_tts
            + self.list_vits_onnx
        )
        return list_tts


def prog_disp(msg, percent, is_gui, progress=None):
    logger.info(msg)


def warn_disp(wrn_lang, is_gui):
    logger.warning(wrn_lang)

def log_with_line_no(msg: str):
    logger.info(f"At line {currentframe().f_back.f_lineno} | {msg}")

class SoniTrCache:
    def __init__(self):
        self.cache = {
            "media": [[]],
            "refine_vocals": [],
            "transcript_align": [],
            "break_align": [],
            "diarize": [],
            "translate": [],
            "subs_and_edit": [],
            "tts": [],
            "acc_and_vc": [],
            "mix_aud": [],
            "output": [],
        }

        self.cache_data = {
            "media": [],
            "refine_vocals": [],
            "transcript_align": [],
            "break_align": [],
            "diarize": [],
            "translate": [],
            "subs_and_edit": [],
            "tts": [],
            "acc_and_vc": [],
            "mix_aud": [],
            "output": [],
        }

        self.cache_keys = list(self.cache.keys())
        self.first_task = self.cache_keys[0]
        self.last_task = self.cache_keys[-1]

        self.pre_step = None
        self.pre_params = []

    def set_variable(self, variable_name, value):
        setattr(self, variable_name, value)

    def task_in_cache(self, step: str, params: list, previous_step_data: dict):
        self.pre_step_cache = None

        if step == self.first_task:
            self.pre_step = None

        if self.pre_step:
            self.cache[self.pre_step] = self.pre_params

            # Fill data in cache
            self.cache_data[self.pre_step] = copy.deepcopy(previous_step_data)

        self.pre_params = params
        # logger.debug(f"Step: {str(step)}, Cache params: {str(self.cache)}")
        if params == self.cache[step]:
            logger.debug(f"In cache: {str(step)}")

            # Set the var needed for next step
            # Recovery from cache_data the current step
            for key, value in self.cache_data[step].items():
                self.set_variable(key, copy.deepcopy(value))
                logger.debug(f"Chache load: {str(key)}")

            self.pre_step = step
            return True

        else:
            logger.debug(f"Flush next and caching {str(step)}")
            selected_index = self.cache_keys.index(step)

            for idx, key in enumerate(self.cache.keys()):
                if idx >= selected_index:
                    self.cache[key] = []
                    self.cache_data[key] = {}

            # The last is now previous
            self.pre_step = step
            return False

    def clear_cache(self, media, force=False):
        self.cache["media"] = self.cache["media"] if len(self.cache["media"]) else [[]]

        if media != self.cache["media"][0] or force:
            # Clear cache
            self.cache = {key: [] for key in self.cache}
            self.cache["media"] = [[]]

            logger.info("Cache flushed")


def get_hash(filepath):
    with open(filepath, "rb") as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:18]


def check_openai_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "To use GPT for translation, please set up your OpenAI API key "
            "as an environment variable in Linux as follows: "
            "export OPENAI_API_KEY='your-api-key-here'. Or change the "
            "translation process in Advanced settings."
        )


class SoniTranslate(SoniTrCache):
    def __init__(self, cpu_mode=False):
        super().__init__()
        if cpu_mode:
            os.environ["SONITR_DEVICE"] = "cpu"
        else:
            os.environ["SONITR_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = os.environ.get("SONITR_DEVICE")
        self.result_diarize = None
        self.align_language = None
        self.result_source_lang = None
        self.edit_subs_complete = False
        self.voiceless_id = None
        self.burn_subs_id = None

        self.vci = ClassVoices(only_cpu=cpu_mode)
        self.vocals = None

        logger.info(f"Working in: {self.device}")

    def multilingual_media_conversion(
        self,
        media_file=None,
        link_media="",
        directory_input="",
        YOUR_HF_TOKEN="",
        preview=False,
        transcriber_model="large-v3",
        batch_size=4,
        compute_type="auto",
        origin_language="Automatic detection",
        target_language="English (en)",
        min_speakers=1,
        max_speakers=1,
        tts_voice00="en-US-EmmaMultilingualNeural-Female",
        tts_voice01="en-US-AndrewMultilingualNeural-Male",
        tts_voice02="en-US-AvaMultilingualNeural-Female",
        tts_voice03="en-US-BrianMultilingualNeural-Male",
        tts_voice04="de-DE-SeraphinaMultilingualNeural-Female",
        tts_voice05="de-DE-FlorianMultilingualNeural-Male",
        tts_voice06="fr-FR-VivienneMultilingualNeural-Female",
        tts_voice07="fr-FR-RemyMultilingualNeural-Male",
        tts_voice08="en-US-EmmaMultilingualNeural-Female",
        tts_voice09="en-US-AndrewMultilingualNeural-Male",
        tts_voice10="en-US-EmmaMultilingualNeural-Female",
        tts_voice11="en-US-AndrewMultilingualNeural-Male",
        video_output_name="",
        mix_method_audio="Adjusting volumes and mixing audio",
        max_accelerate_audio=2.1,
        acceleration_rate_regulation=False,
        volume_original_audio=0.25,
        volume_translated_audio=1.80,
        output_format_subtitle="srt",
        get_translated_text=False,
        get_video_from_text_json=False,
        text_json="{}",
        avoid_overlap=False,
        vocal_refinement=False,
        literalize_numbers=True,
        segment_duration_limit=15,
        diarization_model="pyannote_2.1",
        translate_process="google_translator_batch",
        subtitle_file=None,
        output_type="video (mp4)",
        voiceless_track=False,
        voice_imitation=False,
        voice_imitation_max_segments=3,
        voice_imitation_vocals_dereverb=False,
        voice_imitation_remove_previous=True,
        voice_imitation_method="freevc",
        dereverb_automatic_xtts=True,
        text_segmentation_scale="sentence",
        divide_text_segments_by="",
        soft_subtitles_to_video=True,
        burn_subtitles_to_video=False,
        enable_cache=True,
        custom_voices=False,
        custom_voices_workers=1,
        is_gui=False,
        progress=None,
    ):
        if not YOUR_HF_TOKEN:
            YOUR_HF_TOKEN = os.getenv("YOUR_HF_TOKEN")
            if diarization_model == "disable" or max_speakers == 1:
                if YOUR_HF_TOKEN is None:
                    YOUR_HF_TOKEN = ""
            elif not YOUR_HF_TOKEN:
                raise ValueError("No valid Hugging Face token")
            else:
                os.environ["YOUR_HF_TOKEN"] = YOUR_HF_TOKEN

        if (
            "gpt" in translate_process
            or transcriber_model == "OpenAI_API_Whisper"
            or "OpenAI-TTS" in tts_voice00
        ):
            check_openai_api_key()

        if media_file is None:
            media_file = (
                directory_input if os.path.exists(directory_input) else link_media
            )
        media_file = media_file if isinstance(media_file, str) else media_file.name

        if is_subtitle_file(media_file):
            subtitle_file = media_file
            media_file = ""

        if media_file is None:
            media_file = ""

        if not origin_language:
            origin_language = "Automatic detection"

        if origin_language in UNIDIRECTIONAL_L_LIST and not subtitle_file:
            raise ValueError(
                f"The language '{origin_language}' "
                "is not supported for transcription (ASR)."
            )

        TRANSLATE_AUDIO_TO = LANGUAGES[target_language]
        SOURCE_LANGUAGE = LANGUAGES[origin_language]

        if (
            text_segmentation_scale in ["word", "character"]
            and "subtitle" not in output_type
        ):
            wrn_lang = (
                "Text segmentation by words or characters is typically"
                " used for generating subtitles. If subtitles are not the"
                " intended output, consider selecting 'sentence' "
                "segmentation method to ensure optimal results."
            )
            warn_disp(wrn_lang, is_gui)

        if tts_voice00[:2].lower() != TRANSLATE_AUDIO_TO[:2].lower():
            wrn_lang = (
                "Make sure to select a 'TTS Speaker' suitable for"
                " the translation language to avoid errors with the TTS."
            )
            warn_disp(wrn_lang, is_gui)

        if not media_file and not subtitle_file:
            raise ValueError("Specifify a media or SRT file in advanced settings")

        # Check GPU
        if self.device == "cpu" and compute_type not in COMPUTE_TYPE_CPU:
            logger.info("Compute type changed to float32")
            compute_type = "float32"

        base_video_file = "Video.mp4"
        base_audio_wav = "audio.wav"
        dub_audio_file = "audio_dub_solo.ogg"
        mix_audio_file = "audio_mix.mp3"
        vid_subs = "video_subs_file.mp4"
        video_output_file = "video_dub.mp4"

        if os.path.exists(media_file):
            media_base_hash = get_hash(media_file)
        else:
            media_base_hash = media_file
        self.clear_cache(media_base_hash, force=(not enable_cache))

        self.result_diarize = self.align_language = self.result_source_lang = None
        if not self.task_in_cache("media", [media_base_hash, preview], {}):
            prog_disp("Processing video...", 0.15, is_gui, progress=progress)
            audio_video_preprocessor(
                preview, media_file, base_video_file, base_audio_wav
            )
            logger.debug("Set file complete.")

        if not self.task_in_cache(
            "transcript_align",
            [
                subtitle_file,
                SOURCE_LANGUAGE,
                transcriber_model,
                compute_type,
                batch_size,
                literalize_numbers,
                segment_duration_limit,
                (
                    "l_unit"
                    if text_segmentation_scale in ["word", "character"]
                    and subtitle_file
                    else "sentence"
                ),
            ],
            {"vocals": self.vocals},
        ):
            prog_disp("Transcribing...", 0.30, is_gui, progress=progress)
            SOURCE_LANGUAGE = (
                None
                if SOURCE_LANGUAGE == "Automatic detection"
                else SOURCE_LANGUAGE
            )
            audio, self.result = transcribe_speech(
                base_audio_wav if not self.vocals else self.vocals,
                transcriber_model,
                compute_type,
                batch_size,
                SOURCE_LANGUAGE,
                literalize_numbers,
                segment_duration_limit,
            )
            logger.debug(
                "Transcript complete, "
                f"segments count {len(self.result['segments'])}"
            )

            self.align_language = self.result["language"]
            if not subtitle_file or text_segmentation_scale in [
                "word",
                "character",
            ]:
                prog_disp("Aligning...", 0.45, is_gui, progress=progress)
                try:
                    if self.align_language in ["vi"]:
                        logger.info(
                            "Deficient alignment for the "
                            f"{self.align_language} language, skipping the"
                            " process. It is suggested to reduce the "
                            "duration of the segments as an alternative."
                        )
                    else:
                        self.result = align_speech(audio, self.result)
                        logger.debug(
                            "Align complete, "
                            f"segments count {len(self.result['segments'])}"
                        )
                except Exception as error:
                    logger.error(str(error))

        if self.result["segments"] == []:
            raise ValueError("No active speech found in audio")

        if not self.task_in_cache(
            "break_align",
            [divide_text_segments_by, text_segmentation_scale, self.align_language],
            {"result": self.result, "align_language": self.align_language},
        ):
            if self.align_language in ["ja", "zh", "zh-TW"]:
                divide_text_segments_by += "|!|?|...|。"
            if text_segmentation_scale in ["word", "character"]:
                self.result = linguistic_level_segments(
                    self.result,
                    text_segmentation_scale,
                )
            elif divide_text_segments_by:
                try:
                    self.result = break_aling_segments(
                        self.result,
                        break_characters=divide_text_segments_by,
                    )
                except Exception as error:
                    logger.error(str(error))

        if not self.task_in_cache(
            "diarize",
            [
                min_speakers,
                max_speakers,
                YOUR_HF_TOKEN[: len(YOUR_HF_TOKEN) // 2],
                diarization_model,
            ],
            {"result": self.result},
        ):
            prog_disp("Diarizing...", 0.60, is_gui, progress=progress)
            diarize_model_select = diarization_models[diarization_model]
            self.result_diarize = diarize_speech(
                base_audio_wav if not self.vocals else self.vocals,
                self.result,
                min_speakers,
                max_speakers,
                YOUR_HF_TOKEN,
                diarize_model_select,
            )
            logger.debug("Diarize complete")
        self.result_source_lang = copy.deepcopy(self.result_diarize)

        if not self.task_in_cache(
            "translate",
            [TRANSLATE_AUDIO_TO, translate_process],
            {"result_diarize": self.result_diarize},
        ):
            prog_disp("Translating...", 0.70, is_gui, progress=progress)
            lang_source = (
                self.align_language if self.align_language else SOURCE_LANGUAGE
            )
            self.result_diarize["segments"] = translate_text(
                self.result_diarize["segments"],
                TRANSLATE_AUDIO_TO,
                translate_process,
                chunk_size=1800,
                source=lang_source,
            )
            logger.debug("Translation complete")
            logger.debug(self.result_diarize)

        # Write subtitle
        if not self.task_in_cache(
            "subs_and_edit",
            [
                copy.deepcopy(self.result_diarize),
                output_format_subtitle,
                TRANSLATE_AUDIO_TO,
            ],
            {"result_diarize": self.result_diarize},
        ):
            if output_format_subtitle == "disable":
                self.sub_file = "sub_tra.srt"
            elif output_format_subtitle != "ass":
                self.sub_file = process_subtitles(
                    self.result_source_lang,
                    self.align_language,
                    self.result_diarize,
                    output_format_subtitle,
                    TRANSLATE_AUDIO_TO,
                )

            # Need task
            if output_format_subtitle != "srt":
                _ = process_subtitles(
                    self.result_source_lang,
                    self.align_language,
                    self.result_diarize,
                    "srt",
                    TRANSLATE_AUDIO_TO,
                )

            if output_format_subtitle == "ass":
                convert_ori = "ffmpeg -i sub_ori.srt sub_ori.ass -y"
                convert_tra = "ffmpeg -i sub_tra.srt sub_tra.ass -y"
                self.sub_file = "sub_tra.ass"
                run_command(convert_ori)
                run_command(convert_tra)

        if "video [subtitled]" in output_type:
            output = media_out(
                media_file,
                TRANSLATE_AUDIO_TO + "_subtitled",
                video_output_name,
                "wav"
                if is_audio_file(media_file)
                else ("mkv" if "mkv" in output_type else "mp4"),
                file_obj=base_audio_wav
                if is_audio_file(media_file)
                else base_video_file,
                soft_subtitles=False if is_audio_file(media_file) else True,
                subtitle_files=output_format_subtitle,
            )
            msg_out = output[0] if isinstance(output, list) else output
            logger.info(f"Done: {msg_out}")
            return output

        if not self.task_in_cache(
            "tts",
            [
                TRANSLATE_AUDIO_TO,
                tts_voice00,
                tts_voice01,
                tts_voice02,
                tts_voice03,
                tts_voice04,
                tts_voice05,
                tts_voice06,
                tts_voice07,
                tts_voice08,
                tts_voice09,
                tts_voice10,
                tts_voice11,
                dereverb_automatic_xtts,
            ],
            {"sub_file": self.sub_file},
        ):
            prog_disp("Text to speech...", 0.80, is_gui, progress=progress)
            self.valid_speakers = audio_segmentation_to_voice(
                self.result_diarize,
                TRANSLATE_AUDIO_TO,
                is_gui,
                tts_voice00,
                tts_voice01,
                tts_voice02,
                tts_voice03,
                tts_voice04,
                tts_voice05,
                tts_voice06,
                tts_voice07,
                tts_voice08,
                tts_voice09,
                tts_voice10,
                tts_voice11,
                dereverb_automatic_xtts,
            )

        if not self.task_in_cache(
            "acc_and_vc",
            [
                max_accelerate_audio,
                acceleration_rate_regulation,
                voice_imitation,
                voice_imitation_max_segments,
                voice_imitation_remove_previous,
                voice_imitation_vocals_dereverb,
                voice_imitation_method,
                custom_voices,
                custom_voices_workers,
                copy.deepcopy(self.vci.model_config),
                avoid_overlap,
            ],
            {"valid_speakers": self.valid_speakers},
        ):
            audio_files, speakers_list = accelerate_segments(
                self.result_diarize,
                max_accelerate_audio,
                self.valid_speakers,
                acceleration_rate_regulation,
            )

            prog_disp(
                "Creating final translated video...",
                0.95,
                is_gui,
                progress=progress,
            )
            remove_files(dub_audio_file)
            create_translated_audio(
                self.result_diarize,
                audio_files,
                dub_audio_file,
                False,
                avoid_overlap,
            )

        # Voiceless track, change with file
        hash_base_audio_wav = get_hash(base_audio_wav)

        if not self.task_in_cache(
            "mix_aud",
            [
                mix_method_audio,
                volume_original_audio,
                volume_translated_audio,
                voiceless_track,
            ],
            {},
        ):
            # TYPE MIX AUDIO
            remove_files(mix_audio_file)
            command_volume_mix = f'ffmpeg -y -i {base_audio_wav} -i {dub_audio_file} -filter_complex "[0:0]volume={volume_original_audio}[a];[1:0]volume={volume_translated_audio}[b];[a][b]amix=inputs=2:duration=longest" -c:a libmp3lame {mix_audio_file}'
            command_background_mix = f'ffmpeg -i {base_audio_wav} -i {dub_audio_file} -filter_complex "[1:a]asplit=2[sc][mix];[0:a][sc]sidechaincompress=threshold=0.003:ratio=20[bg]; [bg][mix]amerge[final]" -map [final] {mix_audio_file}'
            if mix_method_audio == "Adjusting volumes and mixing audio":
                # volume mix
                run_command(command_volume_mix)
            else:
                try:
                    # background mix
                    run_command(command_background_mix)
                except Exception as error_mix:
                    # volume mix except
                    logger.error(str(error_mix))
                    run_command(command_volume_mix)

        hash_base_video_file = get_hash(base_video_file)

        if burn_subtitles_to_video:
            hashvideo_text = [
                hash_base_video_file,
                [seg["text"] for seg in self.result_diarize["segments"]],
            ]
            if self.burn_subs_id != hashvideo_text:
                try:
                    logger.info("Burn subtitles")
                    remove_files(vid_subs)
                    command = f"ffmpeg -i {base_video_file} -y -vf subtitles=sub_tra.srt -max_muxing_queue_size 9999 {vid_subs}"
                    run_command(command)
                    base_video_file = vid_subs
                    self.burn_subs_id = hashvideo_text
                except Exception as error:
                    logger.error(str(error))
            else:
                base_video_file = vid_subs

        if not self.task_in_cache(
            "output",
            [hash_base_video_file, hash_base_audio_wav, burn_subtitles_to_video],
            {},
        ):
            # Merge new audio + video
            remove_files(video_output_file)
            run_command(
                f"ffmpeg -i {base_video_file} -i {mix_audio_file} -c:v copy -c:a copy -map 0:v -map 1:a -shortest {video_output_file}"
            )

        output = media_out(
            media_file,
            TRANSLATE_AUDIO_TO,
            video_output_name,
            "mkv" if "mkv" in output_type else "mp4",
            file_obj=video_output_file,
            soft_subtitles=soft_subtitles_to_video,
            subtitle_files=output_format_subtitle,
        )
        msg_out = output[0] if isinstance(output, list) else output
        logger.info(f"Done: {msg_out}")

        return output
