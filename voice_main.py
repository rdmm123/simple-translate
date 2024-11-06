from soni_translate.logging_setup import logger
import torch
import gc
import os
import shutil
import warnings
import edge_tts
import asyncio
from soni_translate.utils import (
    remove_directory_contents,
    create_directories,
)

warnings.filterwarnings("ignore")


class Config:
    def __init__(self, only_cpu=False):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        (
            self.x_pad,
            self.x_query,
            self.x_center,
            self.x_max
        ) = self.device_config(only_cpu)

    def device_config(self, only_cpu) -> tuple:
        if torch.cuda.is_available() and not only_cpu:
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info(
                    "16/10 Series GPUs and P40 excel "
                    "in single-precision tasks."
                )
                self.is_half = False
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
        elif torch.backends.mps.is_available() and not only_cpu:
            logger.info("Supported N-card not found, using MPS for inference")
            self.device = "mps"
        else:
            logger.info("No supported N-card found, using CPU for inference")
            self.device = "cpu"
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = os.cpu_count()

        if self.is_half:
            # 6GB VRAM configuration
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5GB VRAM configuration
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        logger.info(
            f"Config: Device is {self.device}, "
            f"half precision is {self.is_half}"
        )

        return x_pad, x_query, x_center, x_max


BASE_DOWNLOAD_LINK = "https://huggingface.co/r3gm/sonitranslate_voice_models/resolve/main/"
BASE_MODELS = [
    "hubert_base.pt",
    "rmvpe.pt"
]
BASE_DIR = "."


def load_hu_bert(config):
    from fairseq import checkpoint_utils
    from soni_translate.utils import download_manager

    for id_model in BASE_MODELS:
        download_manager(
            os.path.join(BASE_DOWNLOAD_LINK, id_model), BASE_DIR
        )

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

    return hubert_model


class ClassVoices:
    def __init__(self, only_cpu=False):
        self.model_config = {}
        self.config = None
        self.only_cpu = only_cpu

    def apply_conf(
        self,
        tag="base_model",
        file_model="",
        pitch_algo="pm",
        pitch_lvl=0,
        file_index="",
        index_influence=0.66,
        respiration_median_filtering=3,
        envelope_ratio=0.25,
        consonant_breath_protection=0.33,
        resample_sr=0,
        file_pitch_algo="",
    ):

        if not file_model:
            raise ValueError("Model not found")

        if file_index is None:
            file_index = ""

        if file_pitch_algo is None:
            file_pitch_algo = ""

        if not self.config:
            self.config = Config(self.only_cpu)
            self.hu_bert_model = None
            self.model_pitch_estimator = None

        self.model_config[tag] = {
            "file_model": file_model,
            "pitch_algo": pitch_algo,
            "pitch_lvl": pitch_lvl,  # no decimal
            "file_index": file_index,
            "index_influence": index_influence,
            "respiration_median_filtering": respiration_median_filtering,
            "envelope_ratio": envelope_ratio,
            "consonant_breath_protection": consonant_breath_protection,
            "resample_sr": resample_sr,
            "file_pitch_algo": file_pitch_algo,
        }
        return f"CONFIGURATION APPLIED FOR {tag}: {file_model}"


    def make_test(
        self,
        tts_text,
        tts_voice,
        model_path,
        index_path,
        transpose,
        f0_method,
    ):

        folder_test = "test"
        tag = "test_edge"
        tts_file = "test/test.wav"
        tts_edited = "test/test_edited.wav"

        create_directories(folder_test)
        remove_directory_contents(folder_test)

        if "SET_LIMIT" == os.getenv("DEMO"):
            if len(tts_text) > 60:
                tts_text = tts_text[:60]
                logger.warning("DEMO; limit to 60 characters")

        try:
            asyncio.run(edge_tts.Communicate(
                tts_text, "-".join(tts_voice.split('-')[:-1])
            ).save(tts_file))
        except Exception as e:
            raise ValueError(
                "No audio was received. Please change the "
                f"tts voice for {tts_voice}. Error: {str(e)}"
            )

        shutil.copy(tts_file, tts_edited)

        self.apply_conf(
            tag=tag,
            file_model=model_path,
            pitch_algo=f0_method,
            pitch_lvl=transpose,
            file_index=index_path,
            index_influence=0.66,
            respiration_median_filtering=3,
            envelope_ratio=0.25,
            consonant_breath_protection=0.33,
        )

        self(
            audio_files=tts_edited,
            tag_list=tag,
            overwrite=True
        )

        return tts_edited, tts_file

    def run_threads(self, threads):
        # Start threads
        for thread in threads:
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        gc.collect()
        torch.cuda.empty_cache()

    def unload_models(self):
        self.hu_bert_model = None
        self.model_pitch_estimator = None
        gc.collect()
        torch.cuda.empty_cache()

