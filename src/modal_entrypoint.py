import modal
from src.main import main as tr_main

app = modal.App("simple-translate")

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(['git', 'ffmpeg'])
    .pip_install_from_requirements('requirements.txt')
    .workdir('/app')
    .copy_local_file('input.mp4', '/app')
)

@app.function(image=image, gpu='T4')
def translate():
    tr_main()

@app.local_entrypoint()
def main():
    translate.remote()

