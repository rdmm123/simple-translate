import modal
from src.main import translate_video

app = modal.App("simple-translate")

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(['git', 'ffmpeg'])
    .pip_install_from_requirements('requirements.txt')
    .workdir('/app')
    .copy_local_file('input.mp4', '/app')
)

@app.function(image=image, gpu='T4', timeout=600)
def translate(path: str):
    translate_video(path)

@app.local_entrypoint()
def main():
    translate.remote('input.mp4')

