import modal
from src.main import translate_video

app = modal.App("simple-translate")

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

