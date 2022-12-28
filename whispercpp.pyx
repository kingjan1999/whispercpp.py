#!python
# cython: language_level=3
# distutils: language = c++
# distutils: sources= ./whisper.cpp/ggml.c ./whisper.cpp/whisper.cpp

import ffmpeg
import numpy as np
import requests
import os
from pathlib import Path

DEFAULT_MODELS_DIR = str(Path('~/ggml-models').expanduser())

cimport numpy as cnp

cdef int SAMPLE_RATE = 16000
cdef char* TEST_FILE = 'test.wav'
cdef char* DEFAULT_MODEL = 'tiny'
cdef char* LANGUAGE = b'fr'
cdef int N_THREADS = os.cpu_count()

MODELS = {
    'ggml-tiny.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
    'ggml-base.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
    'ggml-small.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
    'ggml-medium.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin',
    'ggml-large.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-large.bin',
}

def model_exists(model_dir, model):
    return os.path.exists(model_dir + "/" + model.decode())

def download_model(model_dir, model):
    print("Saving models to:", model_dir)
    if model_exists(model_dir, model):
        return
    print(f'Downloading {model}...')
    url = MODELS[model.decode()]
    r = requests.get(url, allow_redirects=True)
    with open(model_dir + "/" + model.decode(), 'wb') as f:
        f.write(r.content)


cdef audio_data load_audio(bytes file, int sr = SAMPLE_RATE):
    try:
        out = (
            ffmpeg.input(file, threads=0)
            .output(
                "-", format="s16le",
                acodec="pcm_s16le",
                ac=1, ar=sr
            )
            .run(
                cmd=["ffmpeg", "-nostdin"],
                capture_stdout=True,
                capture_stderr=True
            )
        )[0]
    except:
        raise RuntimeError(f"File '{file}' not found")

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = (
        np.frombuffer(out, np.int16)
        .flatten()
        .astype(np.float32)
    ) / pow(2, 15)

    cdef audio_data data;
    data.frames = &frames[0]
    data.n_frames = len(frames)

    return data

cdef whisper_full_params default_params() nogil:
    cdef whisper_full_params params = whisper_full_default_params(
        whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
    )
    params.print_realtime = True
    params.print_progress = True
    params.translate = False
    params.print_timestamps = False
    params.speed_up = False
    params.language = <const char *> LANGUAGE
    n_threads = N_THREADS
    return params


cdef class Whisper:
    cdef whisper_context * ctx
    cdef whisper_full_params params
    cdef int proc_count

    def __init__(self, model=DEFAULT_MODEL, model_dir=DEFAULT_MODELS_DIR, pb=None):
        model_fullname = f'ggml-{model}.bin'.encode('utf8')
        download_model(model_dir, model_fullname)
        cdef bytes model_b = model_dir.encode('utf8')  + b'/' + model_fullname
        self.ctx = whisper_init(model_b)
        self.params = default_params()
        self.proc_count = 1
        whisper_print_system_info()

    def __dealloc__(self):
        whisper_free(self.ctx)

    def set_params(self, *, language=None, translate=None, proc_count: int =None):
        if language:
            self.params.language = language
        if translate:
            self.params.translate = True
        elif translate is not None:
            self.params.translate = False
        if proc_count and proc_count > 0:
           self.proc_count = proc_count

    def transcribe(self, filename=TEST_FILE):
        print("Loading data..")
        cdef audio_data data = load_audio(<bytes>filename)
        print("Transcribing..")
        return whisper_full_parallel(self.ctx, self.params, data.frames, data.n_frames, self.proc_count)

    def extract_text(self, int res):
        print("Extracting text...")
        if res != 0:
            raise RuntimeError
        cdef int n_segments = whisper_full_n_segments(self.ctx)
        return [
            whisper_full_get_segment_text(self.ctx, i).decode() for i in range(n_segments)
        ]


