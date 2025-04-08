import wave
from functools import lru_cache
from typing import Tuple

import numpy as np
import sherpa_onnx
from huggingface_hub import hf_hub_download


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


@lru_cache(maxsize=30)
def get_file(
    repo_id: str,
    filename: str,
    subfolder: str = ".",
) -> str:
    nn_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        endpoint="https://hf-mirror.com"
    )
    return nn_model_filename


def get_speaker_segmentation_model(repo_id) -> str:
    assert repo_id in (
        "pyannote/segmentation-3.0",
        "Revai/reverb-diarization-v1",
    )

    if repo_id == "pyannote/segmentation-3.0":
        return get_file(
            repo_id="csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
            filename="model.onnx",
        )
    elif repo_id == "Revai/reverb-diarization-v1":
        return get_file(
            repo_id="csukuangfj/sherpa-onnx-reverb-diarization-v1",
            filename="model.onnx",
        )


def get_speaker_embedding_model(model_name) -> str:
    assert (
        model_name
        in three_d_speaker_embedding_models
    )
    model_name = model_name.split("|")[0]

    return get_file(
        repo_id="csukuangfj/speaker-embedding-models",
        filename=model_name,
    )


def get_speaker_diarization(
    segmentation_model: str, embedding_model: str, num_clusters: int, threshold: float
):
    segmentation = get_speaker_segmentation_model(segmentation_model)
    embedding = get_speaker_embedding_model(embedding_model)
    
    # 简单检查是否有CUDA设备可用
    provider = 'cpu'
    try:
        import onnxruntime
        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            provider = 'cuda'
            print("找到CUDA设备，使用GPU")
        else:
            print("未找到CUDA设备，使用CPU")
    except Exception as e:
        print(f"检测GPU设备时出错: {str(e)}，使用CPU")

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation
            ),
            debug=False,
            provider=provider,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding,
            debug=False,
            provider=provider,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )

    print("config", config)

    if not config.validate():
        raise RuntimeError(
            "请检查您的配置并确保所有必需的文件都存在"
        )

    return sherpa_onnx.OfflineSpeakerDiarization(config)


speaker_segmentation_models = [
    "pyannote/segmentation-3.0",
    "Revai/reverb-diarization-v1",
]

three_d_speaker_embedding_models = [
    "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx|68.1MB",
    "3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx|111MB",
    "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx|27MB",
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx|27MB",
]

embedding2models = {
    "3D-Speaker": three_d_speaker_embedding_models,
}
