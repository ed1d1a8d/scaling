from typing import Type

from src.pretrain.models.base import BaseEmbedderConfig
from src.pretrain.models.vision import laion_clip, msft_beit, openai_clip


def get_embedder_index() -> dict[str, Type[BaseEmbedderConfig]]:
    return {
        "laion_clip": laion_clip.LaionClipConfig,
        "msft_beit": msft_beit.MsftBeitConfig,
        "openai_clip": openai_clip.OpenaiClipConfig,
    }
