# Vendored UTMOS22 strong learner — MIT licence (tarepan/SpeechMOS v1.0.0).
# Copied verbatim from speechmos/utmos22/{fairseq_alt,strong/model}.py so we
# can instantiate UTMOS22Strong directly without any torch.hub machinery.
from .model import UTMOS22Strong

__all__ = ["UTMOS22Strong"]
