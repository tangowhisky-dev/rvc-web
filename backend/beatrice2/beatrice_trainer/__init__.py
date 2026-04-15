# beatrice_trainer package — re-exports all public symbols so external callers
# that previously imported from beatrice_trainer continue to work unchanged.
#
# __main__.py is the training entry point (python3 -m beatrice_trainer).
# Import this package only when you need specific symbols; importing __main__
# is avoided at package load time to prevent the RuntimeWarning about __main__
# being found in sys.modules.

from .config import (
    PARAPHERNALIA_VERSION,
    dict_default_hparams,
    AttrDict,
    repo_root,
    is_notebook,
    prepare_training_configs,
    prepare_training_configs_for_experiment,
)
from .io import dump_params, dump_layer
from .layers import (
    CausalConv1d,
    WSConv1d,
    WSLinear,
    CrossAttention,
    ConvNeXtBlock,
    ConvNeXtStack,
    VectorQuantizer,
)
from .models import (
    FeatureExtractor,
    FeatureProjection,
    PhoneExtractor,
    extract_pitch_features,
    PitchEstimator,
    overlap_add,
    generate_noise,
    interp,
    linear_smoothing,
    dc_correction,
    nuttall,
    get_windowed_waveform,
    get_centroid,
    get_static_centroid,
    get_smoothed_power_spec,
    get_static_group_delay,
    get_coarse_aperiodicity,
    d4c_love_train,
    d4c_general_body,
    d4c,
    Vocoder,
    compute_loudness,
    slice_segments,
    ConverterNetwork,
    _normalize,
    SANConv2d,
    get_padding,
    DiscriminatorP,
    DiscriminatorR,
    MultiPeriodDiscriminator,
)
from .train import GradBalancer, prepare_training, _run_main
from .train.checkpoint import (
    get_compressed_optimizer_state_dict,
    get_decompressed_optimizer_state_dict,
)
from .data import WavDataset
# QualityTester deferred — imports torch.hub which downloads on first init.
# Import directly when needed: from .train.evaluation import QualityTester
