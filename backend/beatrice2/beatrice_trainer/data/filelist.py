# Thin wrappers — actual implementations are nested in prepare_training().
# Import this module for type annotations; real call sites use the nested
# functions directly within prepare_training().
from pathlib import Path
from typing import Optional
from random import Random

AUDIO_FILE_SUFFIXES = {'.wav', '.aif', '.aiff', '.fla', '.flac',
                        '.oga', '.ogg', '.opus', '.mp3', '.m4a', '.aac'}
