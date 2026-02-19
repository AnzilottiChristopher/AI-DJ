# Patch for importing AudioProcessor in transition_mixer.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from audio_processor import AudioProcessor
