import torch
import numpy as np
import soundfile as sf

from speechbrain.dataio.preprocess import AudioNormalizer 
from speechbrain.pretrained import EncoderClassifier 

class ExtractXvector:
    """x-vector抽出器

    SpeechBrain提供のVoxCelebで学習されたECAPA-TDNNから抽出されるx-vector.
    192次元.

    Attributes:
        属性の名前 (属性の型): 属性の説明
        属性の名前 (:obj:`属性の型`): 属性の説明.

    """
    def __init__(self):
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.classifier = EncoderClassifier.from_hparams( 
            source='speechbrain/spkrec-ecapa-voxceleb',
            run_opts={'device': self._device},
        ) 
        self.audio_norm = AudioNormalizer()

    def __call__(self, wav_path: str) -> np.ndarray:
        """

        Args:
            wav_path (str): x-vector抽出をおこなう音声のファイルパス

        Returns:
            np.ndarray: 抽出x-vector (192 dims)

        """
        wav, sr = sf.read(wav_path)
        # Amp Normalization -1 ~ 1
        amax = np.amax(np.absolute(wav))
        wav = wav.astype(np.float32) / amax
        # Freq Norm
        wav = self.audio_norm(torch.from_numpy(wav), sr).to(self._device)
        # x-vector Extraction (192 dims)
        embeds = self.classifier.encode_batch(wav).detach().cpu()[0][0]

        return embeds
