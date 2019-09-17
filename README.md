# ASR-common-voice-zh-tw
Automatic Speech Recognition (ASR) system trained on [CommonVoice (zh-TW)](https://voice.mozilla.org/zh-TW/datasets) dataset with [Kaldi](https://github.com/kaldi-asr/kaldi) toolkit. 

Simply run `sh`[`run.sh`](https://github.com/jerrykuo7727/ASR-common-voice-zh-tw/blob/master/run.sh) to train and test the three models below:
1. Monophone
2. Triphone (1st pass): Delta + Delta-Delta
3. Triphone (2nd pass): LDA + MLLT

#
#### If you're familiar with the recipe of TIMIT, you just need to read these codes :)

[`scripts/prepare_data.py`](https://github.com/jerrykuo7727/ASR-common-voice-zh-tw/blob/master/scripts/prepare_data.py) :  Preprocess CommonVoice (zh-TW) for usage of Kaldi.  
[`scripts/prepare_data.ipynb`](https://github.com/jerrykuo7727/ASR-common-voice-zh-tw/blob/master/scripts/prepare_data.ipynb) :  Full details and explanations of the `prepare_data.py`.

#
See [`requirements.txt`](https://github.com/jerrykuo7727/ASR-common-voice-zh-tw/blob/master/requirements.txt) to check if any required package is not installed.
