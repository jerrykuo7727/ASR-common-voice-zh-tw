#!/bin/bash

python3_cmd=python3
export DATA_DIR=/home/jiazhi/Dataset/common-voice_zh-TW_43h_2019-06-12

. ./path.sh || exit 1
. ./cmd.sh || exit 1

nj=4
lm_order=3

. utils/parse_options.sh || exit 1
[[ $# -ge 1 ]] && { echo "Wrong arguments!"; exit 1; }

rm -rf data exp mfcc

echo
echo "---- Prepare basic data from CommonVoice_zh_TW ----"
echo

mkdir data data/train data/test data/local data/local/dict
$python3_cmd scripts/prepare_data.py

echo
echo "---- Prepare acoustic data ----"
echo

# Reverse utt2spk to spk2utt
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

echo
echo "---- Feature extraction ----"
echo

# MFCC
mfccdir=mfcc
steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/train \
                        exp/make_mfcc/train $mfccdir
steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/test \
                        exp/make_mfcc/test $mfccdir

# CMVN
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir
steps/compute_cmvn_stats.sh data/test exp/make_mfcc/test $mfccdir

echo
echo "---- Prepare language data ----"
echo

utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang

echo
echo "---- Create language model ----"
echo

# Check if SRILM was installed
loc=`which ngram-count`;
if [ -z $loc ]; then
    if uname -a | grep 64 >/dev/null; then
        sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
    else
        sdir=$KALDI_ROOT/tools/srilm/bin/i686
    fi
    if [ -f $sdir/ngram-count ]; then
        echo "Using SRILM language modelling tool from $sdir"
        export PATH=$PATH:$sdir
    else
        echo "SRILM toolkit is probably not installed. Instructions: tools/install_srilm.sh"
        exit 1
    fi
fi

# Train language model
local=data/local
mkdir $local/tmp
ngram-count -order $lm_order -write-vocab $local/tmp/vocab-full.txt \
            -wbdiscount -text $local/corpus.txt -lm $local/tmp/lm.arpa

# Make G.FST
lang=data/lang
arpa2fst --disambig-symbol=$local/tmp/lm.arpa $lang/G.fst

echo
echo "---- Monophone | Training ----"
echo

steps/train_mono.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono  || exit 1

echo
echo "---- Monophone | Decoding ----"
echo

utils/mkgraph.sh --mono data/lang exp/mono exp/mono/graph || exit 1
steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
                exp/mono/graph data/test exp/mono/decode
local/score.sh data/test data/lang exp/mono/decode/

echo
echo "---- Monophone | Alignment ----"
echo

steps/align_si.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali || exit 1

echo
echo "---- Triphone (1st): Delta + Delta-Delta | Training ----"
echo

steps/train_deltas.sh --cmd "$train_cmd" 2000 11000 data/train data/lang exp/mono_ali exp/tri1 || exit 1

echo
echo "---- Triphone (1st): Delta + Delta-Delta | Decoding ----"
echo

utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph || exit 1
steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
                exp/tri1/graph data/test exp/tri1/decode
local/score.sh data/test data/lang exp/tri1/decode

echo
echo "---- Triphone (1st): Delta + Delta-Delta | Alignment ----"
echo

steps/align_si.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/tri1 exp/tri1_ali || exit 1

echo 
echo "---- Triphone (2nd): LDA + MLLT | Training ----"
echo 

steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    2000 11000 data/train data/lang exp/tri1_ali exp/tri2

echo 
echo "---- Triphone (2nd): LDA + MLLT | Decoding ----"
echo 

utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph || exit 1
steps/decode.sh --nj $nj --cmd "$decode_cmd" exp/tri2/graph data/test exp/tri2/decode
local/score.sh data/test data/lang exp/tri2/decode

echo
echo "---- Script ended successfully ----"
echo