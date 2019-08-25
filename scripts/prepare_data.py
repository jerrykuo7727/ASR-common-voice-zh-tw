import os
from os.path import join
import pandas as pd
import jieba
from bopomofo.main import trans_sentense

DATA_DIR = os.environ['DATA_DIR']

def kaldi_gender(gender):
    ''' Alter CommonVoice gender into kaldi format. '''
    if gender == 'male':
        return 'm'
    elif gender == 'female':
        return 'f'
    else:
        return 'm'
    
def path2utt(path):
    ''' Convert audio path into utterance id. '''
    return path[:-4].split('_')[-1]

def is_chinese(char):
    ''' Check if character is chinese. '''
    return u'\u4e00' <= char <= u'\u9fff'

def fix_char(sent):
    ''' Fix unusual chinese characters. '''
    sent = sent.replace('内', '內')
    sent = sent.replace('爲', '為')
    sent = sent.replace('柺', '拐')
    sent = sent.replace('庄', '莊')
    sent = sent.replace('麽', '麼')
    sent = sent.replace('污', '汙')
    sent = sent.replace('値', '值')
    return sent
    
def jieba_cut(sent):
    ''' Chinese segmentation with punctuations removed. '''
    return [c for c in jieba.cut(sent) if is_chinese(c)]

def contains_no_eng(text):
    ''' Check if text contains no english. '''
    for char in text:
        if 'a' <= char <= 'z' or \
           'A' <= char <= 'Z' or \
           u'\uff21' <= char <= u'\uff3a' or \
           u'\uff41' <= char <= u'\uff5a':
            return False
    return True

def zhuyin2phones(zhuyin, use_tone=True):
    ''' Convert zhuyin of a charcter to phonemes. '''
    if zhuyin[0] == u'\u02d9':  # Neutral(fifth) tone 
        phones = ' '.join([c for c in zhuyin][1:])
        tone = zhuyin[0]
    else:
        phones = ' '.join([c for c in zhuyin][:-1])
        tone = zhuyin[-1]
    if use_tone:
        phones = f'{phones}{tone}'
    return phones

def fix_phones(phones):
    ''' Fix broken phonemes. '''
    phones = phones.replace('一', 'ㄧ')
    phones = phones.replace('勳', 'ㄒ ㄩ ㄣ-')
    phones = phones.replace('艷', 'ㄧ ㄢˋ')
    phones = phones.replace('曬', 'ㄕ ㄞˋ')
    return phones

def word2phones(word):
    ''' Convert a chinese word to zhuyin phonemes. '''
    zhuyins = trans_sentense(word).split()
    phones = ' '.join([zhuyin2phones(z) for z in zhuyins])
    phones = fix_phones(phones)
    return phones


if __name__ == '__main__':
    jieba.set_dictionary('dict.txt.big')
    jieba.initialize()

    # Read and merge information of train/test set
    train_tsv = join(DATA_DIR, 'train.tsv')
    test_tsv = join(DATA_DIR, 'test.tsv')
    train_df = pd.read_csv(train_tsv, sep='\t')
    test_df = pd.read_csv(test_tsv, sep='\t')

    # Exclude audios with english
    train_df = train_df[train_df.sentence.apply(contains_no_eng)]
    test_df = test_df[test_df.sentence.apply(contains_no_eng)]
    full_df = pd.concat([train_df, test_df])

    ''' Acoustic '''

    # Prepare gender and speaker id for all CommonVoice clients
    full_df.gender = full_df.gender.apply(kaldi_gender)
    client_spk = full_df[['client_id']].drop_duplicates()
    client_spk['spk_id'] = range(1, 1 + len(client_spk))
    full_df = full_df.merge(client_spk)

    # Prepare utterance id and text segmentation for all audios
    full_df['utt_id'] = full_df.path.apply(path2utt)
    full_df.sentence = full_df.sentence.apply(fix_char)
    full_df.sentence = full_df.sentence.apply(jieba_cut)

    # Drop useless columns
    drop_columns = ['client_id', 'up_votes', 'down_votes', 'age', 'accent']
    full_df.drop(columns=drop_columns, inplace=True)

    # Split processed dataset to train/test set
    train_df = full_df[:-len(test_df)]
    test_df = full_df[-len(test_df):]


    # Write acoustic data of train/test set
    for split, df in zip(['train', 'test'], [train_df, test_df]):
        
        # spk2gender
        spk2gender = df[['spk_id', 'gender']].drop_duplicates()
        with open(join('data', split, 'spk2gender'), 'w', encoding='UTF-8') as f:
            for _, row in spk2gender.iterrows():
                f.write(f'{row.spk_id} {row.gender}\n')
                
        # wav.scp
        with open(join('data', split, 'wav.scp'), 'w', encoding='UTF-8') as f:
            for _, row in df.iterrows():
                mp3_path = join(DATA_DIR, 'clips', row.path)
                f.write(f'{row.utt_id} sox {mp3_path} -t wav - |\n')
        
        # text
        with open(join('data', split, 'text'), 'w', encoding='UTF-8') as f:
            for _, row in df.iterrows():
                text = ' '.join(row.sentence)
                f.write(f'{row.utt_id} {text}\n')
  
        # utt2spk
        with open(join('data', split, 'utt2spk'), 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(f'{row.utt_id} {row.spk_id}\n')


    ''' Language '''

    # lexicon.txt
    phone_set = []
    with open('data/local/dict/lexicon.txt', 'w', encoding='UTF-8') as f:
        f.write('!SIL sil\n')
        f.write('<UNK> spn\n')
        for word in set(w for s in full_df.sentence.tolist() for w in s):
            phones = word2phones(word)
            phone_set += phones.split()
            f.write(f'{word} {phones}\n')

    # phone files
    phone_set = set(phone_set)
    with open('data/local/dict/nonsilence_phones.txt', 'w', encoding='UTF-8') as f:
        for phone in phone_set:
            f.write(f'{phone}\n')
            
    with open('data/local/dict/silence_phones.txt', 'w', encoding='UTF-8') as f:
        f.write('sil\nspn')
        
    with open('data/local/dict/optional_silence.txt', 'w', encoding='UTF-8') as f:
        f.write('sil')