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
    
def path2utt(row):
    ''' Convert audio path into utterance id. '''
    utt_id = row.path[:-4].split('_')[-1]
    prefix = row.spk_id
    return f'{prefix}_{utt_id}'

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

def zhuyin2phones(zhuyin, use_tone, sep):
    ''' Convert zhuyin of a charcter to phonemes. '''
    if any(is_chinese(c) for c in zhuyin):
        return zhuyin
    if zhuyin[0] == u'\u02d9':  # Neutral(fifth) tone 
        phones = sep.join([c for c in zhuyin][1:])
        tone = zhuyin[0]
    else:
        phones = sep.join([c for c in zhuyin][:-1])
        tone = zhuyin[-1]
    if use_tone:
        phones = f'{phones}{tone}'
    return phones

def fix_phones(phones, use_tone):
    ''' Fix broken phonemes. '''
    phones = phones.replace('一', 'ㄧ')
    if use_tone:
        phones = phones.replace('勳', 'ㄒ ㄩ ㄣ-')
        phones = phones.replace('艷', 'ㄧ ㄢˋ')
        phones = phones.replace('曬', 'ㄕ ㄞˋ')
    else:
        phones = phones.replace('勳', 'ㄒ ㄩ ㄣ')
        phones = phones.replace('艷', 'ㄧ ㄢ')
        phones = phones.replace('曬', 'ㄕ ㄞ')
    return phones

def word2phones(word, use_tone, sep=' '):
    ''' Convert a chinese word to zhuyin phonemes. '''
    if word == '曬':  # special case
        if use_tone:
            return 'ㄕ ㄞˋ'
        else:
            return 'ㄕ ㄞ'
    zhuyins = trans_sentense(word).split()
    phones = ' '.join([zhuyin2phones(z, use_tone, sep) for z in zhuyins])
    phones = fix_phones(phones, use_tone)
    return phones

def sent2phones(cut_sent, use_tone):
    ''' Convert a segmented sentence into phonemes. '''
    cut_phones, pos = [], 0
    sent = ''.join(cut_sent)
    phones = word2phones(sent, use_tone, sep='').split()
    for word in cut_sent:
        word_len = len(word)
        word_phones = ' '.join(phones[pos:pos + word_len])
        cut_phones.append(word_phones)
        pos += word_len
    return cut_phones


if __name__ == '__main__':

    # Configuration
    use_tone = False
    jieba.set_dictionary('scripts/dict.txt.big')
    jieba.initialize()

    # Read information of train/test set
    train_tsv = join(DATA_DIR, 'train.tsv')
    test_tsv = join(DATA_DIR, 'test.tsv')
    train_df = pd.read_csv(train_tsv, sep='\t')
    test_df = pd.read_csv(test_tsv, sep='\t')

    # Merge data excluding text with english characters
    train_df = train_df[train_df.sentence.apply(contains_no_eng)]
    test_df = test_df[test_df.sentence.apply(contains_no_eng)]
    full_df = pd.concat([train_df, test_df])


    ''' Prepare AM data '''
    print('Preparing AM data...\r', end='')

    # Prepare spk_id, gender and utt_id for all audios
    client_spk = full_df[['client_id']].drop_duplicates()
    client_spk['spk_id'] = range(1, 1 + len(client_spk))
    client_spk.spk_id = client_spk.spk_id.apply(lambda x: str(x).zfill(3))
    full_df = full_df.merge(client_spk)
    full_df.gender = full_df.gender.apply(kaldi_gender)
    full_df['utt_id'] = full_df.apply(path2utt, axis=1)

    # Fix unusual character and do text segmentation
    full_df.sentence = full_df.sentence.apply(fix_char)
    full_df.sentence = full_df.sentence.apply(jieba_cut)

    # Drop useless columns
    drop_columns = ['client_id', 'up_votes', 'down_votes', 'age', 'accent']
    full_df.drop(columns=drop_columns, inplace=True)

    # Split processed dataset to train/test set
    train_df = full_df[:-len(test_df)]
    test_df = full_df[-len(test_df):]

    # Sort train/test set by utt_id for kaldi-mfcc
    train_df = train_df.sort_values(by='utt_id')
    test_df = test_df.sort_values(by='utt_id')


    ''' Write AM data '''

    # Write acoustic data of train/test set
    for split, df in zip(['train', 'test'], [train_df, test_df]):
        
        # spk2gender
        spk2gender = df[['spk_id', 'gender']].drop_duplicates()
        spk2gender = spk2gender.sort_values(by='spk_id')
        with open(join('data', split, 'spk2gender'), 'w', encoding='UTF-8') as f:
            for _, row in spk2gender.iterrows():
                f.write(f'{row.spk_id} {row.gender}\n')
                
        # wav.scp
        with open(join('data', split, 'wav.scp'), 'w', encoding='UTF-8') as f:
            for _, row in df.iterrows():
                mp3_path = join(DATA_DIR, 'clips', row.path)
                f.write(f'{row.utt_id} sox {mp3_path} -t wav -r 16000 - |\n')
        
        # text
        with open(join('data', split, 'text'), 'w', encoding='UTF-8') as f:
            for _, row in df.iterrows():
                text = ' '.join(row.sentence)
                f.write(f'{row.utt_id} {text}\n')
  
        # utt2spk
        with open(join('data', split, 'utt2spk'), 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(f'{row.utt_id} {row.spk_id}\n')

    print('Preparing AM data... DONE')


    ''' Prepare LM data '''
    print('Preparing LM data...\r', end='')

    # Build corpus to train language model
    corpus = full_df.sentence.apply(' '.join).tolist()

    # Build lexicon with zhuyin package
    sents = full_df.sentence.tolist()
    sents_phones = [sent2phones(sent, use_tone) for sent in sents]
    lexicon = set()
    for sent, sent_phones in zip(sents, sents_phones):
        for i, zhuyins in enumerate(sent_phones):
            phonemes = []
            for zhuyin in zhuyins.split():
                if use_tone:
                    phones = [c for c in zhuyin[:-2]]
                    phones.append(zhuyin[-2:])
                else:
                    phones = [c for c in zhuyin]
                phones = ' '.join(phones)
                phonemes.append(phones)
            phonemes = ' '.join(phonemes)
            sent_phones[i] = phonemes
        lexicon.update(zip(sent, sent_phones))
        
    # Build phone set from lexicon
    phone_set = set()
    for _, phones in lexicon:
        phone_set.update(phones.split())

    
    ''' Write LM data '''

    # corpus.txt
    with open('data/local/corpus.txt', 'w', encoding='UTF-8') as f:
        for sent in corpus:
            f.write(f'{sent}\n')
            
    # lexicon.txt
    with open('data/local/dict/lexicon.txt', 'w', encoding='UTF-8') as f:
        f.write('!SIL sil\n')
        f.write('<UNK> spn\n')
        for word, phones in lexicon:
            f.write(f'{word} {phones}\n')
            
    # nonsilence_phones.txt
    with open('data/local/dict/nonsilence_phones.txt', 'w', encoding='UTF-8') as f:
        for phone in phone_set:
            f.write(f'{phone}\n')

    # silence_phones.txt
    with open('data/local/dict/silence_phones.txt', 'w', encoding='UTF-8') as f:
        f.write('sil\nspn\n')
        
    # optional_silence.txt
    with open('data/local/dict/optional_silence.txt', 'w', encoding='UTF-8') as f:
        f.write('sil\n')

    print('Preparing LM data... DONE')
    print('Program `prepare_data.py` ends succesfully.')