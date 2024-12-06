from miditok import REMI, TokenizerConfig
from symusic import Score
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.pre_tokenizers import Split

from tqdm import tqdm
import os
from tokenizers import decoders, processors
import ipdb

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": True,
    "chord_maps": {"+" : (0, 4, 8), "/o7" : (0, 3, 6, 10), "7" : (0, 4, 7, 10), 
                   "M" : (0, 4, 7), "M7" : (0, 4, 7, 11), "m" : (0, 3, 7), 
                   "m7" : (0, 3, 7, 10), "o" : (0, 3, 6), "o7" : (0, 3, 6, 9),
                   "sus2" : (0, 2, 7), "sus4" : (0, 5, 7)},
    "chord_tokens_with_root_note" : True,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}
config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

vocab = tokenizer.vocab
all_tokens = " ".join(vocab.keys())
with open('allvoc', 'w') as out_file:
    out_file.write(all_tokens)

'''
def get_midi_files(folder_path):
    files = []
    for root, dirs, files_in_dir in os.walk(folder_path):
        for file in files_in_dir:
            if '.mid' in file : 
                files.append(os.path.join(root, file)) 
    return files
path = '/tmp2/b11902010/DMIR/Poprovise/dataset/pop1k7'
file_list = get_midi_files(path)
output_path = '/tmp2/b11902010/DMIR/Poprovise/dataset/pop1k7/midi.txt'
with open(output_path, 'w') as out_file:
    for file in tqdm(file_list):
        midi = Score(file)
        tokens = tokenizer(midi)
        out_file.write(" ".join(map(str, tokens[0].tokens)) + "\n")
'''
    
#tokenizer.save(out_path='../tokenizers', filename='miditokenizer.json')
'''
tokens = tokenizer(midi)  # calling the tokenizer will automatically detect MIDIs, paths and tokens
print(len(tokens[0].tokens))
converted_back_midi = tokenizer(tokens)  # PyTorch, Tensorflow and Numpy tensors are supported
print(converted_back_midi)
#tokenizer.save(out_path='../tokenizers', filename='tokenizer.json')
TokenizerConfig.save_to_json()
'''
# Define your tokenizer model (BPE in this case)
'''
tokenizer = Tokenizer(models.BPE())

# Pre-tokenizer: splits input into basic units (words, characters, or custom logic)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=5000, min_frequency=2)
'''
tokenizer = Tokenizer(models.WordLevel())
tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit(' ')
trainer = trainers.WordLevelTrainer(
    vocab_size=5000,      # You can set your vocabulary size here
    min_frequency=1,      # Set the minimum frequency threshold for words to be added to the vocabulary
)
files = ['allvoc']  
tokenizer.train(files, trainer)
tokenizer.save("/tmp2/b11902010/DMIR/Poprovise/tokenizers/tokenizer.json")
tokenizer = Tokenizer.from_file("/tmp2/b11902010/DMIR/Poprovise/tokenizers/tokenizer.json")

# Tokenize a new MIDI data (assuming you've converted it to tokens)
output = tokenizer.encode('Bar_None Position_0 Tempo_121.29 Position_16 Tempo_67.1 Position_24 Tempo_73.87 Pitch_69 Velocity_59 Duration_0.4.8')
print(output.tokens)


'''
TODO:
1. detokenizer (txt file to mid file)
'''