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
        print(len(tokens[0].tokens))
        out_file.write(" ".join(map(str, tokens[0].tokens)) + "\n")
