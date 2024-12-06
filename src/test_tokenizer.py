from transformers import AutoModel, PreTrainedTokenizerFast
from symusic import Score
from transformers import AutoTokenizer
import miditok
from miditok import REMI
from pathlib import Path
import torch
# Paths
local_model_path = "/tmp2/b11902010/DMIR/Poprovise/src/Diffusion-LM/bert-dmir"
new_tokenizer_path = "/tmp2/b11902010/DMIR/Poprovise/tokenizers/tokenizer.json"

bert_model = AutoModel.from_pretrained(local_model_path)
tokenizer = REMI(params=Path(new_tokenizer_path))
midi = Score("/tmp2/b11902010/DMIR/Poprovise/dataset/midi_raw/001/001_4.mid")
encoded_input = tokenizer(midi)
print(encoded_input[0].ids)

# Forward pass through the model with the new tokenized input
input = torch.tensor([encoded_input[0].ids])
output = bert_model(input)
print(output)

#new_tokenizer.save_pretrained(local_model_path)

# Verify the saved tokenizer
print(f"New tokenizer saved to: {local_model_path}")
