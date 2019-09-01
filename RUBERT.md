# Convert ruBERT Tensorflow model to PyTorch

Download and unpack ruBERT to, e.g., ~/data/models/rubert_cased_L-12_H-768_a-12_v1/

then in your projects directory do the following to set up what you need for the conversion

```
git clone <pytorch-transformers>
cd <pytorch-transformers>
virtualenv -p python3 ~/venv/pytorch-transformers
source ~/venv/pytorch-transformers/bin/activate
pip3 install -r requirements.txt
pip3 install tensorflow
```

Convert the Tensorflow ruBERT model to PyTorch with this:

```
$ python3 -m pytorch_transformers.convert_tf_checkpoint_to_pytorch --tf_checkpoint_path /Users/fredriko/data/models/rubert/rubert_cased_L-12_H-768_A-12_v1/bert_model.ckpt.index --bert_config_file /Users/fredriko/data/models/rubert/rubert_cased_L-12_H-768_A-12_v1/bert_config.json --pytorch_dump_path /Users/fredriko/data/models/rubert/rubert_cased_L-12_H-768_A-12_v1/rubert_pytorch.bin
```

Make sure to rename `bert_config.json` to `config.json`. After the conversion, the required files are:

```
config.json
pytorch_model.bin
vocab.txt
```

Use the following to load the converted model in pytorch-transformers:

```
import torch
from pytorch_transformers import *

sample_text = "Рад познакомиться с вами."
my_model_dir = "/Users/fredriko/Dropbox/data/models/rubert/pytorch-rubert/"

tokenizer = BertTokenizer.from_pretrained(my_model_dir)
model = BertModel.from_pretrained(my_model_dir, output_hidden_states=True)

input_ids = torch.tensor([tokenizer.encode(sample_text, add_special_tokens=True)])
print(f"Input ids: {input_ids}")
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]
    print(f"Shape of last hidden states: {last_hidden_states.shape}")
    print(last_hidden_states)

```