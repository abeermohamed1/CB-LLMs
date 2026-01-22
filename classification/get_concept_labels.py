import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
import config as CFG
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from utils import mean_pooling, decorate_dataset, decorate_concepts
import sys
import time

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--concept_text_sim_model", type=str, default="mpnet", help="mpnet, simcse or angle")

parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--sample_size", type=int, default=-1) # New argument for sampling

os.environ["TOKENIZERS_PARALLELISM"] = "false"
args = parser.parse_args(sys.argv[1:]) # Modified line

class SimDataset(torch.utils.data.Dataset):
    def __init__(self, encode_sim):
        self.encode_sim = encode_sim

    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.encode_sim.items()}

        return t

    def __len__(self):
        return len(self.encode_sim['input_ids'])

def build_sim_loaders(encode_sim):
    dataset = SimDataset(encode_sim)
    if args.concept_text_sim_model == 'angle':
        batch_size = 8
    else:
        batch_size = 32
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False)
    return dataloader


print("loading data...")
train_dataset = load_dataset(args.dataset, split='train')

# Modified: Handle IMDb test split as validation
if args.dataset == 'SetFit/sst2':
    val_dataset = load_dataset(args.dataset, split='validation')
"""elif args.dataset == 'imdb':
    val_dataset = load_dataset(args.dataset, split='test')"""

# Implement sampling logic
if args.sample_size > 0:
    print(f"Sampling {args.sample_size} examples from training and validation datasets...")
    train_dataset = train_dataset.shuffle(seed=42).select(range(args.sample_size))
    if args.dataset == 'SetFit/sst2' """or args.dataset == 'imdb'""":
        val_dataset = val_dataset.shuffle(seed=42).select(range(args.sample_size))

#print("training data len: ", len(train_dataset))

# Modified: Print val data len for both SetFit/sst2 and imdb
if args.dataset == 'SetFit/sst2' """or args.dataset == 'imdb'""":
    print("val data len: ", len(val_dataset))


concept_set = CFG.concept_set[args.dataset]
print("concept len: ", len(concept_set))

if args.concept_text_sim_model == 'mpnet':
    print("tokenizing and preparing mpnet")
    tokenizer_sim = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sim_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
    sim_model.eval()
    print("End: tokenizing and preparing mpnet")
elif args.concept_text_sim_model == 'simcse':
    tokenizer_sim = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    sim_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)
    sim_model.eval()
elif args.concept_text_sim_model == 'angle':
    print("tokenizing and preparing angle")
    config = PeftConfig.from_pretrained('SeanLee97/angle-llama-7b-nli-v2')
    tokenizer_sim = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    sim_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).bfloat16()
    sim_model = PeftModel.from_pretrained(sim_model, 'SeanLee97/angle-llama-7b-nli-v2')
    sim_model = sim_model.to(device)
    sim_model.eval()
    train_dataset = train_dataset.map(decorate_dataset, fn_kwargs={"d": args.dataset})
    # Modified: Apply decorate_dataset for imdb as well
    if args.dataset == 'SetFit/sst2' """or args.dataset == 'imdb'""":
        val_dataset = val_dataset.map(decorate_dataset, fn_kwargs={"d": args.dataset})
    concept_set = decorate_concepts(concept_set)
else:
    raise Exception("concept-text sim model should be mpnet, simcse or angle")

encoded_sim_train_dataset = train_dataset.map(
    lambda e: tokenizer_sim(e[CFG.example_name[args.dataset]], padding=True, truncation=True,
                            max_length=args.max_length), batched=True,
    batch_size=len(train_dataset))
encoded_sim_train_dataset = encoded_sim_train_dataset.remove_columns([CFG.example_name[args.dataset]])
#print("end of remove train column")
if args.dataset == 'SetFit/sst2':
    encoded_sim_train_dataset = encoded_sim_train_dataset.remove_columns(['label_text'])
if args.dataset == 'dbpedia_14':
    encoded_sim_train_dataset = encoded_sim_train_dataset.remove_columns(['title'])
encoded_sim_train_dataset = encoded_sim_train_dataset[:len(encoded_sim_train_dataset)]

# Modified: Load and process encoded_sim_val_dataset for both SetFit/sst2 and imdb
if args.dataset == 'SetFit/sst2' """or args.dataset == 'imdb'""":
    encoded_sim_val_dataset = val_dataset.map(
        lambda e: tokenizer_sim(e[CFG.example_name[args.dataset]], padding=True, truncation=True,
                                max_length=args.max_length), batched=True,
        batch_size=len(val_dataset))
    encoded_sim_val_dataset = encoded_sim_val_dataset.remove_columns([CFG.example_name[args.dataset]])
    if args.dataset == 'SetFit/sst2':
        encoded_sim_val_dataset = encoded_sim_val_dataset.remove_columns(['label_text'])
    if args.dataset == 'dbpedia_14': # Keep this specific to dbpedia_14
        encoded_sim_val_dataset = encoded_sim_val_dataset.remove_columns(['title'])
    encoded_sim_val_dataset = encoded_sim_val_dataset[:len(encoded_sim_val_dataset)]
#print("start of tokenizer_sim")
encoded_c = tokenizer_sim(concept_set, padding=True, truncation=True, max_length=args.max_length)
#print("end of tokenizer_sim")
train_sim_loader = build_sim_loaders(encoded_sim_train_dataset)
#print("end of train_sim_loader")
# Modified: Build val_sim_loader for both SetFit/sst2 and imdb
if args.dataset == 'SetFit/sst2' """or args.dataset == 'imdb'""":
    val_sim_loader = build_sim_loaders(encoded_sim_val_dataset)

#print("getting concept labels...")
encoded_c = {k: torch.tensor(v).to(device) for k, v in encoded_c.items()}
#print("start of encoded_c")
with torch.no_grad():
    #print("start of torch.no_grad(")
    if args.concept_text_sim_model == 'mpnet':
        concept_features = sim_model(input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"])
        print("start of mean_pooling")
        concept_features = mean_pooling(concept_features, encoded_c["attention_mask"])
    elif args.concept_text_sim_model == 'simcse':
        concept_features = sim_model(input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"], output_hidden_states=True, return_dict=True).pooler_output
    elif args.concept_text_sim_model == 'angle':
            concept_features = sim_model(output_hidden_states=True, input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"]).hidden_states[-1][:, -1].float()
    else:
        raise Exception("concept-text sim model should be mpnet, simcse or angle")
    #print("start of F.normalize")
    concept_features = F.normalize(concept_features, p=2, dim=1)
    #print("end of F.normalize")

start = time.time()
#print("start of time.time")
train_sim = []
for i, batch_sim in enumerate(train_sim_loader):
    #print("start of for i, batch_sim in enumerate")
    #print("print index =====",i)
    #print("print batch_sim =====",batch_sim)
    print("batch ", str(i), end="\r")
    batch_sim = {k: v.to(device) for k, v in batch_sim.items()}
    with torch.no_grad():
        if args.concept_text_sim_model == 'mpnet':
            #print("start of args.concept_text_sim_model == ")
            text_features = sim_model(input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"])
            #print("start of 1 text_features == ",text_features)
            text_features = mean_pooling(text_features, batch_sim["attention_mask"])
            #print("start of 2 text_features == ",text_features)
        elif args.concept_text_sim_model == 'simcse':
            text_features = sim_model(input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"], output_hidden_states=True, return_dict=True).pooler_output
        elif args.concept_text_sim_model == 'angle':
            text_features = sim_model(output_hidden_states=True, input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"]).hidden_states[-1][:, -1].float()
        else:
            raise Exception("concept-text sim model should be mpnet, simcse or angle")
        #print("start of F.normalize(")
        text_features = F.normalize(text_features, p=2, dim=1)
    #print("start of train_sim.append == ")
    train_sim.append(text_features @ concept_features.T)
#print("start of torch.cat(train_sim, dim=0)",train_sim)
train_similarity = torch.cat(train_sim, dim=0).cpu().detach().numpy()
#print("1 of train_similarity",train_similarity)
end = time.time()
#print("start of time of concept scoring")
#print("time of concept scoring:", (end-start)/3600, "hours")

# Modified: Generate val_similarity for both SetFit/sst2 and imdb
if args.dataset == 'SetFit/sst2' """or args.dataset == 'imdb'""":
    val_sim = []
    for batch_sim in val_sim_loader:
        batch_sim = {k: v.to(device) for k, v in batch_sim.items()}
        with torch.no_grad():
            if args.concept_text_sim_model == 'mpnet':
                text_features = sim_model(input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"])
                text_features = mean_pooling(text_features, batch_sim["attention_mask"])
            elif args.concept_text_sim_model == 'simcse':
                text_features = sim_model(input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"], output_hidden_states=True, return_dict=True).pooler_output
            elif args.concept_text_sim_model == 'angle':
                text_features = sim_model(output_hidden_states=True, input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"]).hidden_states[-1][:, -1].float()
            else:
                raise Exception("concept-text sim model should be mpnet or angle")
            text_features = F.normalize(text_features, p=2, dim=1)
        val_sim.append(text_features @ concept_features.T)
    val_similarity = torch.cat(val_sim, dim=0).cpu().detach().numpy()

d_name = args.dataset.replace('/', '_')
prefix = "./"
if args.concept_text_sim_model == 'mpnet':
    prefix += "mpnet_acs"
elif args.concept_text_sim_model == 'simcse':
    prefix += "simcse_acs"
elif args.concept_text_sim_model == 'angle':
    prefix += "angle_acs"
prefix += "/"
prefix += d_name
prefix += "/"
if not os.path.exists(prefix):
    os.makedirs(prefix)
print("start of concept_labels_train")
#print("start of prefix", prefix)
#print("start of train_similarity", train_similarity)
np.save(prefix + "concept_labels_train.npy", train_similarity)
print("end of concept_labels_train")
# Modified: Save val_similarity for both SetFit/sst2 and imdb
if args.dataset == 'SetFit/sst2' """or args.dataset == 'imdb'""":
    np.save(prefix + "concept_labels_val.npy", val_similarity)

