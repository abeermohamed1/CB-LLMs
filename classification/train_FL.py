import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from utils import normalize, eos_pooling

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--cbl_path", type=str, default="mpnet_acs/imdb/roberta/no_backbone_model_best.pth.tar")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--saga_epoch", type=int, default=500)
parser.add_argument("--saga_batch_size", type=int, default=256)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)
# --- NEW ARGUMENT ---
parser.add_argument("--sample_size", type=int, default=-1, help="Number of training samples to use. -1 for all.")

# Corrected Dataset class for Hugging Face Dataset objects
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, idx):
        item = self.texts[idx]
        t = {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask'])
        }
        return t

    def __len__(self):
        return len(self.texts)

def build_loaders(texts, mode):
    dataset = ClassificationDataset(texts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader

def load_cbl_weights(model, path):
    print(f"Loading weights from {path}...")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        print("Detected checkpoint format (.pth.tar). Extracting state_dict...")
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("Detected raw weight format (.pt).")
        model.load_state_dict(checkpoint)
    return model

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    # Manual overrides for your specific run
    dataset_name = "imdb"
    backbone = 'roberta'
    cbl_name = 'no_backbone' # Forces logic to 'CBL only'
    acs = args.cbl_path.split("/")[0]
    
    print(f"Processing Dataset: {dataset_name} | Backbone: {backbone}")

    print("loading data...")
    train_dataset = load_dataset(dataset_name, split='train')
    
    # --- SAMPLING LOGIC ---
    if args.sample_size > 0:
        print(f"Sampling train dataset to {args.sample_size} examples (Seed: 42)...")
        train_dataset = train_dataset.shuffle(seed=42).select(range(args.sample_size))
    
    if dataset_name == 'SetFit/sst2':
        val_dataset = load_dataset(dataset_name, split='validation')
    test_dataset = load_dataset(dataset_name, split='test')

    if 'roberta' in backbone:
        tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')
    elif 'gpt2' in backbone:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise Exception("backbone should be roberta or gpt2")

    print("tokenizing...")
    # Keep as Dataset objects
    encoded_train_dataset = train_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset_name]], padding=True, truncation=True, max_length=args.max_length), batched=True)
    if dataset_name == 'SetFit/sst2':
        encoded_val_dataset = val_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset_name]], padding=True, truncation=True, max_length=args.max_length), batched=True)
    encoded_test_dataset = test_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset_name]], padding=True, truncation=True, max_length=args.max_length), batched=True)

    print("creating loaders...")
    train_loader = build_loaders(encoded_train_dataset, mode="valid")
    if dataset_name == 'SetFit/sst2':
        val_loader = build_loaders(encoded_val_dataset, mode="valid")
    test_loader = build_loaders(encoded_test_dataset, mode="test")

    concept_set = CFG.concept_set[dataset_name]

    # Model Init
    if 'roberta' in backbone:
        print("preparing CBL only...")
        cbl = CBL(len(concept_set), args.dropout).to(device)
        cbl = load_cbl_weights(cbl, args.cbl_path)
        cbl.eval()
        preLM = RobertaModel.from_pretrained('distilroberta-base').to(device)
        preLM.eval()

    # --- FEATURE EXTRACTION ---
    def extract_features(loader):
        features = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                f = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                f = f[:, 0, :] if 'roberta' in backbone else eos_pooling(f, batch["attention_mask"])
                f = cbl(f)
                features.append(f)
        return torch.cat(features, dim=0).detach().cpu()

    print("Extracting concept features...")
    train_c = extract_features(train_loader)
    if dataset_name == 'SetFit/sst2':
        val_c = extract_features(val_loader)
    test_c = extract_features(test_loader)

    # Normalization & Saving Mean/Std
    train_c, train_mean, train_std = normalize(train_c, d=0)
    train_c = F.relu(train_c)

    prefix = "./" + acs + "/" + dataset_name + "/" + backbone + "/"
    if not os.path.exists(prefix): os.makedirs(prefix)
    
    model_suffix = "cbl_final"
    torch.save(train_mean, prefix + 'train_mean_' + model_suffix)
    torch.save(train_std, prefix + 'train_std_' + model_suffix)

    # Apply same normalization to test
    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)

    # GLM SAGA Data Prep
    train_y = torch.LongTensor(encoded_train_dataset["label"])
    indexed_train_ds = IndexedTensorDataset(train_c, train_y)
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    
    test_y = torch.LongTensor(encoded_test_dataset["label"])
    test_loader = DataLoader(TensorDataset(test_c, test_y), batch_size=args.saga_batch_size, shuffle=False)

    print("Training final layer with GLM SAGA...")
    linear = torch.nn.Linear(train_c.shape[1], CFG.class_num[dataset_name])
    linear.weight.data.zero_(); linear.bias.data.zero_()
    
    output_proj = glm_saga(linear, indexed_train_loader, 0.05, args.saga_epoch, 0.99, k=10, 
                           test_loader=test_loader, do_zero=True, n_classes=CFG.class_num[dataset_name])

    # Save Final Weights
    W_g, b_g = output_proj['path'][-1]['weight'], output_proj['path'][-1]['bias']
    torch.save(W_g, prefix + 'W_g_' + model_suffix)
    torch.save(b_g, prefix + 'b_g_' + model_suffix)
    print(f"Final Test Acc: {output_proj['path'][-1]['metrics']['acc_test']}")
