import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset
import evaluate
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from utils import normalize, eos_pooling

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--cbl_path", type=str, default="mpnet_acs/SetFit_sst2/roberta_cbm/cbl.pt")
parser.add_argument('--sparse', action=argparse.BooleanOptionalAction)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)

# --- CHANGE 1: Updated Dataset class for HF Compatibility ---
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, idx):
        item = self.texts[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask'])
        }

    def __len__(self):
        return len(self.texts)

# --- CHANGE 2: Smart Loading Function (Matches train_FL) ---
def load_cbl_weights(model, path):
    print(f"Loading weights from {path}...")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        print("Detected checkpoint format (.pth.tar).")
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("Detected raw weight format (.pt).")
        model.load_state_dict(checkpoint)
    return model

def build_loaders(texts, mode):
    dataset = ClassificationDataset(texts)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=False)
    return dataloader

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    # Manual overrides for consistency
    dataset_name = "imdb" # Change to 'SetFit/sst2' if needed
    backbone = 'roberta'
    
    acs = args.cbl_path.split("/")[0]
    cbl_name = args.cbl_path.split("/")[-1]
    
    print("loading data...")
    test_dataset = load_dataset(dataset_name, split='test')
    
    print("tokenizing...")
    if 'roberta' in backbone:
        tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')
    elif 'gpt2' in backbone:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

    # Keep as Dataset object for efficiency
    encoded_test_dataset = test_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset_name]], padding=True, truncation=True, max_length=args.max_length), batched=True)

    print("creating loader...")
    test_loader = build_loaders(encoded_test_dataset, mode="test")

    concept_set = CFG.concept_set[dataset_name]

    # --- CHANGE 3: Using Smart Loader for Model Init ---
    if 'roberta' in backbone:
        if 'no_backbone' in cbl_name:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            cbl = load_cbl_weights(cbl, args.cbl_path)
            cbl.eval()
            preLM = RobertaModel.from_pretrained('distilroberta-base').to(device)
            preLM.eval()
        else:
            print("preparing backbone(roberta)+CBL...")
            backbone_cbl = RobertaCBL(len(concept_set), args.dropout).to(device)
            backbone_cbl = load_cbl_weights(backbone_cbl, args.cbl_path)
            backbone_cbl.eval()
    
    # ... GPT2 logic would follow same load_cbl_weights pattern ...

    print("get concept features...")
    FL_test_features = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if 'no_backbone' in cbl_name:
                test_features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                test_features = test_features[:, 0, :] if 'roberta' in backbone else eos_pooling(test_features, batch["attention_mask"])
                test_features = cbl(test_features)
            else:
                test_features = backbone_cbl(batch)
            FL_test_features.append(test_features)
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()

    # --- CHANGE 4: Updated Naming to match train_FL suffix logic ---
    prefix = "./" + acs + "/" + dataset_name.replace('/', '_') + "/" + backbone + "/"
    model_suffix = "_" + cbl_name.replace('.pt', '').replace('.pth.tar', '')
    
    train_mean = torch.load(prefix + 'train_mean' + model_suffix)
    train_std = torch.load(prefix + 'train_std' + model_suffix)

    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)

    final = torch.nn.Linear(in_features=len(concept_set), out_features=CFG.class_num[dataset_name])
    
    # Determine which weights to load (Dense vs Sparse)
    weight_type = "W_g_sparse" if args.sparse else "W_g"
    bias_type = "b_g_sparse" if args.sparse else "b_g"
    
    W_g = torch.load(prefix + weight_type + model_suffix)
    b_g = torch.load(prefix + bias_type + model_suffix)
    
    final.load_state_dict({"weight": W_g, "bias": b_g})
    metric = evaluate.load("accuracy")
    
    with torch.no_grad():
        logits = final(test_c)
        pred = torch.argmax(logits, dim=-1).numpy()
    
    results = metric.compute(predictions=pred, references=encoded_test_dataset["label"])
    print(f"Results ({'Sparse' if args.sparse else 'Dense'}): {results}")
