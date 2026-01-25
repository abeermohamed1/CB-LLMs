import argparse
import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset, concatenate_datasets
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from utils import cos_sim_cubed, get_labels, eos_pooling
import time
import sys

# ==========================================
# 1. FIXED ARGUMENT PARSER
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="imdb")
parser.add_argument("--backbone", type=str, default="roberta")
parser.add_argument('--tune_cbl_only', action=argparse.BooleanOptionalAction, default=True)

# ADDED MISSING ARGUMENTS HERE:
parser.add_argument('--automatic_concept_correction', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--sample_size", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--labeling", type=str, default="mpnet")

parser.add_argument("--cbl_only_batch_size", type=int, default=64)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)

# ==========================================
# 2. DATASET CLASS (FIXED AttributeError)
# ==========================================
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encode_dict, similarity_matrix):
        self.encode_dict = encode_dict
        self.similarity_matrix = similarity_matrix

    def __getitem__(self, idx):
        # Accessing dictionary keys directly to avoid .items() error on HF objects
        t = {key: torch.tensor(self.encode_dict[key][idx]) for key in self.encode_dict.keys()}
        y = torch.FloatTensor(self.similarity_matrix[idx])
        return t, y

    def __len__(self):
        return len(self.encode_dict['input_ids'])

def build_loaders(encode_dict, s, mode, args):
    dataset = ClassificationDataset(encode_dict, s)
    batch_size = args.cbl_only_batch_size if args.tune_cbl_only else args.batch_size
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode=="train"), num_workers=args.num_workers)

# ==========================================
# 3. CHECKPOINT HELPERS
# ==========================================
def save_checkpoint(state, is_best, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'model_best.pth.tar'))

def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        print(f"==> Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], checkpoint['best_loss']
    return 0, float('inf')

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    args = parser.parse_args() # sys.argv[1:] is implied
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [Data Loading]
    print(f"Loading {args.dataset}...")
    train_dataset = load_dataset(args.dataset, split='train')
    
    if args.sample_size > 0:
        print(f"Sampling to {args.sample_size} examples...")
        train_dataset = train_dataset.shuffle(seed=42).select(range(args.sample_size))

    # [Tokenization]
    tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base') if args.backbone == 'roberta' else GPT2TokenizerFast.from_pretrained('gpt2')
    if args.backbone == 'gpt2': tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing...")
    encoded_hf = train_dataset.map(
        lambda e: tokenizer(e['text'], padding='max_length', truncation=True, max_length=args.max_length), 
        batched=True
    )

    # FIXED: Convert HF Dataset to Dict to avoid AttributeError
    encoded_train_dict = {
        'input_ids': encoded_hf['input_ids'],
        'attention_mask': encoded_hf['attention_mask']
    }

    train_similarity = np.load("/content/CB-LLMs/classification/mpnet_acs/imdb/concept_labels_train.npy")

    # [Model Setup]
    concept_count = len(CFG.concept_set[args.dataset])
    if args.backbone == 'roberta':
        if args.tune_cbl_only:
            model = CBL(concept_count, args.dropout).to(device)
            preLM = RobertaModel.from_pretrained('distilroberta-base').to(device)
            preLM.eval()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # [Checkpointing]
    ckpt_dir = f"./checkpoints/{args.dataset}"
    start_epoch, best_loss = load_checkpoint(os.path.join(ckpt_dir, 'checkpoint.pth.tar'), model, optimizer)

    train_loader = build_loaders(encoded_train_dict, train_similarity, "train", args)

    print("Starting training...")
    for e in range(start_epoch, 10):
        model.train()
        epoch_loss = 0
        for i, (batch_text, batch_sim) in enumerate(train_loader):
            batch_text = {k: v.to(device) for k, v in batch_text.items()}
            batch_sim = batch_sim.to(device)

            with torch.no_grad():
                LM_out = preLM(**batch_text).last_hidden_state[:, 0, :]
            
            preds = model(LM_out)
            loss = -cos_sim_cubed(preds, batch_sim)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        save_checkpoint({
            'epoch': e + 1,
            'state_dict': model.state_dict(),
            'best_loss': min(avg_loss, best_loss),
            'optimizer': optimizer.state_dict(),
        }, avg_loss < best_loss, ckpt_dir)
        print(f"Epoch {e+1} Complete. Loss: {avg_loss:.4f}")
