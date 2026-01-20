import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel
from datasets import load_dataset
import config as CFG
from modules import CBL
from utils import normalize

# --- SETTINGS ---
parser = argparse.ArgumentParser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser.add_argument("--cbl_path", type=str, required=True, help="Path to your model_best.pth.tar")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)

# --- DATASET CLASS ---
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

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    # Automatic path detection based on your structure
    # Expected: checkpoints/mpnet_imdb/model_best.pth.tar
    acs = args.cbl_path.split("/")[0] # e.g. checkpoints
    dataset_name = "imdb" 
    backbone = "roberta"
    
    # Path where W_g_cbl_final and b_g_cbl_final live
    # Adjust this if your results are in a different folder
    results_prefix = f"./results/{acs}/{dataset_name}/{backbone}/"

    print(f"--- Loading Data: {dataset_name} ---")
    test_dataset = load_dataset(dataset_name, split='test')
    
    print("Tokenizing...")
    tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')
    encoded_test = test_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[dataset_name]], 
                            padding='max_length', 
                            truncation=True, 
                            max_length=args.max_length), 
        batched=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        ClassificationDataset(encoded_test), 
        batch_size=args.batch_size, 
        shuffle=False
    )

    concept_set = CFG.concept_set[dataset_name]
    num_concepts = len(concept_set)

    # --- MODEL LOADING ---
    print("Initializing Model Components...")
    
    # 1. Load the Backbone (Standard DistilRoBERTa)
    preLM = RobertaModel.from_pretrained('distilroberta-base').to(device)
    preLM.eval()

    # 2. Load the CBL layer (The 40 concepts)
    cbl = CBL(num_concepts, args.dropout).to(device)
    checkpoint = torch.load(args.cbl_path, map_location=device)
    
    # Extract state_dict if it's in a .pth.tar wrapper
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    cbl.load_state_dict(state_dict)
    cbl.eval()

    # 3. Load the Final SAGA Weights
    print(f"Loading weights from: {results_prefix}")
    #W_g = torch.load(os.path.join(results_prefix, "W_g_cbl_final"), map_location=device)
    #b_g = torch.load(os.path.join(results_prefix, "b_g_cbl_final"), map_location=device)

    # 3. Load the Final SAGA Weights
    print(f"Loading weights into memory...")
    # Path to your files
    w_path = './checkpoints/mpnet_imdb/W_g_cbl_final'
    b_path = './checkpoints/mpnet_imdb/b_g_cbl_final'

    # Load the actual Tensors from the disk
    W_g = torch.load(w_path, map_location=device)
    b_g = torch.load(b_path, map_location=device)

    # Safety check: if the file was saved as a dict, extract the tensor
    if isinstance(W_g, dict):
        W_g = W_g.get('weight', W_g.get('W_g', W_g))
    if isinstance(b_g, dict):
        b_g = b_g.get('bias', b_g.get('b_g', b_g))

    print(f"Weights loaded. Shape: {W_g.shape}")
    # --- INFERENCE ---
    print("Running Inference...")
    all_preds = []
    all_labels = np.array(encoded_test['label'])
    
    # To store concept activations for normalization if needed
    FL_test_features = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Step A: Get RoBERTa Features (CLS Token)
            features = preLM(input_ids=batch["input_ids"], 
                             attention_mask=batch["attention_mask"]).last_hidden_state[:, 0, :]
            
            # Step B: Pass through CBL to get Concept Activations
            concept_activations = cbl(features)
            
            # Apply ReLU as per CB-LLM architecture
            concept_activations = F.relu(concept_activations)
            
            # Step C: Final Prediction (Linear Layer)
            logits = torch.matmul(concept_activations, W_g.t()) + b_g
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.append(preds.cpu().numpy())

    # --- RESULTS ---
    all_preds = np.concatenate(all_preds)
    accuracy = (all_preds == all_labels).mean()
    
    print("-" * 30)
    print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
    print("-" * 30)
