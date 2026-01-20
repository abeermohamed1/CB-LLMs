import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
from datasets import load_dataset
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from utils import normalize, get_labels, eos_pooling

parser = argparse.ArgumentParser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser.add_argument("--cbl_path", type=str, required=True, help="Path to your model_best.pth.tar")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)

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

def load_cbl_weights(model, path):
    print(f"Loading weights from {path}...")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    # --- HARDCODED SETTINGS FOR YOUR RUN ---
    dataset_name = 'imdb'
    backbone = 'roberta'
    
    # Path extraction: get the folder where model_best lives
    # This ensures we find the normalization files in the same place
    target_dir = os.path.dirname(args.cbl_path) 
    print(f"Target Directory detected as: {target_dir}")

    print(f"Processing Activations for: {dataset_name}")
    test_dataset = load_dataset(dataset_name, split='test')
    
    # Sample for quick processing
    sample_size = 1000 
    indices = np.random.choice(range(len(test_dataset)), size=sample_size, replace=False)
    test_dataset = test_dataset.select(indices)
    print(f"Sampled {sample_size} examples for quick testing.")
    
    print("Tokenizing...")
    tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')
    encoded_test_dataset = test_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[dataset_name]], 
                            padding='max_length', 
                            truncation=True, 
                            max_length=args.max_length), 
        batched=True
    )

    test_loader = torch.utils.data.DataLoader(
        ClassificationDataset(encoded_test_dataset), 
        batch_size=args.batch_size, 
        shuffle=False
    )

    concept_set = CFG.concept_set[dataset_name]
    num_concepts = len(concept_set)

    # --- MODEL INITIALIZATION ---
    print("Preparing CBL and Frozen Backbone...")
    cbl = CBL(num_concepts, args.dropout).to(device)
    cbl = load_cbl_weights(cbl, args.cbl_path)
    cbl.eval()
    
    preLM = RobertaModel.from_pretrained('distilroberta-base').to(device)
    preLM.eval()

    # --- FEATURE EXTRACTION ---
    print("Extracting concept activations...")
    FL_test_features = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            f = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state[:, 0, :]
            f = cbl(f)
            FL_test_features.append(f)
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()

    # --- FIXED NORMALIZATION LOADING ---
    print(f"Searching for normalization files in {target_dir}...")
    try:
        # Looking for the specific files you generated
        train_mean = torch.load(os.path.join(target_dir, 'train_mean_cbl_final'), map_location='cpu')
        train_std = torch.load(os.path.join(target_dir, 'train_std_cbl_final'), map_location='cpu')
        print("Successfully loaded normalization files.")
    except FileNotFoundError:
        print("!!! Warning: Normalization files not found. Using default (0,1). Results may be inaccurate.")
        train_mean = torch.zeros(num_concepts)
        train_std = torch.ones(num_concepts)

    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)

    # --- GENERATE OUTPUT FILE ---
    output_file = os.path.join(target_dir, 'Concept_activation_report.txt')
    print(f"Writing report to: {output_file}")
    
    # We transpose to iterate through concepts (test_c.T)
    with open(output_file, 'w') as f:
        for i in range(num_concepts):
            f.write(f"CONCEPT: {concept_set[i]}\n")
            f.write("-" * 20 + "\n")
            
            # Get top 5 samples that activated this concept
            values, idxs = test_c[:, i].topk(min(5, sample_size))
            
            # Write texts
            for val, idx in zip(values, idxs):
                if val > 0.0:
                    raw_text = test_dataset[CFG.example_name[dataset_name]][int(idx)].replace('\n', ' ')
                    f.write(f"[Act: {val:.4f}] {raw_text}\n")
                else:
                    f.write("[No Activation]\n")
            f.write("\n")
    
    print("Done!")
