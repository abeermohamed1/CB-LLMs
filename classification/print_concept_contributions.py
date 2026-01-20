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

parser = argparse.ArgumentParser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser.add_argument("--cbl_path", type=str, required=True, help="Path to your model_best.pth.tar")
parser.add_argument('--sparse', action=argparse.BooleanOptionalAction)
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

    dataset_name = 'imdb'
    backbone = 'roberta'
    target_dir = os.path.abspath(os.path.dirname(args.cbl_path)) 

    print(f"--- Contribution Analysis for: {dataset_name} ---")
    print(f"Target directory: {target_dir}")
    
    # Debug: List files in the directory to verify names
    print("Files found in directory:", os.listdir(target_dir))

    test_dataset = load_dataset(dataset_name, split='test')
    sample_size = 1000 
    indices = np.random.choice(range(len(test_dataset)), size=sample_size, replace=False)
    test_dataset = test_dataset.select(indices)
    
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

    # 1. Initialize Models
    cbl = CBL(num_concepts, args.dropout).to(device)
    cbl = load_cbl_weights(cbl, args.cbl_path)
    cbl.eval()
    preLM = RobertaModel.from_pretrained('distilroberta-base').to(device)
    preLM.eval()

    # 2. Feature Extraction
    print("Extracting features...")
    FL_test_features = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            f = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state[:, 0, :]
            f = cbl(f)
            FL_test_features.append(f)
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()

    # 3. Flexible Loading for Normalization & Weights
    def safe_load(filename, fallback_val):
        path = os.path.join(target_dir, filename)
        if os.path.exists(path):
            print(f"Loaded: {filename}")
            return torch.load(path, map_location='cpu')
        else:
            print(f"Warning: {filename} not found. Using fallback.")
            return fallback_val

    # Load Normalization
    train_mean = safe_load('train_mean_cbl_final', torch.zeros(num_concepts))
    train_std = safe_load('train_std_cbl_final', torch.ones(num_concepts))

    # Load Final Layer Weights (Weights are required, will fail if missing)
    w_name = "W_g_sparse_cbl_final" if args.sparse else "W_g_cbl_final"
    b_name = "b_g_sparse_cbl_final" if args.sparse else "b_g_cbl_final"
    
    W_g = torch.load(os.path.join(target_dir, w_name), map_location='cpu')
    b_g = torch.load(os.path.join(target_dir, b_name), map_location='cpu')

    # Apply scaling
    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)

    # 4. Calculation
    with torch.no_grad():
        logits = torch.matmul(test_c, W_g.t()) + b_g
        pred = torch.argmax(logits, dim=-1).numpy()
    
    label = np.array(encoded_test_dataset["label"])
    m = test_c.unsqueeze(1) * W_g.unsqueeze(0) 

    # 5. Save Report
    output_path = os.path.join(target_dir, 'Concept_contribution_report.txt')
    with open(output_path, 'w') as f:
        for i in range(m.shape[0]):
            raw_text = test_dataset[CFG.example_name[dataset_name]][i].replace('\n', ' ')
            f.write(f"TEXT: {raw_text}\n")
            
            vals, idxs = m[i][pred[i]].topk(min(5, num_concepts))
            f.write("Top Contributing Concepts:\n")
            for val, idx in zip(vals, idxs):
                if val > 0.0:
                    f.write(f" - {concept_set[idx]}: {val:.4f}\n")
            
            status = "CORRECT" if pred[i] == label[i] else f"INCORRECT (Label: {label[i]})"
            f.write(f"PREDICTION: {pred[i]} | STATUS: {status}\n")
            f.write("-" * 50 + "\n\n")

    print(f"Done! Report saved to {output_path}")
