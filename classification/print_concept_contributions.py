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
parser.add_argument("--cbl_path", type=str, default="mpnet_acs/SetFit_sst2/roberta_cbm/cbl.pt")
parser.add_argument('--sparse', action=argparse.BooleanOptionalAction)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)

# --- Updated Dataset Class ---
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

# --- Smart Loading Function ---
def load_cbl_weights(model, path):
    print(f"Loading weights from {path}...")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model

def build_loaders(texts, mode):
    dataset = ClassificationDataset(texts)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    # Path setup
    acs = args.cbl_path.split("/")[0]
    dataset_name = "imdb" # Standardized for your current run
    backbone = 'roberta'
    cbl_name = args.cbl_path.split("/")[-1]
    dataset_name = 'imdb'
    backbone = 'roberta'
    cbl_name = 'no_backbone' # Forces logic to 'CBL only'

    print(f"Loading data for: {dataset_name}")
    test_dataset = load_dataset(dataset_name, split='test')
  
    # --- NEW: SAMPLE THE DATASET ---
    # Change 1000 to whatever number of samples you want to test
    sample_size = 1000 
    indices = np.random.choice(range(len(test_dataset)), size=sample_size, replace=False)
    test_dataset = test_dataset.select(indices)
    print(f"Sampled {sample_size} examples for quick testing.")
    # -------------------------------
    
    tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')
    encoded_test_dataset = test_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset_name]], padding=True, truncation=True, max_length=args.max_length), batched=True)

    test_loader = build_loaders(encoded_test_dataset, mode="test")
    concept_set = CFG.concept_set[dataset_name]

    # Model Initialization
    if 'no_backbone' in cbl_name:
        cbl = CBL(len(concept_set), args.dropout).to(device)
        cbl = load_cbl_weights(cbl, args.cbl_path)
        cbl.eval()
        preLM = RobertaModel.from_pretrained('distilroberta-base').to(device)
        preLM.eval()

    # Feature Extraction
    FL_test_features = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            f = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state[:, 0, :]
            f = cbl(f)
            FL_test_features.append(f)
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()

    # Apply Normalization from Training
    prefix = "./" + acs + "/" + dataset_name + "/" + backbone + "/"
    model_suffix = "_" + cbl_name.replace('.pt', '').replace('.pth.tar', '')
    
    train_mean = torch.load(prefix + 'train_mean' + model_suffix)
    train_std = torch.load(prefix + 'train_std' + model_suffix)
    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)

    # Load Final Layer Weights
    weight_type = "W_g_sparse" if args.sparse else "W_g"
    bias_type = "b_g_sparse" if args.sparse else "b_g"
    W_g = torch.load(prefix + weight_type + model_suffix)
    b_g = torch.load(prefix + bias_type + model_suffix)

    # Logic for Individual Contributions
    # m[i][j][k] = Activation of concept k for sample i * weight of concept k for class j
    with torch.no_grad():
        logits = torch.matmul(test_c, W_g.t()) + b_g
        pred = torch.argmax(logits, dim=-1).numpy()
    
    label = encoded_test_dataset["label"]
    correct_indices = np.where(pred == label)[0]
    mispred_indices = np.where(pred != label)[0]

    # Calculate contribution matrix
    m = test_c.unsqueeze(1) * W_g.unsqueeze(0) # [Batch, Classes, Concepts]

    # Interpretability Output
    output_path = prefix + 'Concept_contribution' + model_suffix + ('_sparse' if args.sparse else '') + '.txt'
    with open(output_path, 'w') as f:
        for i in range(m.size(0)):
            # Write the raw review text
            f.write(test_dataset[CFG.example_name[dataset_name]][i].replace('\n', ' ') + '\n')
            
            # Get top 5 concepts contributing to the Predicted Class
            vals, idxs = m[i][pred[i]].topk(5)
            
            # Write concept names
            for val, idx in zip(vals, idxs):
                if val > 0.0:
                    f.write(f"{concept_set[idx]}\n")
                else:
                    f.write('\n')
            
            # Write contribution values
            for val in vals:
                if val > 0.0:
                    f.write(f"{val:.4f}\n")
                else:
                    f.write('\n')
            
            # Conclusion line
            status = f"Pred: {pred[i]} (Correct)" if i not in mispred_indices else f"Pred: {pred[i]} (Incorrect, Label: {label[i]})"
            f.write(status + '\n\n')

    print(f"Contribution analysis saved to: {output_path}")
