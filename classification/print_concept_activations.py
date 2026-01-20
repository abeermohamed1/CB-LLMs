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
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)

# --- CHANGE 1: HF Dataset Compatibility ---
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

# --- CHANGE 2: Smart Loading Function ---
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

    # Automatic path parsing
    acs = args.cbl_path.split("/")[0]
    dataset_name = args.cbl_path.split("/")[1] if 'sst2' not in args.cbl_path.split("/")[1] else args.cbl_path.split("/")[1].replace('_', '/')
    backbone = args.cbl_path.split("/")[2]
    cbl_name = args.cbl_path.split("/")[-1]
    dataset_name = 'imdb'
    backbone = 'roberta'
    cbl_name = 'no_backbone' # Forces logic to 'CBL only'

    print(f"Processing Activations for: {dataset_name}")

    print("loading data...")
    test_dataset = load_dataset(dataset_name, split='test')
    # --- NEW: SAMPLE THE DATASET ---
    # Change 1000 to whatever number of samples you want to test
    sample_size = 1000 
    indices = np.random.choice(range(len(test_dataset)), size=sample_size, replace=False)
    test_dataset = test_dataset.select(indices)
    print(f"Sampled {sample_size} examples for quick testing.")
    # -------------------------------
    
    print("tokenizing...")
    if 'roberta' in backbone:
        tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')
    elif 'gpt2' in backbone:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

    encoded_test_dataset = test_dataset.map(lambda e: tokenizer(e[CFG.example_name[dataset_name]], padding=True, truncation=True, max_length=args.max_length), batched=True)

    print("creating loader...")
    test_loader = build_loaders(encoded_test_dataset, mode="test")
    

    concept_set = CFG.concept_set[dataset_name]

    # --- CHANGE 3: Model Init with load_cbl_weights ---
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
    # (GPT2 logic follows same pattern if needed)

    print("extracting features...")
    FL_test_features = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if 'no_backbone' in cbl_name:
                f = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                f = f[:, 0, :] if 'roberta' in backbone else eos_pooling(f, batch["attention_mask"])
                f = cbl(f)
            else:
                f = backbone_cbl(batch)
            FL_test_features.append(f)
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()

    # --- CHANGE 4: Matching Prefix and Suffix logic ---
    prefix = "./" + acs + "/" + dataset_name.replace('/', '_') + "/" + backbone + "/"
    model_suffix = "_" + cbl_name.replace('.pt', '').replace('.pth.tar', '')
    
    train_mean = torch.load(prefix + 'train_mean' + model_suffix)
    train_std = torch.load(prefix + 'train_std' + model_suffix)

    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)

    label = encoded_test_dataset["label"]

    # Calculate error rate based on top concept activations
    error_rate = []
    for i in range(test_c.T.size(0)):
        error = 0
        total = 0
        value, s = test_c.T[i].topk(5)
        for j in range(5):
            if value[j] > 1.0:
                total += 1
                if get_labels(i, dataset_name) != label[s[j]]:
                    error += 1
        if total != 0:
            error_rate.append(error/total)
    
    if error_rate:
        print("avg error rate:", sum(error_rate) / len(error_rate))

    # Save Interpretability File
    output_file = prefix + 'Concept_activation' + model_suffix + '.txt'
    with open(output_file, 'w') as f:
        for i in range(test_c.T.size(0)):
            f.write(CFG.concept_set[dataset_name][i] + '\n')
            value, s = test_c.T[i].topk(5)
            # Write top 5 examples for this concept
            for j in range(5):
                if value[j] > 0.0:
                    text = test_dataset[CFG.example_name[dataset_name]][s[j]].replace('\n', ' ')
                    f.write(text + '\n')
                else:
                    f.write('\n')
            # Write activation values
            for j in range(5):
                if value[j] > 0.0:
                    f.write("{:.4f}\n".format(float(value[j])))
                else:
                    f.write('\n')
            f.write('\n')
    
    print(f"Concept activations saved to: {output_file}")
