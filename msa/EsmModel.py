import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import EsmModel, EsmTokenizer
from tqdm import trange
from peft import LoraConfig, get_peft_model,PeftModel

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EsmExtraction(nn.Module):
    def __init__(self, esm_model, esm_tokenizer, device, max_length=512, hidden_dim=256, dropout_rate=0.1):
        super(EsmExtraction, self).__init__()
        self.esm_model = esm_model  # Use the pre-trained ESM model as the backbone (embedding extractor)
        self.esm_tokenizer = esm_tokenizer
        self.device = device
        self.max_length = max_length
        self.shared_fc = nn.Linear(1280, hidden_dim)  # Shared fully connected layer (optional, can remove if not needed)  ESM-1b produces 1280-d embeddings
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        inputs = self.esm_tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length) # Tokenize the input variant_sequence using the tokenizer
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.esm_model(**inputs)  # ESM model processes the tokenized input and outputs hidden states Extract embeddings from ESM model
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Use the <CLS> token (first token) embedding for classification Shape: (batch_size, hidden_size)
        shared_rep = torch.relu(self.shared_fc(cls_embedding))  # Pass through the shared fully connected layer (optional) Shape: (batch_size, hidden_dim)
        return shared_rep


class EsmClassificationHead(nn.Module):
    """
    Define the custom classification head
    Head for sentence-level classification tasks
    """
    def __init__(self, hidden_dim, dropout_rate, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EsmClassification(nn.Module):
    """
    Define the multitask model with ESM backbone
    """
    def __init__(self, hidden_dim, dropout_rate, num_classes_task_1):
        super(EsmClassification, self).__init__()
        self.task_1_head = EsmClassificationHead(hidden_dim, dropout_rate, num_classes_task_1) # Task-specific classification heads

    def forward(self, shared_rep):
        task_1_output = self.task_1_head(shared_rep)  #   Task-specific classification heads Shape: (batch_size, num_classes_task_1)
        return task_1_output


class MultitaskEsmModel(nn.Module):
    def __init__(self, extraction_net, classification_net, probability_of_true=0.4):
        super(MultitaskEsmModel, self).__init__()
        self.extraction_net = extraction_net  # ESM feature extraction network
        self.classification_net = classification_net  # Task-specific classification network
        self.probability_of_true = probability_of_true  # Masking probability for label corruption

    def forward(self, x):
        shared_rep = self.extraction_net(x) # Input `x` is the sequence data (protein variant_sequence, etc.) shape: (batch_size, feature_dim)
        task_1_output = self.classification_net(shared_rep) # Compute the task-specific predictions (before attention)
        return task_1_output


class EsmDataset(Dataset):

    def __init__(self, file_path, file_type='csv', am_pathogenicity=None):
        """
        Args:
            file_path (str): Path to the CSV or XLSX file.
            file_type (str): Type of the file ('csv' or 'xlsx').
        """
        self.file_path = file_path
        self.file_type = file_type.lower()
        if self.file_type == 'csv':
            self.data = pd.read_csv(self.file_path)
        elif self.file_type in ['xlsx', 'xls']:
            self.data = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file type. Please use 'csv' or 'xlsx'.")
        # if am_class is not None:
        #     self.data['am_class'] = self.data['am_class'].replace(am_class)
        self.variant_sequence = self.data['variant_sequence'].tolist()
        self.am_pathogenicity = self.data['am_pathogenicity'].tolist()

    def __len__(self):
        return len(self.variant_sequence)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {'sequence': str,'label': Any}
        """
        return {
            'variant_sequence': self.variant_sequence[idx],
            'am_pathogenicity': self.am_pathogenicity[idx]
        }



def get_model(device):
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    lora_config = LoraConfig(
        r=LORA_R,  # the dimension of the low-rank matrices
        lora_alpha=LORA_ALPHA,  # scaling factor for the weight matrices
        lora_dropout=LORA_DROPOUT,  # dropout probability of the LoRA layers
        bias="none",
        target_modules = ["query", "key", "value", "dense"]
    )

    hidden_dim = 128
    dropout_rate = 0.2
    num_classes_task_1 = 6
    probability_of_true = 0.9
    model_name = "facebook/esm2_t33_650M_UR50D" # Load the pre-trained ESM model from Hugging Face
    esm_model = EsmModel.from_pretrained(model_name, output_hidden_states=True)  # Load the trained model from the checkpoint (make sure you point to the right directory)
    esm_tokenizer = EsmTokenizer.from_pretrained(model_name)
    extraction_net = EsmExtraction(esm_model, esm_tokenizer, device, hidden_dim=128, dropout_rate=dropout_rate)     # Initialize model components
    classification_net = EsmClassification(hidden_dim, dropout_rate, num_classes_task_1)
    model = MultitaskEsmModel(
        extraction_net=extraction_net,  # ESM feature extraction model (already defined)
        classification_net=classification_net,  # Classification network (already defined)
        probability_of_true=probability_of_true
    )
    model = get_peft_model(model, lora_config).to(device)
    return model
