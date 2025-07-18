import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .base import TextEncoder

class BERTTextEncoder(TextEncoder):
    """BERT Text Encoder"""
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 feature_dim: int = 768, dropout_rate: float = 0.3):
        super().__init__(feature_dim)
        self.feature_dim = feature_dim
        self.bert_model = BertModel.from_pretrained(model_name)
        
        # Layer normalization and dropout
        self.ln = nn.LayerNorm(self.get_feature_dim())
        self.dropout = nn.Dropout(dropout_rate)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.get_feature_dim(), self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output
        embeddings = self.ln(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.projection(embeddings)
        return F.normalize(embeddings, dim=-1)

    def load_pretrained(self, model_path: str):
        """Load pretrained model from path"""
        self.bert_model.load_state_dict(torch.load(model_path))