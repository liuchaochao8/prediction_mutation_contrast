
import torch.nn     as     nn
from   transformers import EsmModel, EsmTokenizer
from transformers import AutoModel, AutoTokenizer
"""
    后期加入lora
"""

#---嵌入模型
class EsmEmbedding(nn.Module):
    # def __init__(self, model_name, max_length):
    def __init__(self, model_name, max_length, device):
        super( EsmEmbedding, self ).__init__()
        
        self.model_name    = model_name                                                           # 模型名称
        self.esm_model     = EsmModel.from_pretrained( model_name, output_hidden_states=True ) # 加载模型

        self.esm_tokenizer = EsmTokenizer.from_pretrained( model_name )                        # 加载tokenizer

        self.max_length = max_length
    
    def forward(self, x, device):
        inputs  = self.esm_tokenizer(x, return_tensors="pt", padding=True,  # tokenizer
                                            truncation=True, max_length=self.max_length)
        inputs  = {key: value.to(device) for key, value in inputs.items()}  # 加入deveci
        outputs = self.esm_model( **inputs )                                # 获取输出
        esm_emb = outputs.last_hidden_state[:, 0, :]                        # 获取embedding
        return esm_emb