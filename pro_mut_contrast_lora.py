#
import torch
import torch.nn    as nn
import torch.optim as optim
                
# from model.esm_embedding         import EsmEmbedding
from model.esm_embedding         import EsmEmbedding

from dataset.mut_protein_dataset import MutaProDataWrapper
from losses                      import ProMutaContrastLoss
import torch, gc
from peft import LoraConfig, get_peft_model,PeftModel

# tokenizer后的示例：：最长50
"""
{
    'input_ids': tensor([
        [ 101, 1234, 5678, ..., 0, 0, 0],  # 序列 1 的 token IDs，填充到 max_length=50
        [ 101, 2345, 6789, ..., 0, 0, 0],  # 序列 2 的 token IDs，填充到 max_length=50
        # ... 共 128 个序列
    ]),  # 形状: (128, 50)

    'attention_mask': tensor([
        [1, 1, 1, ..., 0, 0, 0],  # 序列 1 的 attention mask
        [1, 1, 1, ..., 0, 0, 0],  # 序列 2 的 attention mask
        # ... 共 128 个序列
    ])   # 形状: (128, 50)
}

"""

# self.esm_model(**inputs)的输出
"""
last_hidden_state
"""

# 原序列由ESM嵌入，变异序列由新模型嵌入
# 回归损失？
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# import torch
# torch.cuda.set_device("3")


            
def train( model, dataloader, criterion, optimizer ,batch_size,epoch):

    # model.half() # 减少精度，后续可优化
    model = nn.DataParallel(model)  # 将模型分布在多个GPU上
    device = torch.device("cuda")  # 指定gpu2为主GPU
    model.to(device)

    model.train()
    f = open("loss_train.txt", "a+")
    avg_train_loss = 0
    for batch_idx, (wild_seq, muta_seq, prob_muta) in enumerate( dataloader ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        # print(device)
        # print("TO DEVICE......" )


        #----获取嵌入
        # print("ENBEDDING......")
        wild_emb = model( wild_seq,  device) # 参考序列嵌入
        muta_emb = model( muta_seq ,device) # 变异序列嵌入
        
        #----损失计算
        # print("LOSSING......")
        loss = criterion( wild_emb, muta_emb, prob_muta.to(device), batch_size )
        avg_train_loss += loss
        # print(loss)
        # print("LOSS BACKWARD......\n")
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        gc.collect()
        torch.cuda.empty_cache()


    avg_train_loss /= len(wild_seq)
    print(avg_train_loss.item())
    print("写入txt")
    f.write(str(avg_train_loss.item()) + '\n')
    f.close()


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"Trainable model parameters: {trainable_model_params}.\n"
          f"All model parameters: {all_model_params}.\n"
          f"Percentage: {round(trainable_model_params / all_model_params * 100, 3)}%")



def main():
    #---1、参数设置
    # 使用超小数据集
    wild_data_path = "data/little_wild_protein_50.csv" # 参考序列数据路径
    muta_data_path = "data/little_muta_protein_300.csv" # 变异序列数据路径
    # 使用LoRA
    batch_size = 16
    num_epochs = 100
    max_length = 128
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.2
    lora_config = LoraConfig(
        r=LORA_R,  # the dimension of the low-rank matrices
        lora_alpha=LORA_ALPHA,  # scaling factor for the weight matrices
        lora_dropout=LORA_DROPOUT,  # dropout probability of the LoRA layers
        bias="none",
        target_modules=["query", "key", "value", "dense"]
    )


    #---2、获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    #---3、导入获取Esm嵌入的模型：：后期可能使用lora模型
    print("MODEL BUILDING......")
    model_name = "facebook/esm2_t33_650M_UR50D"       # 模型名称
    local_model_path = "model_get_esm/"
                                      # 序列最大长度
    Esm_model  = EsmEmbedding( model_name, max_length, device )
    # Esm_model  = Esm_model.to(device)

    #---3.5、lora模型
    Esm_model = get_peft_model(Esm_model, lora_config).to(device)
    print_number_of_trainable_model_parameters(Esm_model)


    #---4、数据集加载：：还未添加裁剪
    print("DATA LOADING......")
    dataset    = MutaProDataWrapper( wild_data_path, muta_data_path, batch_size )
    dataloader = dataset.get_data_loader( dataset, batch_size ) # 是否要to(device)

    #---5、损失函数构造：：
    temperature = 0.07
    criterion   = ProMutaContrastLoss( temperature=temperature )
    criterion   = criterion.to(device)

    #---6、获取优化器
    optimizer = optim.Adam(Esm_model.parameters(), lr=1e-2)
    print("DATA TRAINING......")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n")
        train( Esm_model, dataloader, criterion, optimizer ,batch_size,epoch)
    
    
    

main()
    
