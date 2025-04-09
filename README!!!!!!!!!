pro_mut_contrast.py---------使用ESM2预测
pro_mut_contrast_esm1v.py--------使用ESM1V预测
pro_mut_contrast_lora.py---------ESM2+LoRa

数据集使用原始数据的前300行训练（完整数据太大，服务器显存太小，batchsize如果很小，误差大。batchsize大跑不起来）


完整数据集问一下廖青老师，AlphaMissense数据集

\msa\main.py--------使用MSA Transformer 调整嵌入方式 而且只针对一个蛋白。目前没用
                    后续使用GEMME方法扩充数据集（看论文：GEMME）
                    使用GEMME找到更多相似蛋白，


如果只是用一种蛋白质做：数据在：prediction_mutation_contrast代码/data/R1R8_data.csv

对比学习的loss函数，还在和红文老师讨论。
        loss目前存在的问题：多种蛋白一起训练时，用cos距离算误差有问题：不同蛋白的距离不同，训练不好。
        （一种蛋白可以用cos距离）
