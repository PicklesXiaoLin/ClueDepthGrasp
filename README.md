# ClueDepthGrasp
This code is used to reproduce the model in ClueDepth Grasp article.

本代码仅用于描述了“ClueDepth Grasp Leveraging positional clues of depth for completing depth of transparent objects”中出现的网络结构

# 内容
1 model.py 中包含 DenseNet+Transformer 的结构
2 model.py 中包含 Cluedepth 输入到网络中训练，该 Depth image 生成代码需结合 Normal image
3 model.py 中包含多模态U-Net模块的流程，将四种模态数据整合
4 train.py 中包含 Sobel-Gan 生成式对抗网络的详细参数

# 注意事项
1 数据集为cleargrasp，数据集来自 https://sites.google.com/view/cleargrasp。总共70G
