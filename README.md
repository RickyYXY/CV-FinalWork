# CV Final Homework
本代码包含了我们CV大作业的全部实现内容，主要分为原版RAFT模型与训练，以及基于自监督方法训练的RAFT。具体细节详见我们的pdf文件。

## RAFT
RAFT的训练和推理详见README_RAFT.md。

## Self-Supervised RAFT
进行自监督训练的RAFT的训练方法和RAFT相似，只需将运行的脚本由train.py改为train_unsup.py即可，其余步骤和README_RAFT.md中一致。注意运行时迭代次数和学习率的设置需要和train_unsup.py中的默认值一致。
