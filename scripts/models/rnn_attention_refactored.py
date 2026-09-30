import torch
import torch.nn as nn
from scripts.models.selfattention import SelfAttention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNWithAttention(nn.Module):
    '''
    RNN 的所有模型默认开启 Bi-Directional，否则会造成 Attention 效果大打折扣
    这个模型主要与 Transformer 进行比较，因此实验设计点不需要考虑是否关掉双向机制
    支持架构: rnn, gru, lstm
    '''
    def __init__(
        self,
        vocab_size=8000,
        embedding_dim=64,
        hidden_dim=128,
        padding_idx=3,

        architecture="gru",
        bidirectional=True,       # 默认开启双向
        enable_attention = True
    ):
        super().__init__()

        # 全局 Attention 开关
        self.enable_attention = enable_attention

        # 全局 model 架构标志
        self.architecture = architecture

        # 全局双向标志 （1 单项，2 双向）
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 双向模式下，RNN 输出隐藏维度翻倍 (e.g. 128 * 2 = 256)
        self.rnn_output_dim = hidden_dim * self.num_directions

        # 1. 动态生成 Recurrent 模块
        if architecture not in ('rnn', 'gru', 'lstm'):
            raise ValueError(f"Unsurpported architecture: {architecture}, choose between 'rnn', 'gru' or 'lstm'")

        rnn_class = {
            "rnn": nn.RNN,
            "gru": nn.GRU,
            "lstm": nn.LSTM
        }.get(self.architecture)

        
        # 初始化 Embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )


        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )


        # Attention 和线性分类头都接收 RNN 的实际输出维度
        # 单向: hidden_dim
        # 双向: hidden_dim * 2
        self.attention = SelfAttention(self.rnn_output_dim)

        self.classifier = nn.Linear(self.rnn_output_dim, 1)


    def encode(self, input_ids, mask):

        # [B, L] -> [B, L, E]
        x = self.embedding(input_ids)

        # 数据预处理：将数据和实际序列长度打包，减少无效 Pad 计算
        lengths = mask.sum(dim=1).long()  # 保持在 GPU 上
        
        # 将长度强制约束在 [1, L]，防止出现全 Padding 极端情况导致的 0 长度报错
        lengths = lengths.clamp(min=1)

        # 打包 (Packing)：跳过 Padding 计算
        # 这一部分需要单独开 GPU Runtime 测试
        packed_x = pack_padded_sequence(
            x, 
            lengths=lengths.cpu(),  # 这里单独测试
            batch_first=True, 
            enforce_sorted=False
        )

        # 送入 RNN 计算
        packed_states, last_hidden = self.rnn(packed_x)

        if self.architecture == "lstm":
            # 对 LSTM 架构需要特殊处理 last hidden
            # last_hidden 是一个元组 (h_n, c_n)，仅需提取隐藏状态 h_n
            last_hidden = last_hidden[0]  # [num_directions, B, H]
        
        # 5. 解包 (Unpacking)：还原回 [B, L, H]
        states, _ = pad_packed_sequence(
            packed_states,
            batch_first=True, 
            total_length=input_ids.size(1) 
        )

        # 根据是否打开 Attention 进行特征聚合
        if self.enable_attention:
            # 开启 Attention 时，在 Unpacked 的 states 上做 Self-Attention 与 Pooling
            padding_mask = ~mask
            attended, weights = self.attention(states, padding_mask)

            # Masked Mean Pooling
            valid_mask = mask.unsqueeze(-1).float()
            context = (attended * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-9)

        else:
            # 不开启 Attention 时，直接使用 last_hidden 作为最终语义编码
            # 因为使用了 pack_padded_sequence 机制，因此 last_hidden 没有受到 Pad 污染
            weights = None

            # 还需要根据单向和双向进行特殊处理
            if self.bidirectional:
                # last_hidden[0] 为正向末尾状态，last_hidden[1] 为反向末尾状态，拼接后获得完整的编码
                context = torch.cat([last_hidden[0], last_hidden[1]], dim=-1)

            else:
                # 单向模式: [1, B, H] -> [B, H]
                context = last_hidden.squeeze(0)

        return context, weights


    def forward(self, input_1, input_2, mask_1, mask_2):

        # Candidate 1
        repr_1 = self.encode(input_1, mask_1)

        # Candidate 2
        repr_2 = self.encode(input_2, mask_2)

        context_1, weights_1 = repr_1
        context_2, weights_2 = repr_2

        weights = [weights_1, weights_2]

        # 共享分类器
        logit_1 = self.classifier(context_1)
        logit_2 = self.classifier(context_2)

        # [B, 1] + [B, 1]
        # -> [B, 2]
        logits = torch.cat(
            [logit_1, logit_2],
            dim=1
        )

        # 在不开 Attention 的情况下，weights 里面全是 None
        return logits, weights


    @torch.no_grad()
    def predict(self, input_1, input_2, mask_1, mask_2):
        '''
        不带反向传播的预测函数，不涉及梯度下降
        这里不会打开 model.eval()，防止在训练过程中忘记关闭，造成 Silent Failure
        进行性能测试时应手动调整模型的 train 和 eval 模式
        '''
        outputs = self.forward(input_1, input_2, mask_1, mask_2)

        return outputs