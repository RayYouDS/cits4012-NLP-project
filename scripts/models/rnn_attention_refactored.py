import torch
import torch.nn as nn
from scripts.models.selfattention import SelfAttention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNWithAttention(nn.Module):
    """
    A configurable recurrent encoder with optional self-attention.

    Supported architectures: RNN, GRU, and LSTM.

    Bidirectional encoding is enabled by default. For comparisons with
    Transformer models, the experimental design focuses on the recurrent
    architecture rather than removing bidirectionality, which may reduce
    the effectiveness of the attention mechanism.
    """
    def __init__(
        self,
        vocab_size=8000,
        embedding_dim=64,
        hidden_dim=128,
        padding_idx=3,

        architecture="gru",
        bidirectional=True,       # Enable bidirectional encoding by default
        enable_attention = True,
        dropout_rate=0.1  # Global dropout rate
    ):
        super().__init__()

        # Global switch for enabling or disabling attention
        self.enable_attention = enable_attention

        # Identifier for the recurrent architecture
        self.architecture = architecture

        # Number of recurrent directions: 1 for unidirectional, 2 for bidirectional
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Bidirectional RNNs concatenate forward and backward outputs,
        # doubling the output dimension (e.g., 128 * 2 = 256)
        self.rnn_output_dim = hidden_dim * self.num_directions

        # Dynamically select the recurrent module
        if architecture not in ('rnn', 'gru', 'lstm'):
            raise ValueError(f"Unsurpported architecture: {architecture}, choose between 'rnn', 'gru' or 'lstm'")

        rnn_class = {
            "rnn": nn.RNN,
            "gru": nn.GRU,
            "lstm": nn.LSTM
        }.get(self.architecture)

        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        # Dropout applied to token embeddings
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )


        # Both attention and the classification head use the actual
        # recurrent output dimension:
        # hidden_dim for unidirectional models, hidden_dim * 2 for bidirectional models
        self.attention = SelfAttention(self.rnn_output_dim)

        # Dropout applied before the classification head to help reduce overfitting
        self.fc_dropout = nn.Dropout(p=dropout_rate)

        self.classifier = nn.Linear(self.rnn_output_dim, 1)


    def encode(self, input_ids, mask):

        # [B, L] -> [B, L, E]
        x = self.embedding(input_ids)
        x = self.emb_dropout(x) # Apply dropout to embeddings

        # Compute valid sequence lengths to avoid recurrent computation over padding
        lengths = mask.sum(dim=1).long()
        
        # Clamp lengths to [1, L] to avoid errors caused by zero-length sequences
        lengths = lengths.clamp(min=1)

        # Pack sequences to skip padded positions during recurrent computation
        packed_x = pack_padded_sequence(
            x, 
            lengths=lengths.cpu(),
            batch_first=True, 
            enforce_sorted=False
        )

        # Run the recurrent encoder
        packed_states, last_hidden = self.rnn(packed_x)

        if self.architecture == "lstm":
            # LSTM returns a tuple (h_n, c_n); only the hidden state h_n is needed
            last_hidden = last_hidden[0]  # [num_directions, B, H]
        
        # Unpack the recurrent outputs back to [B, L, H]
        states, _ = pad_packed_sequence(
            packed_states,
            batch_first=True, 
            total_length=input_ids.size(1) 
        )

        # Aggregate token representations according to the attention setting
        if self.enable_attention:
            # Apply self-attention to the unpacked recurrent states
            padding_mask = ~mask
            attended, weights = self.attention(states, padding_mask)

            # Masked Mean Pooling
            valid_mask = mask.unsqueeze(-1).float()
            context = (attended * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-9)

        else:
            # Use the final hidden state directly when attention is disabled.
            # Packed sequences ensure that padding does not affect the final hidden state.
            weights = None

            # Handle unidirectional and bidirectional hidden states separately
            if self.bidirectional:
                # For the final layer, -2 indexes the forward state
                # and -1 indexes the backward state
                forward_last = last_hidden[-2]
                backward_last = last_hidden[-1]
                context = torch.cat([forward_last, backward_last], dim=-1)

            else:
                # For a unidirectional model, use the final layer's hidden state
                context = last_hidden[-1]

        # Apply dropout before the classification head
        context = self.fc_dropout(context)

        return context, weights


    def forward(self, input_1, input_2, mask_1, mask_2):

        # Candidate 1
        repr_1 = self.encode(input_1, mask_1)

        # Candidate 2
        repr_2 = self.encode(input_2, mask_2)

        context_1, weights_1 = repr_1
        context_2, weights_2 = repr_2

        weights = [weights_1, weights_2]

        # Apply the shared classification head to both candidates
        logit_1 = self.classifier(context_1)
        logit_2 = self.classifier(context_2)

        # [B, 1] + [B, 1]
        # -> [B, 2]
        logits = torch.cat(
            [logit_1, logit_2],
            dim=1
        )

        # Attention weights are None for both candidates when attention is disabled
        return logits, weights


    @torch.no_grad()
    def predict(self, input_1, input_2, mask_1, mask_2):
        """
        Perform inference without computing gradients.

        This method does not switch the model to evaluation mode automatically.
        This avoids silently changing the model's training state. Set the model
        to eval() explicitly when performing evaluation or benchmarking.
        """
        outputs = self.forward(input_1, input_2, mask_1, mask_2)

        return outputs