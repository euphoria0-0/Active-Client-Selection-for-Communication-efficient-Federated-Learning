import torch.nn as nn


class BLSTM(nn.Module):
    def __init__(self, embedding_dim=64, vocab_size=500, blstm_hidden_size=32, mlp_hidden_size=64, blstm_num_layers=1, num_classes=2):
        super(BLSTM, self).__init__()
        # AFL: 64-dim embedding, 32-dim BLSTM, MLP with one layer(64-dim)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.blstm = nn.LSTM(input_size=embedding_dim, hidden_size=blstm_hidden_size, num_layers=blstm_num_layers,
                             batch_first=True, bidirectional=True)
        #self.fc1 = nn.Linear(blstm_hidden_size*2, mlp_hidden_size)
        #self.fc2 = nn.Linear(mlp_hidden_size, 2)
        self.fc = nn.Linear(blstm_hidden_size*2, num_classes)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        lstm_out, _ = self.blstm(embeds)
        final_hidden_state = lstm_out[:, -1]
        #output = self.fc1(final_hidden_state)
        #output = self.fc2(output)
        output = self.fc(final_hidden_state)
        return output