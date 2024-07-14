import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class VehicleDynamicEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5):
        super(VehicleDynamicEmbedding, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        return out[:, -1, :]  # Return the last hidden state

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size, num_nodes, _ = Wh.size()
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)
        return a_input

class VehicleRelationEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5, alpha=0.2):
        super(VehicleRelationEmbedding, self).__init__()
        self.gat = GraphAttentionLayer(input_size, hidden_size, dropout, alpha)

    def forward(self, x, adj):
        return self.gat(x, adj)

class Encoder(nn.Module):
    def __init__(self, dynamic_input_size, dynamic_hidden_size, relation_input_size, relation_hidden_size, map_size, combined_size, dropout=0.5):
        super(Encoder, self).__init__()
        self.vehicle_dynamic_embedding = VehicleDynamicEmbedding(dynamic_input_size, dynamic_hidden_size, dropout=dropout)
        self.vehicle_relation_embedding = VehicleRelationEmbedding(relation_input_size, relation_hidden_size, dropout=dropout)
        self.fc_combined = nn.Linear(dynamic_hidden_size + relation_hidden_size + map_size, combined_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dynamic_data, relation_data, adj, map_data):
        dynamic_embedding = self.vehicle_dynamic_embedding(dynamic_data)
        relation_embedding = self.vehicle_relation_embedding(relation_data, adj)
        combined_embedding = torch.cat((dynamic_embedding, relation_embedding, map_data), dim=1)
        combined_embedding = self.fc_combined(combined_embedding)
        combined_embedding = self.dropout(combined_embedding)
        return combined_embedding

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Predict the next position
        return out

class WorldModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(WorldModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, dynamic_data, relation_data, adj, map_data):
        encoder_output = self.encoder(dynamic_data, relation_data, adj, map_data)
        encoder_output = encoder_output.unsqueeze(1)  # Add sequence dimension for decoder
        decoder_output = self.decoder(encoder_output)
        return decoder_output