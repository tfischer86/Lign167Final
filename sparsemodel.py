import torch
from torch import nn
import torch.nn.functional as F
import math
import random

class SparseLayer(nn.Linear):
    def __init__(self, in_features, out_features, p_weight=0.25, zeta=0.3, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.p_weight = p_weight
        self.zeta = zeta
        self.sparse = False

    def initialize_sparse_weights(self):
        with torch.no_grad():
            # Erdos-Renyi Sparsity
            #p_weight = (self.epsilon * (self.in_features + self.out_features)/(self.in_features * self.out_features))
            self.weight[torch.rand(self.weight.size()) >= self.p_weight] = 0

        self.zero_weights = (self.weight == 0)
        self.sparse = True

    def num_weights(self):
        return torch.count_nonzero(self.weight).item()

    def grow_connections(self):
        with torch.no_grad():
            # get flattened view of the weights
            weight = self.weight.view(-1)

            # generate new weights
            stdv = math.sqrt(4/self.in_features)
            new_values = torch.randn(self.num_removed, device=self.weight.device) * stdv

            # randomly assign the values to empty weights
            empty_weight_indices = (weight == 0).nonzero().flatten().tolist()
            indices = random.sample(empty_weight_indices, self.num_removed)
            self.new_connections = indices
            weight[indices] = new_values

    def cut_connections(self):
        with torch.no_grad():
            num_weights = torch.count_nonzero(self.weight).item()
            num_removed = int(self.zeta * num_weights)

            weight = self.weight.view(-1)

            # find weights closest to 0
            live_weight = weight.clone()
            # only consider live weights
            live_weight[live_weight == 0] = math.inf
            _, killed = torch.topk(torch.abs(live_weight), k=num_removed, largest=False)
            weight[killed] = 0 # self.weights is modified due to using view()
            self.num_removed = len(killed)

    def evolve(self):
        self.cut_connections()
        self.grow_connections()
        self.zero_weights = (self.weight == 0)

    def freeze_zero_weights(self):
        with torch.no_grad():
            self.weight[self.zero_weights] = 0

    def undo_new_connections(self):
        with torch.no_grad():
            self.weight.view(-1)[self.new_connections] = 0


class SparseEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, args, batch_first = True):
        super().__init__(d_model=args.embed_dim, nhead=args.nhead, dim_feedforward=args.feedforward_dim, dropout=args.transformer_drop_rate, batch_first=batch_first)
        self.linear1 = SparseLayer(args.embed_dim, args.feedforward_dim, p_weight=args.sparse_layer_pweight, zeta=args.sparse_layer_zeta)
        self.linear2 = SparseLayer(args.feedforward_dim, args.embed_dim, p_weight=args.sparse_layer_pweight, zeta=args.sparse_layer_zeta)


class TransformerEmbeddings(nn.Module):
    """
    Transformer embeddings, based off of the BertEmbeddings class from
    https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/bert/modeling_bert.py
    """
    def __init__(self, vocab_size, max_length, embed_size, dropout, layer_norm_eps=1e-5):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.pos_embeddings = nn.Embedding(max_length, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("position_ids", torch.arange(max_length).expand((1, -1)))

    def forward(self, x):
        pos_embeds = self.pos_embeddings(self.position_ids[:x.shape[1]])
        word_embeds = self.word_embeddings(x)
        embeddings = word_embeds + pos_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Classifier(nn.Module):
    def __init__(self, args, target_size):
        super().__init__()
        input_dim = args.embed_dim
        self.dropout = nn.Dropout(args.classifier_drop_rate)
        self.relu = nn.ReLU(inplace=True)

        if args.classifier == 'sparse':
            self.fc1 = SparseLayer(input_dim, args.classifier_hidden_dim, p_weight=args.classifier_pweight, zeta=args.classifier_zeta)
            self.fc2 = SparseLayer(args.classifier_hidden_dim, target_size, p_weight=args.classifier_pweight, zeta=args.classifier_zeta)
        else:
            self.fc1 = nn.Linear(input_dim, args.classifier_hidden_dim)
            self.fc2 = nn.Linear(args.classifier_hidden_dim, target_size)

    def forward(self, x):
        x = self.fc1(self.dropout(x))
        return self.fc2(self.relu(x))

class EvolveModel(nn.Module):
    def init_sparse_layers(self):
        for m in self.modules():
            if isinstance(m, SparseLayer):
                m.initialize_sparse_weights()

    def evolve(self):
        for m in self.modules():
            if isinstance(m, SparseLayer):
                m.evolve()

    def freeze_zero_weights(self):
        for m in self.modules():
            if isinstance(m, SparseLayer):
                m.freeze_zero_weights()

    def undo_new_connections(self):
        for m in self.modules():
            if isinstance(m, SparseLayer):
                m.undo_new_connections()


class SparseModel(EvolveModel):
    def __init__(self, args, tokenizer, target_size=60):
        super().__init__()
        # input embeddings
        self.vocab_size = len(tokenizer)
        self.embeddings = TransformerEmbeddings(self.vocab_size, args.max_len, args.embed_dim, args.transformer_drop_rate)

        # transformer layers
        if args.transformer == "sparse":
            self.encoder_layer = SparseEncoderLayer(args)
        else:
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.embed_dim, nhead=args.nhead, dim_feedforward=args.feedforward_dim, dropout=args.transformer_drop_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.num_transformer_layers)

        # output classifier
        self.classify = Classifier(args, target_size)

        # setup sparse layers
        self.init_sparse_layers()

    def forward(self, input_ids, padding_mask):
        embeddings = self.embeddings(input_ids)
        outputs = self.encoder(embeddings, src_key_padding_mask=padding_mask)
        cls_output = outputs[:, 0, :]

        return self.classify(cls_output)
