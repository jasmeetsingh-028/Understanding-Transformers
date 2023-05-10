##attention mechaninsm

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads = 8):
        super(SelfAttention, self).__init__()
        #embed_size = dimension of the embedding
        #heads = number of heads
        #head_dim = dimension of each head (embed_size/heads)
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        #query, key, value
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        #fc_out = fully connected layer to combine the heads together (concatenation)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] #number of training examples in the batch
        print('inside self attention:')
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  #length of each sequence in the batch

        print(values.shape)

        #split embedding into self.heads pieces (number of heads)

        values = values.reshape(N, value_len, self.heads, self.head_dim) #reshape into N, value_len, heads, head_dim
        keys = keys.reshape(N, key_len, self.heads, self.head_dim) #reshape into N, key_len, heads, head_dim
        query = query.reshape(N, query_len, self.heads, self.head_dim) #reshape into N, query_len, heads, head_dim

        print("after reshaping: ", values.shape)

        #multiplication of query and key

        #query shape: (N, query_len, heads, head_dim)
        #key shape: (N, key_len, heads, head_dim)
        #energy shape: (N, heads, query_len, key_len)

        #matrix multiplication of query and key

        energy = torch.einsum("nqhd,nkhd->nhqk", [query,keys] ) #einsum is einstein summation

        print("energy shape: ", energy.shape)
        #nqhd = batch size, query_len, heads, head_dim
        #nkhd = batch size, key_len, heads, head_dim
        #nhqk = batch size, heads, query_len, key_len
        
        #masking
        if mask is not None:
            #print(f"energy before masking: {energy}")
            energy = energy.masked_fill(mask == 0, float("-1e20")) #mask == 0 means that the token is a pad token
            
            #print("energy after masking: ", energy)

        attention = torch.softmax(energy / (self.embed_size **(1/2)), dim=3) #dim=3 means that we are applying softmax to the last dimension

        print("attention shape: ", attention.shape)

        #attention shape: (N, heads, query_len, key_len)
        #values shape: (N, value_len, heads, head_dim)
        #out shape: (N, query_len, heads, head_dim)
        #multiply attention with values

        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )

        print("out shape: ", out.shape)
        #flatten out

        out = self.fc_out(out)

        print('out shape after fc_out: ', out.shape)

        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forwrd = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
            
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, key, value, query, mask):
        print(f"inside transformer block: ")

        attention = self.attention(keys = key, query = query, values = value, mask = mask)
        

        #query is added to attention: skip connection
        x = self.dropout(self.norm1(self.norm1(attention + query)))

        print("--------------------x shape--------------------: ", x.shape)

        forward = self.feed_forwrd(x)
        print("--------------------x shape--------------------: ", forward.shape)
        out = self.dropout(self.norm2(forward + x))
        return out
    

#embedding + positional encoding + transformer block

class Encoder(nn.Module):
    def __init__(self, embed_size, heads, src_vocab_size, device, forward_expansion, dropout, max_length):
        super(Encoder,self).__init__()

        self.embed_Size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size= embed_size, heads=heads, dropout=dropout, forward_expansion=forward_expansion)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        print("Inside Encoder:")
        N, seq_length = x.shape   #n examples of some seq_length
        print(f"N: {N}, seq_length: {seq_length}")
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)   #this makes it aware about the positions of the words
        print(f"positions: {positions} positions.shape: {positions.shape}")
        #print(f"self.word_embedding(x): {self.word_embedding(x)} self.word_embedding(x).shape: {self.word_embedding(x).shape}")
        #print(f"self.position_embedding(positions): {self.position_embedding(positions)} self.position_embedding(positions).shape: {self.position_embedding(positions).shape}")
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        print(f"addition shape: {(self.word_embedding(x) + self.position_embedding(positions)).shape}")
        print(f"out.shape: after dropout {out.shape}")

        for layer in self.layers:
            out = layer(out, out, out, mask)

        print(f"------------------out.shape: after transformer block------------------ {out.shape}")

        return out
    


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size= embed_size, heads=heads, dropout=dropout, forward_expansion=forward_expansion)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, trg_mask):  #x is the target sentence and value, key are the encoder outputs of the source sentence and src_mask is the mask for the source sentence and trg_mask is the mask for the target sentence
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    

class Decoder(nn.Module):
    def __init__(self, embed_size, heads, trg_vocab_size, device, forward_expansion, dropout, max_length, num_layers):
        super(Decoder, self).__init__()
        self.embed_Size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout,device) 
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)   #last layer of the decoder
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        print("inside decoder: -------------------")
        print(x.shape)
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)   #this makes it aware about the positions of the words
        print("embeddings shape: ", self.word_embedding(x).shape, self.position_embedding(positions).shape)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        print(f"x.shape: after dropout {x.shape}")

        #-----------------------------------------------------------------------------------------------------------------------------------------
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size = 256, num_layers = 6, forward_expansion = 4, heads = 8, dropout = 0, device = "cuda" if torch.cuda.is_available() else 'cpu', max_length = 100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(embed_size, heads, src_vocab_size, device, forward_expansion, dropout, max_length)

        self.decoder = Decoder(embed_size, heads, trg_vocab_size, device, forward_expansion, dropout, max_length, num_layers)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # this is the src_mask for the encoder 
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        # (N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        print(f"Inside Transformer forward:")
        print({src.shape, trg.shape})
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        print("Masks and their shapes: ")
        print(f"src_mask: {src_mask}")
        print(f"trg_mask: {trg_mask}")
        print(src_mask.shape, trg_mask.shape)
        enc_src = self.encoder(src, src_mask)
        print(f"after encoder operation: {enc_src.shape}")
        print(f"{trg}, {enc_src.shape}, {src_mask}, {trg_mask}")
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out
    

if __name__ == "__main__":  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x = torch.tensor([[1,5,6,4,3,9,5,2,0], [1,8,7,3,4,5,6,7,2]]).to(device)

    trg = torch.tensor([[1,7,4,3,5,9,2,0], [1,5,6,2,4,7,6,2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    print(x.shape, trg[:, :-1])
    print(f"start: ")
    out = model(x, trg[:, :-1])
    print(out.shape)

