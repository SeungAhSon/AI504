import math
import numpy as np
import easydict
import argparse
import random
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import load_dataset
from transformers import AutoTokenizer



class eli5_dataset(Dataset):
    def __init__(self,tokenizer, config, is_train):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.block_size = config.max_position_embeddings
        
        if is_train is True:
            data = load_dataset("eli5", split="train_asks[:30000]")
            data = data.select(range(10000))
        else:
            data = load_dataset("eli5", split="validation_asks[:1000]")

        data = data.flatten() 
        data = data.map(self.preprocess_function, batched=True,num_proc=8,remove_columns=data.column_names)
        data = data.map(self.group_texts, batched=True, num_proc=8)
        result =[]
        for i in data:
            result.append(i['input_ids'])
        self.final_data = torch.tensor(result).to(torch.int64)
        
    def preprocess_function(self, examples):
        return self.tokenizer([" ".join(x) for x in examples["answers.text"]])
    
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]]) 
        if total_length >= (self.block_size-2):
            total_length = (total_length // (self.block_size-2)) * (self.block_size-2)
        result = {
            k: [[self.tokenizer.bos_token_id]+t[i : i + self.block_size-2]+[self.tokenizer.eos_token_id] for i in range(0, total_length, self.block_size-2)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
        
    def __len__(self):
        return len(self.final_data)
    
    def __getitem__(self, idx):
        return self.final_data[idx]


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        emb_dim,
        num_heads,
        dropout=0.0,
        bias=False,
        encoder_decoder_attention=False,  
        causal = False
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == self.emb_dim, "emb_dim must be divisible by num_heads"

        self.encoder_decoder_attention = encoder_decoder_attention
        self.causal = causal
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=bias)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.head_dim,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)



    def scaled_dot_product(self,
                           query: torch.Tensor,
                           key: torch.Tensor,
                           value: torch.Tensor,
                           attention_mask: torch.BoolTensor):

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.emb_dim) 

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1) 
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value) 

        return attn_output, attn_probs


    def MultiHead_scaled_dot_product(self,
                       query: torch.Tensor,
                       key: torch.Tensor,
                       value: torch.Tensor,
                       attention_mask: torch.BoolTensor):

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if self.causal:
                attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(0).unsqueeze(1), float("-inf"))
            else:
                attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), float("-inf"))


        attn_weights = F.softmax(attn_weights, dim=-1) 
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value) 
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        concat_attn_output_shape = attn_output.size()[:-2] + (self.emb_dim,)
        attn_output = attn_output.view(*concat_attn_output_shape)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor = None,
        ):

        q = self.q_proj(query)

        if self.encoder_decoder_attention:
            k = self.k_proj(key)
            v = self.v_proj(key)

        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attn_output, attn_weights = self.MultiHead_scaled_dot_product(q,k,v,attention_mask)
        return attn_output, attn_weights

class PositionWiseFeedForward(nn.Module):

    def __init__(self, emb_dim: int, d_ff: int, dropout: float = 0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.activation = nn.ReLU()
        self.w_1 = nn.Linear(emb_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, emb_dim)
        self.dropout = dropout

    def forward(self, x):
        residual = x
        x = self.activation(self.w_1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.w_2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x + residual 
    

# Use this function in your custom language model
class SinusoidalPositionalEmbedding(nn.Embedding):

    def __init__(self, num_positions, embedding_dim):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight) 

    @staticmethod
    def _init_weight(out: nn.Parameter):
        n_pos, embed_dim = out.shape
        pe = nn.Parameter(torch.zeros(out.shape))
        for pos in range(n_pos):
            for i in range(0, embed_dim, 2):
                pe[pos, i].data.copy_( torch.tensor( np.sin(pos / (10000 ** ( i / embed_dim)))) )
                pe[pos, i + 1].data.copy_( torch.tensor( np.cos(pos / (10000 ** ((i + 1) / embed_dim)))) )
        pe.detach_()

        return pe

    @torch.no_grad()
    def forward(self, input_ids):
      bsz, seq_len = input_ids.shape[:2]
      positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
      return super().forward(positions)

# Use this function in your custom language model    
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.ffn_dim = config.ffn_dim
        self.self_attn = MultiHeadAttention(
            emb_dim=self.emb_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            causal=True,
        )
        self.dropout = config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.emb_dim)
        self.PositionWiseFeedForward = PositionWiseFeedForward(self.emb_dim, self.ffn_dim, config.dropout)
        self.final_layer_norm = nn.LayerNorm(self.emb_dim)


    def forward( 
        self,
        x,
        causal_mask=None,
    ):
        residual = x

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x, 
            attention_mask=causal_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        x = self.PositionWiseFeedForward(x)
        x = self.final_layer_norm(x)

        return (
            x,
            self_attn_weights,
        )
    

class CustomLanguageModel(nn.Module):

    def __init__(self, config, tokenizer):
        super().__init__()

        ############################## Todo 1 ########################################
        ######## Fill in the %%. Do not change the order of the function order #######

        self.embeddings = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=tokenizer.pad_token_id)
        self.positional_embeddings = SinusoidalPositionalEmbedding(config.max_position_embeddings, config.emb_dim)

        self.decoder_layers_1 = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.decoder_layers_2 = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])

        self.rnn = nn.RNN(input_size = config.emb_dim, hidden_size = config.rnn_hidden_size, num_layers = config.rnn_num_layers, batch_first = True)
        self.fc = nn.Linear(config.rnn_hidden_size, config.emb_dim)

        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)

        ############################## Todo 1 ########################################

    def generate_mask(self, trg): 
        tmp = torch.ones(trg.size(1), trg.size(1), dtype=torch.bool)
        mask = torch.arange(tmp.size(-1))
        dec_attention_mask = tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), False).to(trg.device)
        return dec_attention_mask

    def forward(self, input_ids):
        
        ############################## Todo 2 ########################################
        ######## Use the function defined in __init__ and generate_mask ##############
        embeddings = self.embeddings(input_ids)
        positional_embeddings = self.positional_embeddings(input_ids)

        x = embeddings + positional_embeddings

        causal_mask = self.generate_mask(x)

        for layer in self.decoder_layers_1:
            x, _ = layer(x, causal_mask)

        rnn_output, _ = self.rnn(x)
        x = self.fc(rnn_output)

        for layer in self.decoder_layers_2:
            x, _ = layer(x, causal_mask)

        logits = self.lm_head(x)

        return logits
        ############################## Todo 2 ########################################



def main():
    # Do not change
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default = 0)
    parser.add_argument('--stop_epoch', type=int, default = 10)
    parser.add_argument('--output_path', type=str, default = ".")
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device allocated in {DEVICE}")


    # Model Configuration

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    config = easydict.EasyDict({
        "emb_dim":128,
        "ffn_dim":512,
        "attention_heads":4,
        "dropout":0.2,
        "max_position_embeddings":200,
        "num_decoder_layers":3,
        "rnn_hidden_size":64,
        "rnn_num_layers":3,
        "vocab_size" : len(tokenizer),
    })


    
    BATCH_SIZE = 32
    trainset = eli5_dataset(tokenizer, config, is_train=True)
    validset = eli5_dataset(tokenizer, config, is_train=False)

    print(len(trainset)) # 14790
    print(len(validset)) # 5655
    print(len(trainset[0])) # 200
    print(len(validset[0])) # 200

    train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(validset, batch_size=BATCH_SIZE)
    


    model = CustomLanguageModel(config, tokenizer)



    learning_rate = 5e-4
    optimizer = optim.Adam(model.parameters(),lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()

    model.to(DEVICE)

    NUM_EPOCH = 20
    for epoch in range(NUM_EPOCH):
        model.train()
        for batch in tqdm(train_dataloader):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            ############################## Todo 3 ########################################

            # Train your model
            logits = model(batch)
            logits = logits[:, :-1, :].contiguous()
            labels = batch[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            ############################## Todo 3 ########################################
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            model.eval()
            eval_loss=0

            for batch in tqdm(valid_dataloader):
                batch = batch.to(DEVICE)
                ############################## Todo 4 ########################################

                # Evaluation your model
                logits = model(batch)
                logits = logits[:, :-1, :].contiguous()
                labels = batch[:, 1:].contiguous()
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                ############################## Todo 4 ########################################
                eval_loss += loss.item()
        
        print((f"Epoch: {epoch}, Val loss: {eval_loss:.3f}"))


        if epoch == args.stop_epoch:
            
            loss = round(eval_loss, 3)
            print(loss)

            script_path = __file__
            script_name = os.path.basename(script_path)
            script_name = script_name[script_name.rfind("/")+1:]
            script_name = script_name[:script_name.rfind(".")]

            with open(os.path.join(args.output_path, 'result.txt'), 'a') as f:
                f.write(f"{script_name}\t{str(loss)}\n")
            break


if __name__ == '__main__':
    main()