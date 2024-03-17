# Do not add any other packages
# Use these packages to write your code
# Do not change
import os
import copy
import json
import math
import random
import easydict
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import AutoTokenizer

PAD = '[PAD]'
IMG_PATCH = '<img>'

# Do not change
class eli5_dataset(Dataset):
    def __init__(self, tokenizer, config, is_train):
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = config.max_position_embeddings

        if is_train is True:
            data = load_dataset("eli5", split="train_asks[:10000]")
        else:
            data = load_dataset("eli5", split="validation_asks[:1000]")

        data = data.flatten()
        data = data.map(self.preprocess_function, batched=True, num_proc=8, remove_columns=data.column_names)
        data = data.map(self.group_texts, batched=True, num_proc=8)
        result = []
        for i in data:
            result.append(i['input_ids'])
        self.final_data = torch.tensor(result).to(torch.int64)

    def preprocess_function(self, examples):
        return self.tokenizer([" ".join(x) for x in examples["answers.text"]])

    def group_texts(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= (self.block_size - 2):
            total_length = (total_length // (self.block_size - 2)) * (self.block_size - 2)
        result = {
            k: [[self.tokenizer.bos_token_id] + t[i: i + self.block_size - 2] + [self.tokenizer.eos_token_id] for i in
                range(0, total_length, self.block_size - 2)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def __len__(self):
        return len(self.final_data)

    def __getitem__(self, idx):
        return self.final_data[idx]

# Do not change
def transform_fn(is_train):
    if is_train:
        return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    else:
        return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

# Do not change
class VisionLLMDatset(Dataset):
    def __init__(self, args, tokenizer, config, is_train):
        super().__init__()

        self.transform = transform_fn(is_train)

        self.json_file = args.json_file

        self.tokenizer = tokenizer
        self.block_size = config.max_position_embeddings
        self.num_img_token = args.num_img_token
        self.img_path = args.img_path

        self.ignore_idx = -100
        self.begin_signal = tokenizer.bos_token
        self.end_signal = tokenizer.eos_token

        with open(self.json_file) as json_file:
            data = json.load(json_file)

        if is_train:
            data = data[:1000]
        else:
            data = data[1000:]

        self.data = data

    def preprocess(self, conversation):
        image = IMG_PATCH * self.num_img_token
        question = self.begin_signal + "human: " + conversation[0]['value'] + self.end_signal
        answer = self.begin_signal + "assistant: " + conversation[1]['value'] + self.end_signal

        tokenized_img = self.tokenizer(image, return_tensors="pt")
        tokenized_q = self.tokenizer(question, return_tensors="pt")

        combined_qa = question + answer
        tokenized_qa = self.tokenizer(combined_qa, padding="max_length", truncation=True,
                                      max_length=(self.block_size - len(tokenized_img.input_ids[0])), return_tensors="pt")

        input_ids = tokenized_qa.input_ids[0]
        label = copy.deepcopy(input_ids)
        len_of_q = len(tokenized_q.input_ids[0])
        label[:len_of_q] = self.ignore_idx

        len_of_pad = tokenized_qa.input_ids.eq(self.tokenizer.pad_token_id).sum().item()
        label[-len_of_pad:] = self.ignore_idx
        label = torch.concat([(torch.zeros_like(tokenized_img.input_ids[0]) + self.ignore_idx), label])

        return input_ids, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        meta = self.data[idx]

        image_id = meta['image']
        image = Image.open(os.path.join(self.img_path, image_id)).convert('RGB')
        image = self.transform(image)

        conversation = meta['conversation']
        input_id, label = self.preprocess(conversation)

        return dict(image=image, input_ids=input_id, label=label)  # , attention_mask=attention_mask)

# Do not change
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            emb_dim,
            num_heads,
            dropout=0.0,
            bias=False,
            encoder_decoder_attention=False,
            causal=False
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

        attn_output, attn_weights = self.MultiHead_scaled_dot_product(q, k, v, attention_mask)
        return attn_output, attn_weights
# Do not change
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
# Do not change
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
                pe[pos, i].data.copy_(torch.tensor(np.sin(pos / (10000 ** (i / embed_dim)))))
                pe[pos, i + 1].data.copy_(torch.tensor(np.cos(pos / (10000 ** ((i + 1) / embed_dim)))))
        pe.detach_()

        return pe

    @torch.no_grad()
    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)
# Do not change
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

    def forward(self, x, causal_mask=None,):
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

        return (x, self_attn_weights,)

class Stage1_CustomClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        ############################## Todo 1 ########################################
        # Stage 1: CNN-based Image Classifier
        # Define layers in the order of Convolutional Blocks, Pooling Layer, and Fully Connected Layer
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU()
        )
        
        self.pooling = nn.AvgPool2d(kernel_size=6, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 10)

        ############################## Todo 1 ########################################


    def forward(self, x):
        ############################## Todo 2 ########################################
        ########       Use the function defined in __init__             ##############
        
        # Forward pass through Convolutional Blocks
        x = self.conv_block(x)
        features_before_pooling = x.clone()
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        return {'avgpool': features_before_pooling, 'logits': x}
        ############################## Todo 2 ########################################



class Stage2_CustomLanguageModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()

        ############################## Todo 3 ########################################
        # Stage 2: Transformer Decoder-based Language Model
        ######## Fill in the %%. Do not change the order of the function order #######
        self.embeddings = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=tokenizer.pad_token_id)
        self.positional_embeddings = SinusoidalPositionalEmbedding(config.max_position_embeddings, config.emb_dim)

        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])

        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)
        ############################## Todo 3 ########################################

    def generate_mask(self, trg):
        tmp = torch.ones(trg.size(1), trg.size(1), dtype=torch.bool)
        mask = torch.arange(tmp.size(-1))
        dec_attention_mask = tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), False).to(trg.device)
        return dec_attention_mask

    def forward(self, input):
        ############################## Todo 4 ########################################
        ######## Use the function defined in __init__ and generate_mask ##############

        # Embeddings
        embeddings = self.embeddings(input)
        positional_embeddings = self.positional_embeddings(input)
        
        x = embeddings + positional_embeddings
        causal_mask = self.generate_mask(x)
        
        for layer in self.decoder_layers:
            x, _ = layer(x, causal_mask=causal_mask)

        x = self.lm_head(x)
        
        return x
        ############################## Todo 4 ########################################


class Stage3_CustomVisionLanguageModel(nn.Module):
    def __init__(self, vision_encoder, text_decoder):
        super().__init__()
        ############################## Todo 5 #################elf.text_decoder#######################
        ###### Stage 3: Vision Language Model

        ###### First. freeze the stages 1 and 2 model here
        for param in vision_encoder.parameters():
            param.requires_grad = False
            
        for param in text_decoder.parameters():
            param.requires_grad = False

        ###### Second. Define layers in the order of vision_encoder, projection_layer, and text_decoder (Do not change the order).
         
        self.vision_encoder = vision_encoder
        self.projection_layer = nn.Linear(64, 128)
        self.text_decoder = text_decoder

        ############################## Todo 5 ########################################

    def forward(self, img, input_ids):
        ############################## Todo 6 ########################################
        ########       Use the function defined in __init__             ##############
        image_features = self.vision_encoder(img)
        features_before_pooling = image_features['avgpool']

        reshaped_features = features_before_pooling.view(features_before_pooling.size(0), -1, features_before_pooling.size(1))
        projected_features = self.projection_layer(reshaped_features)

        text_embeddings = self.text_decoder.embeddings(input_ids)
        combined_input = torch.cat((projected_features, text_embeddings), dim=1)
        
        #-----------
        positional_embeddings = self.text_decoder.positional_embeddings(combined_input)
        x = combined_input + positional_embeddings
        casual_mask = self.text_decoder.generate_mask(combined_input)

        for layer in self.text_decoder.decoder_layers:
            x, _ = layer(x, casual_mask)  

        x = self.text_decoder.lm_head(x)
        
        return x

        ############################## Todo 6 ########################################


def stage1(args, device):
    # Prepare dataset for stage 1
    train_dataset_stage1 = datasets.CIFAR10('./', download=True, train=True, transform=transform_fn(is_train=True))
    valid_dataset_stage1 = datasets.CIFAR10('./', download=False, train=False, transform=transform_fn(is_train=False))

    train_dataloader_stage1 = DataLoader(train_dataset_stage1, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader_stage1 = DataLoader(valid_dataset_stage1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Prepare model
    model_stage1 = Stage1_CustomClassifier()
    model_stage1.to(device)

    #
    learning_rate = 5e-4
    optimizer = optim.SGD(model_stage1.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
  
    for epoch in range(args.total_epoch):
      	
        model_stage1.train()
        for batch in tqdm(train_dataloader_stage1, total=len(train_dataloader_stage1)):
            img, target = batch
            img = img.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            ############################## Todo 7 ########################################
            ####### Train your model

            # Forward pass
            output = model_stage1(img)['logits']
            loss = criterion(output, target)

            ############################## Todo 7 ########################################
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model_stage1.eval()
            eval_loss = 0

            for batch in tqdm(valid_dataloader_stage1, total=len(valid_dataloader_stage1)):
                img, target = batch
                img = img.to(device)
                target = target.to(device)
                ############################## Todo 8 ########################################
                ####### Evaluate your model

                # Forward pass                
                output = model_stage1(img)['logits']
                loss = criterion(output, target)

                ############################## Todo 8 ########################################
                eval_loss += loss.item()

        print((f"Epoch: {epoch}, Val loss: {eval_loss:.3f}"))

    return model_stage1, eval_loss

def stage2(args, config, tokenizer, device):
    # Dataloader for the stage 2
    train_dataset_stage2 = eli5_dataset(tokenizer, config, is_train=True)
    valid_dataset_stage2 = eli5_dataset(tokenizer, config, is_train=False)

    train_dataloader_stage2 = DataLoader(train_dataset_stage2, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader_stage2 = DataLoader(valid_dataset_stage2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_stage2 = Stage2_CustomLanguageModel(config, tokenizer)
    model_stage2.to(device)

    learning_rate = 5e-4
    optimizer = optim.Adam(model_stage2.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.total_epoch):
      	
        model_stage2.train()
        for batch in tqdm(train_dataloader_stage2, total=len(train_dataloader_stage2)):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            ############################## Todo 9 ########################################
            ####### Train your model
            # Extract input_ids and target from the batch
            logits = model_stage2(batch)
            logits = logits[:, :-1, :].contiguous()
            labels = batch[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            ############################## Todo 9 ########################################
            loss.backward()
            optimizer.step()
    
        with torch.no_grad():
            model_stage2.eval()
            eval_loss = 0

            for batch in tqdm(valid_dataloader_stage2, total=len(valid_dataloader_stage2)):
                batch = batch.to(device)
                ############################## Todo 10 ########################################
                ####### Evaluate your model
                logits = model_stage2(batch)
                logits = logits[:, :-1, :].contiguous()
                labels = batch[:, 1:].contiguous()
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                ############################## Todo 10 ########################################
                eval_loss += loss.item()

        print((f"Epoch: {epoch}, Val loss: {eval_loss:.3f}"))

    return model_stage2, eval_loss

def stage3(args, model_stage1, model_stage2, tokenizer, config, device):
    # Dataloader for the stage 3
    train_dataset_stage3 = VisionLLMDatset(args, tokenizer, config, is_train=True)
    valid_dataset_stage3 = VisionLLMDatset(args, tokenizer, config, is_train=False)

    train_dataloader_stage3 = DataLoader(train_dataset_stage3, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader_stage3 = DataLoader(valid_dataset_stage3, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_stage3 = Stage3_CustomVisionLanguageModel(model_stage1, model_stage2)
    model_stage3.to(device)

    learning_rate = 5e-4
    optimizer = optim.Adam(model_stage3.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.total_epoch):
        model_stage3.train()
        for batch in tqdm(train_dataloader_stage3, total=len(train_dataloader_stage3)):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()

            ############################## Todo 11 ########################################
            ####### Train your model
            logits = model_stage3(images, input_ids)
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
         
            ############################## Todo 11 ########################################
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model_stage3.eval()
            eval_loss = 0

            for batch in tqdm(valid_dataloader_stage3, total=len(valid_dataloader_stage3)):
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                ############################## Todo 12 ########################################
                ####### Evaluate your model
                logits = model_stage3(images, input_ids)
                logits = logits[:, :-1, :].contiguous()
                labels = labels[:, 1:].contiguous()
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                ############################## Todo 12 ###Ã¯Â¿Â¼#####################################
                eval_loss += loss.item()

        print((f"Epoch: {epoch}, Val loss: {eval_loss:.3f}"))

    return eval_loss


def main():
    # Do not change
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total_epoch', type=int, default=3)
    parser.add_argument('--num_img_token', type=int, default=36)
    parser.add_argument('--output_path', type=str, default=".")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--img_path', type=str, default='.')
    parser.add_argument('--json_file', type=str, default='.')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    # Fix the seed for reproducibilty
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device allocated in {device}")


    # Model Configuration
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': PAD})
    tokenizer.add_tokens(IMG_PATCH, special_tokens=True)

    config = easydict.EasyDict({
        "emb_dim": 128,
        "ffn_dim": 256,
        "attention_heads": 4,
        "dropout": 0.2,
        "max_position_embeddings": 200,
        "num_decoder_layers": 4,
        "vocab_size": len(tokenizer),
    })

    # logging
    script_path = __file__
    script_name = os.path.basename(script_path)
    script_name = script_name[script_name.rfind("/") + 1:]
    script_name = script_name[:script_name.rfind(".")]

    model_stage1, eval_loss_stage1 = stage1(args, device)
    with open(os.path.join(args.output_path, 'result_stage1.txt'), 'a') as f:
        f.write(f"{script_name}\t{str(round(eval_loss_stage1, 3))}\n")

    model_stage2, eval_loss_stage2 = stage2(args, config, tokenizer, device)
    with open(os.path.join(args.output_path, 'result_stage2.txt'), 'a') as f:
        f.write(f"{script_name}\t{str(round(eval_loss_stage2, 3))}\n")

    eval_loss_stage3 = stage3(args, model_stage1, model_stage2, tokenizer, config, device)
    with open(os.path.join(args.output_path, 'result_stage3.txt'), 'a') as f:
        f.write(f"{script_name}\t{str(round(eval_loss_stage3, 3))}\n")


if __name__ == '__main__':
    main()