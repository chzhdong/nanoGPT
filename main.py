import torch
import argparse
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from models.model import BigramLanguageModel

def txt_to_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, encode, decode


def get_batch(split, train_data, val_data, config):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i+1 : i + config.block_size + 1] for i in ix])
    return x.to(config.device), y.to(config.device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, train_data, val_data, config)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(model, train_data, val_data, config):
    print(f"[CUDA] Before alloc: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    torch.zeros(1, device=config.device)
    print(f"[CUDA] After alloc:  {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    model = model.to(config.device)
    print(f"[Model] {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_losses, val_losses = [], []

    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            print(f"[Iter {iter}] Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

        xb, yb = get_batch('train', train_data, val_data, config)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('./result/loss_plot.png')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/input.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--n_embd', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(42)
    args = parse_args()

    class Config: pass
    config = Config()
    for k, v in vars(args).items():
        setattr(config, k, v)

    train_data, val_data, vocab_size, encode, decode = txt_to_data(config.data_path)
    model = BigramLanguageModel(
        config.n_embd, 
        config.n_head, 
        config.n_layer, 
        vocab_size, 
        config.block_size, 
        config.dropout, 
        config.device
    )
    train(model, train_data, val_data, config)

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print("[Sample Output]:\n")
    print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
