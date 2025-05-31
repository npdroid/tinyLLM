import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from modules import NeuralNet, CharTokenizer
from warnings import warn
torch.manual_seed(42)  # the answer to life, the universe, and everything


def train(model: NeuralNet, dataloader: DataLoader,
          optimizer: torch.optim.Optimizer, device: str | None = None):
    model.train()
    total_loss = 0.0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)  # [B, T, vocab_size]

        loss = model.loss(logits, target_ids)  # compute loss

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model: NeuralNet, dataloader: DataLoader, device: str | None = None,
             summarize_loss: bool = False):
    model.eval()
    total_loss = 0.0

    loader = dataloader if summarize_loss else tqdm(dataloader, desc="Evaluating")
    for batch in loader:
        input_ids, target_ids = [x.to(device) for x in batch]

        logits = model(input_ids)
        # print(logits.shape)
        loss = model.loss(logits, target_ids)

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def generate(model: NeuralNet, prompt: str, tokenizer: CharTokenizer, device: str | None = None):
    model.eval()
    context_len = model._context_length
    if len(prompt) > 2 / 3 * context_len:
        warn(f'Prompt larger than 2/3 the context length: {len(prompt)} vs {context_len}')
        prompt = prompt[:context_len // 2] if len(prompt) > context_len // 2 else prompt
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

    if device:
        input_ids = input_ids.to(device)

    for _ in range(context_len - len(token_ids)):
        input_ids = input_ids.to(device).to(torch.int)
        logits = model(input_ids)
        next_token = logits[:, -1, :].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(input_ids.squeeze(0).tolist())
