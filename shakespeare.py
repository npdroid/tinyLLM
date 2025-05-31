from modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.manual_seed(42)  # the answer to life, the universe, and everything
from toolkit import train, evaluate, generate


# %% Create dataset
class DocDataset(Dataset):
    def __init__(self, tokens: list[int], chunk_size: int) -> None:
        self.chunk_size = chunk_size
        n_chunks = len(tokens) // (chunk_size + 1)
        tokens = tokens[:n_chunks * (chunk_size + 1)]
        self.data = torch.tensor(tokens, dtype=torch.int).view(-1, chunk_size + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][:-1], self.data[idx][1:]  # input, next-token target


class ShakespearePred(NeuralNet):
    def __init__(self, config: SysConfig) -> None:
        super().__init__(config.T)
        self.input_layer = InputEmbedding(config)
        self.output_layer = OutputLayer(config, self.input_layer.token_embed)
        self.nn_ = nn.Sequential(self.input_layer,
                                 # transformer block(s)
                                 TransformerBlock(config),
                                 TransformerBlock(config),
                                 TransformerBlock(config),
                                 self.output_layer)

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the digit system.
        :param torch.Tensor token_indices: Token indices of shape (B, T)
        :return torch.Tensor: Output logits of shape (B, T, V)
        """
        return self.nn_(token_indices)

    def loss(self, logits: torch.Tensor, target_indx: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(input=logits.view(-1, logits.size(-1)), target=target_indx.view(-1))

        # create model
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

# load Shakespeare text data
with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
    doc = f.read().lower()

tokenizer = CharTokenizer(list(set(doc)))
print(f'Loaded Shakespeare dataset with {tokenizer.nvocab} unique characters.')
print(f'Total characters in dataset: {len(doc)}')

ShakespeareConfig = SysConfig(V=tokenizer.nvocab, B=32, T=256, C=128, H=4)
tinyllm = ShakespearePred(ShakespeareConfig)
tinyllm.to(DEVICE)

# create dataset and dataloaders
dataset = DocDataset(tokenizer.encode(doc), ShakespeareConfig.T)
train_data, test_data = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_data, batch_size=ShakespeareConfig.B, shuffle=True)
test_loader = DataLoader(test_data, batch_size=ShakespeareConfig.B, shuffle=False)

# initialize model, optimizer, and training loop
model_params = sum(p.numel() for p in tinyllm.parameters() if p.requires_grad)
print(f'Created model with # params: {model_params}')
optimizer = torch.optim.Adam(tinyllm.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
checkpoint_path = 'checkpoints/shakespeare.pth'
checkpoint_interval = 50  # save checkpoint every 50 epochs

# %% Training loop
# NOTE using lowercase to reduce vocab size
input_prompt = "We are accounted poor citizens, the patricians good. What authority surfeits on would relieve us:".lower()
print(f'>>> Initial output: [{input_prompt}]')
print(generate(tinyllm, input_prompt,
               tokenizer, device=DEVICE))

log_statistics = True
use_tensorboard = True


if log_statistics:  # only because vscode/tensorboard is not allowing me to export the data :(
    statistics = open('./train_stat.csv', 'w')
    statistics.write(f'# EPOCH, TRAIN_LOSS, EVAL_LOSS, LR\n')

if use_tensorboard:
    writer = SummaryWriter(log_dir='runs/shakespeare')


for epoch in range(100):
    train_loss = train(tinyllm, train_loader, optimizer, device=DEVICE)
    test_loss = evaluate(tinyllm, test_loader, device=DEVICE, summarize_loss=True)
    scheduler.step(test_loss)
    if use_tensorboard:
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
    if log_statistics:
        statistics.write(f'{train_loss}, {test_loss}, {optimizer.param_groups[0]['lr']}\n')
        statistics.flush()

    print(f'--> Output at {epoch} [{input_prompt}]')
    print(generate(tinyllm, input_prompt,
                   tokenizer, device=DEVICE))

    if epoch % checkpoint_interval == 0 and epoch > 0:
        # Save checkpoint
        tinyllm.eval()
        torch.save(tinyllm.state_dict(), checkpoint_path)
        print(f'... Saved checkpoint at epoch {epoch} to {checkpoint_path}')
