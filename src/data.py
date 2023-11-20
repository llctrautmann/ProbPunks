from datasets import load_dataset
from params import *
import datasets as ds
from utils import collate_fn
from torch.utils.data import DataLoader

# Data loading and preprocessing
dataset = load_dataset("huggingnft/cryptopunks")
dataset = dataset['train'].train_test_split(test_size=0.05)

train_loader = DataLoader(dataset["train"], batch_size=hp.batch_size, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(dataset["test"], batch_size=hp.batch_size, collate_fn=collate_fn)

