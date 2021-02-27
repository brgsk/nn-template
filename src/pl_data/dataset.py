from omegaconf import DictConfig, ValueNode
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MyDataset(Dataset):
    def __init__(
        self, name: ValueNode, path: ValueNode, train: bool, cfg: DictConfig, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.name = name
        self.train = train

        self.mnist = MNIST(path, train=train, download=True, **kwargs)

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, index: int):
        return self.mnist[index]

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"
