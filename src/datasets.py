from pathlib import Path

from torch.utils.data import Dataset


class TweetEvalSetiment(Dataset):
    def __init__(self, src="sentiment", type="train") -> None:
        super().__init__()
        assert type in ["train", "val", "test"]

        src = Path(src)
        self.data = open(src / f"{type}_text.txt").read().splitlines()
        self.gt = list(
            map(lambda x: int(x), open(src / f"{type}_labels.txt").read().splitlines())
        )

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, index):
        return self.data[index], self.gt[index]
