# AG News Text Classification

## Датасет

**AG News** — корпус коротких новостных статей, 4 класса:

| Класс | Категория |
|-------|-----------|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

120 000 обучающих примеров, 7 600 тестовых. Классы сбалансированы.

## Модели

### Baseline

```python
class baseline(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, worker, aggregation_type):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.worker = worker(hidden_dim, hidden_dim, batch_first=True)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, 4)

        self.dropout = nn.Dropout(p=0.75)
        self.nonlin = nn.Tanh()
        self.aggregation_type = aggregation_type

    def common1(self, x) -> List[torch.Tensor]:
        x = self.emb(x)
        output, _ = self.worker(x)

        if self.aggregation_type == 'max':
            pool = output.max(dim=1)[0]
        elif self.aggregation_type == 'mean':
            pool = output.mean(dim=1)
        else:
            raise ValueError("Invalid aggregation_type")
        output = output[:, -1, :]
        return output, pool

    def common2(self, x) -> torch.Tensor:
        x = self.dropout(self.hidden_layer(self.nonlin(x)))
        return self.projection(self.nonlin(x))

    def forward(self, x) -> torch.Tensor:
        _, x = self.common1(x)
        return self.common2(x)
```

### Advanced

```python
class advanced(baseline):
    def __init__(self, vocab_size: int, hidden_dim: int, worker, aggregation_type):
        super().__init__(vocab_size, hidden_dim, worker, aggregation_type)
        self.hidden_layer = nn.Linear(2 * hidden_dim, hidden_dim * 2)
        self.projection = nn.Linear(hidden_dim * 2, 4)

    def forward(self, x) -> torch.Tensor:
        outp, pool = self.common1(x)
        return self.common2(torch.cat([outp, pool], dim=-1))
```

## Стек

- Python, PyTorch
- HuggingFace Datasets
- NLTK
- NumPy, Matplotlib, Seaborn