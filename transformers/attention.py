import torch
from torch import nn, Tensor


class Attention(nn.Module):
    def __init__(self, word_size:int=512, embed_dim:int=64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dim_K = torch.tensor(embed_dim)
        self.query = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.key  = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)

    def self_attention(self, Q:Tensor, K:Tensor, V:Tensor) -> Tensor:
        K_T = torch.transpose(K, 0, 1)
        score = torch.matmul(Q, K_T)  / torch.sqrt(self.dim_K)
        score = torch.softmax(score, dim=-1)
        Z = torch.matmul(score, V)
        return Z

    def forward(self, x:Tensor) -> Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Z = self.self_attention(Q, K, V)
        return Z


def test_forward_Attention():
    attention = Attention(word_size=512, embed_dim=64)

    # Tạo các embedding của 3 từ
    word1 = torch.randn(1, 512)  # Embedding của từ thứ nhất
    word2 = torch.randn(1, 512)  # Embedding của từ thứ hai
    word3 = torch.randn(1, 512)  # Embedding của từ thứ ba

    # Gộp các embedding thành một tensor đầu vào
    input_tensor = torch.cat([word1, word2, word3], dim=0)

    # Forward pass để tính toán đầu ra
    output = attention(input_tensor)

    # In ra kết quả đầu ra
    print(output)
    print(output.shape) #torch.Size([3, 64])


class MultilHeadAttention(nn.Module):
    def __init__(self, word_size: int = 512, embed_dim: int = 64, n_head:int=8) -> None:
        super().__init__()
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.dim_K = torch.tensor(embed_dim)
        self.W0 = nn.Parameter(torch.empty(embed_dim * n_head, embed_dim))
        nn.init.xavier_normal_(self.W0)
        self.multihead = nn.ModuleList([
            Attention(word_size, embed_dim) for _ in range(n_head)
        ])

    def forward(self, x: Tensor) -> Tensor:
        Z_s = torch.cat([head(x) for head in self.multihead], dim=1)
        Z = torch.matmul(Z_s, self.W0)
        # assert Z_s.shape[1] == self.W0.shape[0]
        # print('Z_s.shape = ',Z_s.shape , '*', self.W0.shape, ' = self.W0')
        return Z

def _test_forward_MultilHeadAttention(word_size=512, embed_dim=64, n_head=8,
                                      device=torch.device('cuda:0')):
    
    mha = MultilHeadAttention(word_size=word_size,
                              embed_dim=embed_dim,
                              n_head=n_head).to(device=device)

    # Tạo các embedding của 3 từ
    word1 = torch.randn(1, word_size)  # Embedding của từ thứ nhất
    word2 = torch.randn(1, word_size)  # Embedding của từ thứ hai
    word3 = torch.randn(1, word_size)  # Embedding của từ thứ ba

    # Gộp các embedding thành một tensor đầu vào
    input_tensor = torch.cat([word1, word2, word3], dim=0).to(device=device)

    # Forward pass để tính toán đầu ra
    output = mha(input_tensor)

    # In ra kết quả đầu ra
    # print(output)
    # print(output.shape) #torch.Size([3, 64])

def test_forward_MultilHeadAttention():
    from random import randint, choice, seed
    from time import time
    seed(0)
    start = time()
    for i in range(1000):

        kwargs = {'word_size':choice([512, 768, 1024]),
                'embed_dim':choice([64, 512, 1024]),
                'n_head':randint(2, 12),
                'device':torch.device('cpu')
                }
        _test_forward_MultilHeadAttention(**kwargs)
    end = time()
    print('runtime = ', end-start)
