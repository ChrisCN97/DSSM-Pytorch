from data_prepro import trainData
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

"""
size: (batch_num, (batch_size, trigram_dimension)) e.g.(100, (1024, 30k))
query: Query sample
doc_p: Doc Positive sample
doc_n1, doc_n2, doc_n3, doc_n4: Doc Negative sample
"""
query, doc_p, doc_n1, doc_n2, doc_n3, doc_n4 = trainData()
batch_size = 1024
trigram_dimension = 30000

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        assert (trigram_dimension == 30000)
        self.l1 = nn.Linear(30000, 300)
        nn.init.xavier_uniform_(self.l1.weight)
        self.l2 = nn.Linear(300, 300)
        nn.init.xavier_uniform_(self.l2.weight)
        self.l3 = nn.Linear(300, 128)
        nn.init.xavier_uniform_(self.l3.weight)

    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.tanh(self.l3(x))
        return x

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

def train(epoch=20):
    for e_idx in range(epoch):
        batch_idx = 0
        for q, p, n1, n2, n3, n4 in zip(query, doc_p, doc_n1, doc_n2, doc_n3, doc_n4):
            # size: (batch_size, 128)
            out_q = net(q)
            out_p = net(p)
            out_n1 = net(n1)
            out_n2 = net(n2)
            out_n3 = net(n3)
            out_n4 = net(n4)

            # # Relevance measured by cosine similarity
            # size: (batch_size)
            cos_qp = torch.cosine_similarity(out_q, out_p, dim=1)
            cos_qn1 = torch.cosine_similarity(out_q, out_n1, dim=1)
            cos_qn2 = torch.cosine_similarity(out_q, out_n2, dim=1)
            cos_qn3 = torch.cosine_similarity(out_q, out_n3, dim=1)
            cos_qn4 = torch.cosine_similarity(out_q, out_n4, dim=1)
            cos_uni = torch.cat((cos_qp, cos_qn1, cos_qn2, cos_qn3, cos_qn4), 1)  # size: (batch_size,5)

            # # posterior probability computed by softmax
            softmax_qp = F.softmax(cos_uni, dim=1)[:, 0]  # size: (batch_size)
            loss = -torch.log(torch.prod(softmax_qp))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_idx += 1
