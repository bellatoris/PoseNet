import torch
import torch.nn as nn


class RegNet(nn.Module):
    def __init__(self, original_model, batch_size, seq_length=5, gru_layer=1):
        super(RegNet, self).__init__()

        # feature들을 마지막 fully connected layer를 제외화고 ResNet으로 부터 가져옴
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        # self.features2 = nn.Sequential(*list(original_model.classifier.children())[:-4])
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.gru_layer = gru_layer
        self.hidden = torch.autograd.Variable(torch.randn(gru_layer, batch_size, 512).cuda())

        # Fully connected GRU
        self.rnn = nn.GRU(512, 512, gru_layer, dropout=0.5)

        # pose regressor
        self.trans_regressor = nn.Sequential(
            nn.Linear(512, 3)
        )
        self.rotation_regressor = nn.Sequential(
            nn.Linear(512, 4)
        )

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

        for m in self.trans_regressor.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.5)
                m.bias.data.zero_()

        for m in self.rotation_regressor.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(0)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, inpt):
        batch_size = self.batch_size
        f0 = self.features(inpt[:, 0])
        f0 = f0.view(batch_size, -1)
        # f0 = self.features2(f0)

        f1 = self.features(inpt[:, 1])
        f1 = f1.view(batch_size, -1)
        # f1 = self.features2(f1)

        f2 = self.features(inpt[:, 2])
        f2 = f2.view(batch_size, -1)
        # f2 = self.features2(f2)

        f3 = self.features(inpt[:, 3])
        f3 = f3.view(batch_size, -1)
        # f3 = self.features2(f3)

        f4 = self.features(inpt[:, 4])
        f4 = f4.view(batch_size, -1)
        # f4 = self.features2(f4)

        f = torch.stack((f0, f1, f2, f3, f4), dim=0).view(self.seq_length, batch_size, -1)

        _, hn = self.rnn(f, self.hidden)
        hn = hn[self.gru_layer - 1].view(batch_size, -1)

        trans = self.trans_regressor(hn)
        rotation = self.rotation_regressor(hn)

        return trans, rotation

