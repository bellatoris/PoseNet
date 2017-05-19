import torch
import torch.nn as nn


class RegNet(nn.Module):
    def __init__(self, original_model, batch_size, seq_length=5, gru_layer=1):
        super(RegNet, self).__init__()

        # feature들을 마지막 fully connected layer를 제외화고 ResNet으로 부터 가져옴
        self.features = nn.Sequential(*list(original_model.children())[:-1])

        self.batch_size = batch_size
        self.seq_length = seq_length
        # self.gru_layer = gru_layer
        # self.hidden = torch.autograd.Variable(torch.randn(gru_layer, batch_size, 512).cuda())

        # Fully connected GRU
        # self.rnn = nn.GRU(512, 512, gru_layer, dropout=0.5)
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout()

        self.regressor = nn.Sequential(
            # nn.Linear(512, 2048),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        # pose regressor
        self.trans_regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 3),
        )
        self.scale_regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 1),
        )
        self.rotation_regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 4),
        )

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()

        for m in self.trans_regressor.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        for m in self.scale_regressor.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_()

        for m in self.rotation_regressor.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, inpt):
        batch_size = self.batch_size
        f0 = self.features(inpt[:, 0])
        f0 = f0.view(batch_size, -1)

        f1 = self.features(inpt[:, 1])
        f1 = f1.view(batch_size, -1)

        # f2 = self.features(inpt[:, 2])
        # f2 = f2.view(batch_size, -1)
        #
        # f3 = self.features(inpt[:, 3])
        # f3 = f3.view(batch_size, -1)
        #
        # f4 = self.features(inpt[:, 4])
        # f4 = f4.view(batch_size, -1)
        #
        # f = torch.stack((f0, f1, f2, f3, f4), dim=0).view(self.seq_length, batch_size, -1)

        f = torch.cat((f0, f1), dim=1)

        # _, hn = self.rnn(f, self.hidden)
        # hn = hn[self.gru_layer - 1].view(batch_size, -1)
        # hn = self.relu(hn)
        # hn = self.dropout(hn)
        # hn = self.regressor(hn)
        hn = self.regressor(f)

        trans = self.trans_regressor(hn)

        # trans_norm = torch.norm(trans, dim=1)
        # trans = torch.div(trans, torch.cat((trans_norm, trans_norm, trans_norm), dim=1))

        scale = self.scale_regressor(hn)
        rotation = self.rotation_regressor(hn)

        return trans, scale, rotation

