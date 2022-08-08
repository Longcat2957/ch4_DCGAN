import os
from re import L
import sys
from tkinter.tix import IMAGE

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils as vutils

import utils
from tqdm import tqdm
from time import time

#
# Model parameters
#

CUDA = True
DATA_PATH = '~/Data/mnist'
OUT_PATH = 'output'
LOG_FILE = os.path.join(OUT_PATH, 'log.txt')
BATCH_SIZE = 128
IMAGE_CHANNEL = 1   # 흑백
Z_DIM = 100
G_HIDDEN = 64       # 생성기 은닉층의 최소 채널 크기
X_DIM = 64
D_HIDDEN = 64       # 판별기 은닉층의 최소 채널 크기
EPOCH_NUM = 25
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
seed = 1

#
# 네트워크를 만들기 전의 준비
#
utils.clear_folder(OUT_PATH)
print("Logging to {}\n".format(LOG_FILE))
sys.stdout = utils.StdOut(LOG_FILE)
CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version : {}".format(torch.__version__))
if CUDA:
    # if cuda is available
    print("CUDA version: {}\n".format(torch.version.cuda))
if seed is None:
    # if need random
    seed = np.random.randint(1, 10000)
np.random.seed(seed)
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
cudnn.benchmark = True
device = torch.device("cuda:0" if CUDA else "cpu")

def weights_init(m):
    """custom weights initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#
# 생성기 네트워크
#

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True)
        )
        self.last = nn.Sequential(
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.last(x)
        return out


#
# 판별기 네트워크
#
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)

            # BatchNorm2d가 없는이유는??
            # 모든 층에 배치 정규화를 적용 할 때 원본 논문에서 지적한 것처럼 샘플 진동 및
            # 모델 불완전성을 초래할 수 있기 때문이다.
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.last = nn.Sequential(
            nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.last(x)
        return x.view(-1, 1).squeeze(1)

if __name__ == '__main__':

    netG = Generator().to(device)
    netG.apply(weights_init)
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()        # 이진 분류에 사용되는 손실함수이며
                                    # BCELoss 함수를 쓰기 위해서는 마지막 레이어가 시그모이드 함수이어야 한다.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    dataset = dset.MNIST(root=DATA_PATH, download=True, \
        transform=T.Compose([
            T.Resize(X_DIM),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
            ])
        )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
    )

    # 학습절차는 기본적으로 다음과 같습니다.
    # (1) 실제 데이터로 판별기를 학습시키고 실제 데이터로 인식하기
    # (2) 위조 데이터를 사용하여 판별기를 학습시키고 이를 가짜로 인식하기
    # (3) 가짜 데이터로 생성기를 학습시키고, 실제 데이터로 인식하기

    # 처음 (1), (2) 단계를 통해 판별기는 실제 데이터와 가짜 데이터의
    # 차이점을 구분하는 방법을 배우게 됩니다.

    # 세 번째 단계에서는 생성기에게 판별기가 생성된 샘플을 혼동시키는
    # 방법을 알려주게 됩니다.

    viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
    for epoch in range(EPOCH_NUM):
        for i, data in tqdm(enumerate(dataloader)):
            x_real = data[0].to(device)
            real_label = torch.full((x_real.size(0),), REAL_LABEL, dtype=torch.float32, device=device)
            fake_label = torch.full((x_real.size(0),), FAKE_LABEL, dtype=torch.float32, device=device)

            # Update D with real data
            netD.zero_grad()
            y_real = netD(x_real)
            loss_D_real = criterion(y_real, real_label)
            loss_D_real.backward()

            # Update D with fake data
            z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device=device)
            x_fake = netG(z_noise)
            y_fake = netD(x_fake.detach())
            loss_D_fake = criterion(y_fake, fake_label)
            loss_D_fake.backward()
            optimizerD.step()

            # Update G with fake data
            netG.zero_grad()
            y_fake_r = netD(x_fake)
            loss_G = criterion(y_fake_r, real_label)
            loss_G.backward()
            optimizerG.step()

            if i % 100 == 0:
                print('Epoch {} [{}/{}] loss_D_real: {:.4f} loss_D_fake: {:.4f} loss_G: {:.4f}'.format(
                epoch, i, len(dataloader),
                loss_D_real.mean().item(),
                loss_D_fake.mean().item(),
                loss_G.mean().item()
                ))
                vutils.save_image(x_real, os.path.join(OUT_PATH, 'real_samples.png'), normalize=True)
                with torch.no_grad():
                    viz_sample = netG(viz_noise)
                    vutils.save_image(viz_sample, os.path.join(OUT_PATH, 'fake_samples_{}.png'.format(epoch)), normalize=True)
    torch.save(netG.state_dict(), os.path.join(OUT_PATH, 'netG_{}.pth'.format(epoch)))
    torch.save(netD.state_dict(), os.path.join(OUT_PATH, 'netD_{}.pth'.format(epoch)))



