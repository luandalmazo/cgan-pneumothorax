from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.adam import Adam
from tqdm import tqdm
import torch.nn as nn
import torchvision
import torch
import argparse

from dataset import PneumoDataset
from discriminator import Discriminator
from generator import Generator

from utils import weights_init, show_tensor_grayscale
from datetime import datetime
from PIL import Image
import sys


# MODEL ARGUMENTS
parser = argparse.ArgumentParser(description='cgan for data augmentation')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--glr', default=1e-6, type=float)
parser.add_argument('--dlr', default=1e-6, type=float)
parser.add_argument('--batch_size', default=16, type=int)
args = parser.parse_args()

epochs, glr, dlr = args.epochs, args.glr, args.dlr
batch_size = args.batch_size

dataloader = DataLoader(PneumoDataset(), batch_size=batch_size, shuffle=True)


gen = Generator(num_classes=2)
gen_opt = torch.optim.Adam(gen.parameters(), lr=glr)

disc = Discriminator(num_classes=2)
disc_opt = torch.optim.Adam(disc.parameters(), lr=dlr)

# gen.apply(weights_init)
# disc.apply(weights_init)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# gen = nn.DataParallel(gen)
# disc = nn.DataParallel(disc)
gen.to(device)
disc.to(device)

# criterion = nn.MSELoss()
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()


for epoch in range(epochs):
    gen.train()
    disc.train()
    print(f"epoch {epoch}")

    mean_disc_loss = .0
    mean_gen_loss = .0
    image_count = 0

    for _, pair in enumerate(dataloader):
        
        real, label = pair

        image_count += len(real)

        real = real.to(device)
        label = label.long().to(device)
        gen_fake = gen(label)
        
        # print(real.shape)
        # print(label.shape)
        # print(gen_fake.shape)

        assert not torch.isnan(label).any()
        assert not torch.isnan(real).any()
        assert not torch.isnan(gen_fake).any()

        # Calculate discriminator loss
        disc_opt.zero_grad()

        fake = gen_fake.detach()
        fake_result = disc(fake, label)
        disc_fake_loss = criterion(fake_result, torch.zeros_like(fake_result))
        real_result = disc(real, label)
        disc_real_loss = criterion(real_result, torch.ones_like(real_result))

        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        disc_loss.backward()
        # print("\tdisc", disc_loss.item())
        mean_disc_loss += disc_loss.item()
        
        disc_opt.step()


        # Calculate generator loss
        # adversarial loss
        gen_opt.zero_grad()

        fake = gen_fake # non-detached
        disc_result = disc(fake, label) 

        adversarial_loss = criterion(disc_result, torch.ones_like(disc_result))
        # print(adversarial_loss)

        adversarial_loss.backward()
        # print("\tgen", gen_loss.item())
        mean_gen_loss += adversarial_loss.item()
        gen_opt.step()

    show_tensor_grayscale(fake.detach(), show="save", name=f"samples/{epoch}")
    sys.stdout.flush()


    print("disc loss:", mean_disc_loss / image_count)
    print("gen loss:", mean_gen_loss / image_count)


time = str(datetime.now())
torch.save(gen, f"gen-{time}.pkl")
# torch.save(disc, f"disc-{time}.pkl")
    