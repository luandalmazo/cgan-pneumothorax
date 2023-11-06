from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.adam import Adam
from tqdm import tqdm
import torch.nn as nn
import torchvision
import torch
import argparse

from dataset import PneumoDataset, augment_transform, small_transform
from discriminator import Discriminator
from generator import Generator

from utils import weights_init, show_tensor_grayscale
from datetime import datetime
from PIL import Image
import sys


# MODEL ARGUMENTS
parser = argparse.ArgumentParser(description='cgan for data augmentation')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--glr', default=2e-5, type=float)
parser.add_argument('--dlr', default=2e-5, type=float)
# parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--checkpoint', default="", type=str)
args = parser.parse_args()

epochs, glr, dlr = args.epochs, args.glr, args.dlr
batch_size = args.batch_size
checkpoint = args.checkpoint

dataset = PneumoDataset(transform=small_transform)  
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not checkpoint:
    gen = Generator(num_classes=2)
    disc = Discriminator(num_classes=2)
else:
    gen = torch.load(f"gen-{checkpoint}.pkl", map_location=device)
    disc = torch.load(f"disc-{checkpoint}.pkl", map_location=device)

disc_opt = torch.optim.Adam(disc.parameters(), lr=dlr)
gen_opt = torch.optim.Adam(gen.parameters(), lr=glr)
# gen.apply(weights_init)
# disc.apply(weights_init)




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
        disc_opt.step()
        # print("\tdisc", disc_loss.item())
        mean_disc_loss += disc_loss.item()
        


        # Calculate generator loss
        # adversarial loss
        gen_opt.zero_grad()

        fake = gen_fake # non-detached
        disc_result = disc(fake, label) 

        adversarial_loss = criterion(disc_result, torch.ones_like(disc_result))
        # print(adversarial_loss)

        adversarial_loss.backward()
        gen_opt.step()
        # print("\tgen", gen_loss.item())
        mean_gen_loss += adversarial_loss.item()

    # to_show = real

    # if epoch % 10:
    to_show = torch.concatenate((fake.detach()[:10], real[:10]), dim=0)
    show_tensor_grayscale(to_show, show="save", name=f"samples/{epoch}", nrow=5)
    sys.stdout.flush()


    print("disc loss:", mean_disc_loss / image_count)
    print("gen loss:", mean_gen_loss / image_count)


time = str(datetime.now())
torch.save(gen, f"gen-{time}.pkl")
# torch.save(disc, f"disc-{time}.pkl")
    