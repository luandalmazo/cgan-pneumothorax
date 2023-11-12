from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.adam import Adam
from tqdm import tqdm
import torch.nn as nn
import torchvision
import torch
import argparse

from dataset import PneumoDataset, augment_transform, small_transform, default_transform
# from discriminator import Discriminator, get_gradient_penalty
# from generator import Generator
from unet import Generator, Discriminator


from utils import weights_init, show_tensor_grayscale
from datetime import datetime
from PIL import Image
import sys


# MODEL ARGUMENTS
parser = argparse.ArgumentParser(description='cgan for data augmentation')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--glr', default=3e-5, type=float)
parser.add_argument('--dlr', default=3e-5, type=float)
# parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--checkpoint', default="", type=str)
# parser.add_argument('--wgan_coeff', default=10.0, type=float
parser.add_argument('--ppl', default=1, type=float)
# )
args = parser.parse_args()

epochs, glr, dlr = args.epochs, args.glr, args.dlr
batch_size = args.batch_size
checkpoint = args.checkpoint
# wgan_coeff = args.wgan_coeff
ppl_coeff = args.ppl

# dataset = PneumoDataset(transform=small_transform)  #64x64
dataset = PneumoDataset(transform=default_transform)  #256x256
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
gen.apply(weights_init)
disc.apply(weights_init)


gen = nn.DataParallel(gen)
disc = nn.DataParallel(disc)
gen.to(device)
disc.to(device)

# adv_criterion = nn.MSELoss()
# adv_criterion = nn.BCELoss()
adv_criterion = nn.BCEWithLogitsLoss()
ppl_criterion = nn.L1Loss()


for epoch in range(epochs):
    gen.train()
    disc.train()
    print(f"epoch {epoch}")

    mean_disc_loss = .0
    mean_adv_loss = .0
    mean_ppl_loss = .0
    image_count = 0

    for _, pair in enumerate(dataloader):
        
        real, mask = pair
        image_count += len(real)

        real = real.to(device)
        mask = mask.long().to(device)
        gen_fake = gen(mask)
        
        # print(real.shape)
        # print(mask.shape)
        # print(gen_fake.shape)

        assert not torch.isnan(mask).any()
        assert not torch.isnan(real).any()
        assert not torch.isnan(gen_fake).any()

        # Calculate discriminator loss
        disc_opt.zero_grad()

        fake = gen_fake.detach()
        fake_result = disc(fake, mask)
        real_result = disc(real, mask)
        
        disc_fake_loss = adv_criterion(fake_result, torch.zeros_like(fake_result))
        disc_real_loss = adv_criterion(real_result, torch.ones_like(real_result))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        # penalty = get_gradient_penalty(disc, real, fake, mask)
        # disc_loss = torch.mean(fake_result) - torch.mean(real_result) + wgan_coeff * penalty
        
        disc_loss.backward()
        disc_opt.step()
        # print("\tdisc", disc_loss.item())
        mean_disc_loss += disc_loss.item()
        
        # Calculate generator loss
        # adversarial loss
        gen_opt.zero_grad()

        fake = gen_fake # non-detached
        disc_result = disc(fake, mask) 

        adversarial_loss = adv_criterion(disc_result, torch.ones_like(disc_result))
        ppl_loss = ppl_criterion(fake, real)
        # adversarial_loss = -1 * torch.mean(disc_result)
        # print(adversarial_loss)
        gen_loss = adversarial_loss + ppl_coeff * ppl_loss

        gen_loss.backward()
        gen_opt.step()
        # print("\tgen", gen_loss.item())
        mean_adv_loss += adversarial_loss.item()
        mean_ppl_loss += ppl_loss.item()

    # to_show = real

    # if (epoch % 10) == 0:
    if True:
        to_show = torch.concatenate((fake.detach()[:10], real[:10]), dim=0)
        show_tensor_grayscale(to_show, show="save", name=f"samples/{epoch}", nrow=5)


    print("disc loss:", mean_disc_loss / image_count)
    print("gen adv loss:", mean_adv_loss / image_count)
    print("gen ppl loss:", mean_ppl_loss / image_count, f"coeff:{ppl_coeff}")
    sys.stdout.flush()

    if ((epoch % 50) == 0) and (epoch != 0):
        time = str(datetime.now())
        torch.save(gen, f"models/gen-{time}-epoch{epoch}.pkl")


time = str(datetime.now())
torch.save(gen, f"gen-{time}.pkl")
# torch.save(disc, f"disc-{time}.pkl")
    