import numpy as np
import torch
import os
import argparse
import time

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from networks.img_transf import ImageTransformNet
from networks.perceptual import Vgg16

# Global Variables
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 2
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-7


def train(args):
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" %torch.cuda.current_device())

    # visualization of training controlled by flag
    visualize = (args.visualize != None)

    # define network
    image_transformer = ImageTransformNet().type(dtype)
    optimizer = Adam(image_transformer.parameters(), LEARNING_RATE) 

    loss_mse = torch.nn.MSELoss()

    # load vgg network
    vgg = Vgg16().type(dtype)

    # get training dataset
    dataset_transform = transforms.Compose([
        transforms.Scale(IMAGE_SIZE),           # scale shortest side to image_size
        transforms.CenterCrop(IMAGE_SIZE),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    train_dataset = datasets.ImageFolder(args.dataset, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)


    style_name = "super_resolution"

    for e in range(EPOCHS):

        # track values for...
        img_count = 0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0

        # train network
        image_transformer.train()
        for batch_num, (x, label) in enumerate(train_loader):
            img_batch_read = len(x)
            img_count += img_batch_read

            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network
            x = Variable(x).type(dtype)
            y_hat = image_transformer(x)

            # get vgg features
            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]      
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.item()

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = TV_WEIGHT*(diff_i + diff_j)
            aggregate_tv_loss += tv_loss.item()

            # total loss
            total_loss = content_loss + tv_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if ((batch_num + 1) % 100 == 0):
                status = f"{time.ctime()}  Epoch {e + 1}:  " \
                         f"[{img_count}/{len(train_dataset)}]  Batch:[{batch_num+1}]  " \
                         f"agg_content: {aggregate_content_loss/(batch_num+1.0):.6f}  " \
                         f"agg_tv: {aggregate_tv_loss/(batch_num+1.0):.6f}  " \
                         f"content: {content_loss:.6f}  tv: {tv_loss:.6f} "
                print(status)

            if ((batch_num + 1) % 1000 == 0) and (visualize):
                image_transformer.eval()

                if not os.path.exists("visualization"):
                    os.makedirs("visualization")
                if not os.path.exists("visualization/%s" %style_name):
                    os.makedirs("visualization/%s" %style_name)

                pass

                print("images saved")
                image_transformer.train()

    # save model
    image_transformer.eval()

    if use_cuda:
        image_transformer.cpu()

    if not os.path.exists("models"):
        os.makedirs("models")
    filename = "models/" + str(style_name) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
    torch.save(image_transformer.state_dict(), filename)
    
    if use_cuda:
        image_transformer.cuda()


def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train a model to do style transfer")
    train_parser.add_argument("--dataset", type=str, required=True, help="path to a dataset")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--visualize", type=int, default=None, help="Set to 1 if you want to visualize training")

    args = parser.parse_args()

    # command
    if (args.subcommand == "train"):
        print("Training!")
        train(args)
    else:
        print("invalid command")


if __name__ == '__main__':
    main()








