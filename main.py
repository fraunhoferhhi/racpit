import numpy as np
import xarray as xr
import torch
import os
import argparse
import time
import json

import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn

from networks.img_transf import ImageTransformNet, MultiTransformNet
from networks.perceptual import Vgg16, RDNet, RDPerceptual, RACPIT

from utils.slicer import train_test_slice
from utils.provider import RadarDataset
from utils.preprocess import open_recordings
from utils.visualization import spec_plot

# Global Variables
BATCH_TRANSFER = 4
BATCH_CLASSIFY = 32
LEARNING_RATE = 1e-3
EPOCHS = 100
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-7

REAL_PATH = "/mnt/infineon-radar/preprocessed/real"
SYNTH_PATH = "/mnt/infineon-radar/preprocessed/synthetic"

# Radar Processing variables
range_length = 128
doppler_length = 128
time_length = 64
hop_length = 8
ignore_dims = False


def train_transfer(args):
    # GPU enabling
    if args.gpu is None:
        use_cuda = False
        dtype = torch.FloatTensor
        label_type = torch.LongTensor
        print("No GPU training")
    else:
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        label_type = torch.cuda.LongTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" % torch.cuda.current_device())

    epochs = args.epochs

    # visualization of training controlled by flag
    visualize = 0 if args.visualize is None else args.visualize

    # define network
    if args.range:
        image_transformer = MultiTransformNet(num_inputs=2, num_channels=1).type(dtype)
    else:
        image_transformer = ImageTransformNet(num_channels=1).type(dtype)
    optimizer = Adam(image_transformer.parameters(), LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    loss_mse = torch.nn.MSELoss()

    # get training dataset
    config = args.config
    input_path = args.input
    output_path = args.output
    print(f"Training with configurations {', '.join(config)}.")
    print(f"Using data from {input_path} as the input and data from {output_path} as an output.")

    train_load = 0.8
    num_workers = 4

    if args.recordings is not None:
        split = 'no-cut'
        train_load = 0.5
    elif args.segments is None:
        split = 'single'
    else:
        with open(args.segments, "r") as f:
            split = json.load(f)

    recordings_input = open_recordings(config, input_path,
                                       load=True, range_length=range_length, doppler_length=doppler_length)
    recordings_output = open_recordings(config, output_path,
                                        load=True, range_length=range_length, doppler_length=doppler_length)

    # Merge recordings from all configs
    recordings_input = [r for recs in recordings_input.values() for r in recs]
    recordings_output = [r for recs in recordings_output.values() for r in recs]
    if args.range:
        recordings_input = [r.rename_vars(range_spect="range_real") for r in recordings_input]
        recordings_output = [r.rename_vars(range_spect="range_synth") for r in recordings_output]
    else:  # drop range spectrograms
        recordings_input = [r.drop_vars("range_spect") for r in recordings_input]
        recordings_output = [r.drop_vars("range_spect") for r in recordings_output]

    recordings = [xr.merge([rec_real.rename_vars(doppler_spect="doppler_real"),
                            rec_synth.rename_vars(doppler_spect="doppler_synth")], combine_attrs="drop_conflicts")
                  for rec_real, rec_synth in zip(recordings_input, recordings_output)]

    if args.classes is not None:
        classes = args.classes
        print(f"Selecting classes {classes}")
        new_recs = []
        for r in recordings:
            if r.label in classes:
                new_recs.append(r)
        print(f"{len(new_recs)} out of {len(recordings)} selected")
        recordings = new_recs

    slice_kwargs = dict(spec_length=time_length, stride=hop_length, train_load=train_load, copy_split=0)
    loader_kwargs = dict(batch_size=BATCH_TRANSFER, shuffle=True, num_workers=num_workers, pin_memory=True)

    print("Preloading datasets...")
    slice_output = slice_datasets(recordings, split=split, **slice_kwargs)
    if args.recordings is None:
        if args.segments is None:
            [train_dataset, test_dataset], tgt_segments = slice_output
        else:
            train_dataset, test_dataset = slice_output
    elif args.recordings == 'first':
        [train_dataset, test_dataset], tgt_segments = slice_output
    elif args.recordings == 'last':
        [test_dataset, train_dataset], tgt_segments = slice_output
    else:
        raise ValueError(f"Unrecognized recordings option {args.recordings}")

    train_loader = DataLoader(train_dataset, **loader_kwargs)

    test_indices = np.random.choice(len(test_dataset), size=visualize, replace=False)

    class_num = train_dataset.class_num
    input_shapes = train_dataset.feature_shapes
    input_shapes = input_shapes[:len(input_shapes)//2]
    print(f"Number of classes: {class_num}")
    print(f"Feature shapes: {input_shapes}\n")

    # load perceptual loss network
    if args.model is None:
        perceptual_net = Vgg16().type(dtype)    # Only works with single input i.e. without range
    else:
        perceptual_net = RDPerceptual(args.model, input_shapes=input_shapes, class_num=class_num).type(dtype)

    log_id = args.log

    # calculate gram matrices for style feature layer maps we care about
    # style_features = vgg(style)
    # style_gram = [utils.gram(fmap) for fmap in style_features]

    loss_logs = []

    for e in range(epochs):

        # track values for...
        img_count = 0
        aggregate_content_loss = 0.0
        aggregate_classify_loss = 0.0
        aggregate_tv_loss = 0.0
        batch_num = 0

        # train network
        image_transformer.train()
        for batch_num, (feature_batch, label) in enumerate(train_loader):
            img_batch_read = len(label)
            img_count += img_batch_read

            if args.range:
                [real_range, real_doppler, synth_range, synth_doppler] = feature_batch
                x = [Variable(real_feat).type(dtype) for real_feat in (real_range, real_doppler)]
                y_c = [Variable(synth_feat).type(dtype) for synth_feat in (synth_range, synth_doppler)]
                pass
            else:
                [real_batch, synth_batch] = feature_batch
                x = Variable(real_batch).type(dtype)
                y_c = Variable(synth_batch).type(dtype)
            label_true = Variable(label).type(label_type)

            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network
            y_hat = image_transformer(x)

            # get vgg features
            y_c_features = perceptual_net(y_c)
            y_hat_features = perceptual_net(y_hat)

            # calculate classification loss w.r.t. input
            label_pred = y_hat_features[0]
            classify_loss = CONTENT_WEIGHT*criterion(label_pred, label_true)
            aggregate_classify_loss += classify_loss.item()

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.item()

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            if args.range:
                diff_i = 0.0
                diff_j = 0.0
                for y_h in y_hat:
                    diff_i += torch.sum(torch.abs(y_h[:, :, :, 1:] - y_h[:, :, :, :-1]))
                    diff_j += torch.sum(torch.abs(y_h[:, :, 1:, :] - y_h[:, :, :-1, :]))
            else:
                diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
                diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = args.tv_weight*(diff_i + diff_j)
            aggregate_tv_loss += tv_loss.item()

            # total loss
            total_loss = content_loss + tv_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if (batch_num + 1) % 100 == 0:
                status = f"{time.ctime()}  Epoch {e + 1}:  " \
                         f"[{img_count}/{len(train_dataset)}]  Batch:[{batch_num+1}]  " \
                         f"agg_content: {aggregate_content_loss/(batch_num+1.0):.6f}  " \
                         f"agg_class: {aggregate_classify_loss / (batch_num + 1.0):.6f}  " \
                         f"agg_tv: {aggregate_tv_loss/(batch_num+1.0):.6f}  " \
                         f"content: {content_loss:.6f}  class: {classify_loss:.6f}  tv: {tv_loss:.6f} "
                print(status)

            if ((batch_num + 1) % 1000 == 0) and visualize is not None:
                image_transformer.eval()

                if not os.path.exists("visualization"):
                    os.makedirs("visualization")
                if not os.path.exists("visualization/%s" %log_id):
                    os.makedirs("visualization/%s" %log_id)

                for img_index in test_indices:
                    test_ds = test_dataset.dataset[int(img_index)]
                    doppler_test = torch.from_numpy(test_ds.doppler_real.values[None, None, :, :])

                    plt_path = f"visualization/{log_id}/" \
                               f"{test_ds.activity}_{test_ds.date.replace(':','-')}_e{e+1}_b{batch_num+1}.png"

                    x_test = Variable(doppler_test, requires_grad=False).type(dtype)
                    titles = ("Real data", "Synthetic data", "Generated data")
                    if args.range:
                        range_test = torch.from_numpy(test_ds.range_real.values[None, None, :, :])
                        x_test = [Variable(range_test, requires_grad=False).type(dtype), x_test]
                        range_hat, doppler_hat = image_transformer(x_test)
                        range_hat = range_hat.cpu().detach().numpy()
                        doppler_hat = doppler_hat.cpu().detach().numpy()
                        test_ds["range_gen"] = (['time', 'range'], np.squeeze(range_hat), {"units": "dB"})
                        test_ds["doppler_gen"] = (['time', 'doppler'], np.squeeze(doppler_hat), {"units": "dB"})
                        range_ds = test_ds[["range_real", "range_synth", "range_gen"]]
                        doppler_ds = test_ds[["doppler_real", "doppler_synth", "doppler_gen"]]
                        fig, axes = plt.subplots(3, 2, figsize=(11, 6))
                        spec_plot(range_ds, axes=[ax[0] for ax in axes], vmin=-40, vmax=0, add_colorbar=False)
                        spec_plot(doppler_ds, axes=[ax[1] for ax in axes], vmin=-40, vmax=0, add_colorbar=False)
                        for ax_pair, title in zip(axes, titles):
                            if title != titles[-1]:
                                for ax in ax_pair:
                                    ax.axes.get_xaxis().set_visible(False)
                            ax_pair[0].set_title(title)
                        cbar = fig.colorbar(axes[0][0].get_images()[0], ax=axes, orientation='vertical')
                        cbar.set_label('Amplitude [dB]')
                    else:
                        y_hat_test = image_transformer(x_test).cpu().detach().numpy()
                        test_ds["doppler_gen"] = (['time', 'doppler'], np.squeeze(y_hat_test), {"units": "dB"})
                        spec_plot(test_ds, vmin=-40, vmax=0, cbar_global="Amplitude [dB]")

                        axes = plt.gcf().axes
                        for ax, title in zip(axes, titles):
                            if title != titles[-1]:
                                ax.axes.get_xaxis().set_visible(False)
                            ax.set_title(title)

                    plt.savefig(plt_path)
                    plt.close()

                print("images saved")
                image_transformer.train()

        loss_logs.append({'content': aggregate_content_loss/(batch_num+1.0),
                          'class': aggregate_classify_loss/(batch_num+1.0),
                          'tv': aggregate_tv_loss/(batch_num+1.0)})

    # save model
    image_transformer.eval()

    if use_cuda:
        image_transformer.cpu()

    with open(f"log/{args.log}_loss.json", "w") as wf:
        json.dump(loss_logs, wf, indent=4)

    if args.plot:
        content_loss = [log['content'] for log in loss_logs]
        class_loss = [log['class'] for log in loss_logs]
        fig, [ax_content, ax_class] = plt.subplots(2, 1)

        ax_content.plot(content_loss)
        ax_content.set_xlabel("Epochs")
        ax_content.set_ylabel("Content loss")

        ax_class.plot(class_loss)
        ax_class.set_xlabel("Epochs")
        ax_class.set_ylabel("Classification loss")

        plt.savefig(f"log/{args.log}.png")

    filename = "models/" + args.log + ".model"
    log_dir = os.path.dirname(filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    torch.save(image_transformer.state_dict(), filename)

    if use_cuda:
        image_transformer.cuda()


def train_classify(args):
    # GPU enabling
    if args.gpu is None:
        use_cuda = False
        in_type = torch.FloatTensor
        out_type = torch.LongTensor
        print("No GPU training")
    else:
        use_cuda = True
        in_type = torch.cuda.FloatTensor
        out_type = torch.cuda.LongTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" % torch.cuda.current_device())

    # get training dataset
    config = args.config
    data_path = args.dataset
    print(f"Training with configurations {', '.join(config)} from the dataset under {data_path}")

    epochs = args.epochs

    num_workers = 4

    if args.no_split:
        train_load = 'fake-load'
        split = None
    elif args.recordings is None:
        split = 'single'
        train_load = 0.8
    else:
        split = 'no-cut'
        train_load = 0.5

    recordings = open_recordings(config, data_path, load=True, range_length=range_length, doppler_length=doppler_length)
    # Merge recordings from all configs
    train_recordings = [r for recs in recordings.values() for r in recs]
    if not args.range:  # drop range spectrograms
        train_recordings = [r.drop_vars("range_spect") for r in train_recordings]

    slice_kwargs = dict(spec_length=time_length, stride=hop_length, train_load=train_load, copy_split=0)
    loader_kwargs = dict(batch_size=BATCH_CLASSIFY, shuffle=True, num_workers=num_workers, pin_memory=True)

    print("Preloading datasets...")
    if args.no_split:
        train_dataset = slice_datasets(train_recordings, split=split, **slice_kwargs)
        test_dataset = train_dataset
        tgt_segments = {}
    elif args.recordings is None or args.recordings == 'first':
        [train_dataset, test_dataset], tgt_segments = slice_datasets(train_recordings, split=split, **slice_kwargs)
    elif args.recordings == 'last':
        [test_dataset, train_dataset], tgt_segments = slice_datasets(train_recordings, split=split, **slice_kwargs)
    else:
        raise ValueError(f"Unexpected recordings argument {args.recordings}")
    with open(f"log/{args.log}_segments.json", "w") as wf:
        json.dump(tgt_segments, wf, indent=4)

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    class_num = train_dataset.class_num
    input_shapes = train_dataset.feature_shapes
    print(f"Number of classes: {class_num}")
    print(f"Feature shapes: {input_shapes}\n")

    # define network
    c_net = RDNet(input_shapes=input_shapes, class_num=class_num).type(in_type)

    optimizer = Adam(c_net.parameters(), LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    loss_logs = []

    early_stop_thresh = .96

    for e in range(epochs):

        # track values for...
        img_count = 0
        train_loss = 0.0
        train_acc = 0.0
        test_acc = 0.0
        batch_num = 0

        # train network
        c_net.train()
        for batch_num, batch in enumerate(train_loader):
            feature_batch, label_batch = batch
            if args.range:
                x = [Variable(feat).type(in_type) for feat in feature_batch]
            else:
                x = Variable(feature_batch[0]).type(in_type)

            img_batch_read = len(label_batch)
            img_count += img_batch_read

            # zero out gradients
            optimizer.zero_grad()

            # input batch to classifier network
            y_true = Variable(label_batch).type(out_type)
            y_pred = c_net(x)

            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            train_acc += evaluate(c_net, [batch]).item() * img_batch_read
            if (batch_num + 1) % 100 == 0:
                test_acc = evaluate(c_net, test_loader).item()
                status = f"{time.ctime()}  Epoch {e + 1}:  " \
                         f"[{img_count}/{len(train_dataset)}]  Batch:[{batch_num + 1}]  " \
                         f"train_loss: {train_loss / (batch_num + 1.0):.6f}  " \
                         f"train_acc: {train_acc / img_count:.6f}  test_acc: {test_acc:.6f}"
                print(status)

        loss_logs.append({'train_loss': train_loss / (batch_num + 1.0),
                          'train_acc': train_acc / img_count, 'test_acc': test_acc})

        if train_acc / img_count > early_stop_thresh:
            print("***Early stopping training***")
            break

    # save model
    c_net.eval()

    if use_cuda:
        c_net.cpu()

    with open(f"log/{args.log}_loss.json", "w") as wf:
        json.dump(loss_logs, wf, indent=4)

    if args.plot:
        loss = []
        train_acc = []
        test_acc = []
        for log in loss_logs:
            loss.append(log['train_loss'])
            train_acc.append(log['train_acc'])
            test_acc.append(log['test_acc'])
        fig, [ax_loss, ax_acc] = plt.subplots(2, 1)
        ax_loss.plot(loss)
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Train loss")

        ax_acc.plot(train_acc)
        ax_acc.plot(test_acc)
        ax_acc.set_xlabel("Epochs")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend(["train", "test"])
        plt.savefig(f"log/{args.log}.png")

    filename = "models/" + args.log + ".model"
    log_dir = os.path.dirname(filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    torch.save(c_net.state_dict(), filename)

    if use_cuda:
        c_net.cuda()

    return loss_logs[-1]["train_acc"]


def test(args):
    # GPU enabling
    if args.gpu is None:
        dtype = torch.FloatTensor
        print("No GPU in use")
    else:
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" % torch.cuda.current_device())

    # get training dataset
    config = args.config
    data_path = args.dataset

    train_load = 0.5
    num_workers = 4

    recordings = open_recordings(config, data_path, load=True, range_length=range_length, doppler_length=doppler_length)
    # Merge recordings from all configs
    recordings = [r for recs in recordings.values() for r in recs]
    if not args.range:  # drop range spectrograms
        recordings = [r.drop_vars("range_spect") for r in recordings]

    slice_kwargs = dict(spec_length=time_length, stride=hop_length, train_load=train_load)
    loader_kwargs = dict(batch_size=BATCH_TRANSFER, shuffle=False, num_workers=num_workers, pin_memory=True)

    print("Preloading datasets...")
    if args.recordings is None:
        if args.segments is None:
            test_dataset = slice_datasets(recordings, split=None, **slice_kwargs)
        else:
            with open(args.segments, "r") as f:
                segments = json.load(f)
            _, test_dataset = slice_datasets(recordings, split=segments, **slice_kwargs)
    elif args.recordings == 'first':
        [test_dataset, _], tgt_segments = slice_datasets(recordings, split='no-cut', **slice_kwargs)
    elif args.recordings == 'last':
        [_, test_dataset], tgt_segments = slice_datasets(recordings, split='no-cut', **slice_kwargs)
    else:
        raise ValueError(f"Unrecognized recordings option {args.recordings}")

    test_loader = DataLoader(test_dataset, **loader_kwargs)

    class_num = test_dataset.class_num
    input_shapes = test_dataset.feature_shapes
    print(f"Number of classes: {class_num}")
    print(f"Feature shapes: {input_shapes}\n")

    # load network including transformer and classifier
    if args.transformer is None:
        trans_c_net = RDNet(input_shapes=input_shapes, class_num=class_num).type(dtype)
        trans_c_net.load_state_dict(torch.load(args.classifier))
    else:
        trans_c_net = RACPIT(trans_path=args.transformer, model_path=args.classifier,
                             input_shapes=input_shapes, class_num=class_num).type(dtype)
    trans_c_net.eval()

    accuracy, predict_info = evaluate(trans_c_net, test_loader, predict_info=True)
    accuracy = accuracy.item()
    print(f"{accuracy:.6f} accuracy on configurations {', '.join(config)} from {data_path}")

    true_labels = predict_info["real"]
    predictions = predict_info["predict"]
    correct = predict_info["correct"].cpu().detach().numpy().astype(np.bool)
    misclassified = np.nonzero(np.logical_not(correct))[0]

    confusion_matrix(predictions, true_labels, nb_classes=class_num)

    visualize = args.visualize

    if visualize <= 0:
        return accuracy

    if not os.path.exists("visualization/%s" % args.log):
        os.makedirs("visualization/%s" % args.log)

    true_labels = true_labels.cpu().detach().numpy().astype(np.int)
    predictions = predictions.cpu().detach().numpy().astype(np.int)

    test_indices = np.random.choice(misclassified, size=visualize, replace=False)
    activities = test_dataset.attrs['activities']

    for img_index in test_indices:
        test_ds = test_dataset.dataset[int(img_index)]
        doppler_test = torch.from_numpy(test_ds.doppler_spect.values[None, None, :, :])

        true_activity = activities[true_labels[img_index]]
        pred_activity = activities[predictions[img_index]]

        assert true_activity != pred_activity, "Prediction is correct"
        assert true_activity == test_ds.activity, "The true activity does not coincide with the embedded activity"

        plt_path = f"visualization/{args.log}/" \
                   f"{true_activity}_{test_ds.date.replace(':', '-')}_{pred_activity}.png"

        x_test = Variable(doppler_test, requires_grad=False).type(dtype)
        titles = ("Real data", f"Generated data, classified as {pred_activity}")
        if args.range:
            range_test = torch.from_numpy(test_ds.range_spect.values[None, None, :, :])
            x_test = [Variable(range_test, requires_grad=False).type(dtype), x_test]
            range_hat, doppler_hat = trans_c_net.transformer(x_test)
            range_hat = range_hat.cpu().detach().numpy()
            doppler_hat = doppler_hat.cpu().detach().numpy()
            test_ds["range_gen"] = (['time', 'range'], np.squeeze(range_hat), {"units": "dB"})
            test_ds["doppler_gen"] = (['time', 'doppler'], np.squeeze(doppler_hat), {"units": "dB"})
            range_ds = test_ds[["range_spect", "range_gen"]]
            doppler_ds = test_ds[["doppler_spect", "doppler_gen"]]
            fig, axes = plt.subplots(2, 2, figsize=(11, 6))
            spec_plot(range_ds, axes=[ax[0] for ax in axes], vmin=-40, vmax=0, add_colorbar=False)
            spec_plot(doppler_ds, axes=[ax[1] for ax in axes], vmin=-40, vmax=0, add_colorbar=False)
            for ax_pair, title in zip(axes, titles):
                if title != titles[-1]:
                    for ax in ax_pair:
                        ax.axes.get_xaxis().set_visible(False)
                ax_pair[0].set_title(title)
            cbar = fig.colorbar(axes[0][0].get_images()[0], ax=axes, orientation='vertical')
            cbar.set_label('Amplitude [dB]')
        else:
            y_hat_test = trans_c_net.transformer(x_test).cpu().detach().numpy()
            test_ds["doppler_gen"] = (['time', 'doppler'], np.squeeze(y_hat_test), {"units": "dB"})
            spec_plot(test_ds, vmin=-40, vmax=0, cbar_global="Amplitude [dB]")

            axes = plt.gcf().axes
            for ax, title in zip(axes, titles):
                if title != titles[-1]:
                    ax.axes.get_xaxis().set_visible(False)
                ax.set_title(title)

        plt.savefig(plt_path)
        plt.close()

    return accuracy


def slice_datasets(recordings, spec_length, stride, train_load=0.8, split=None, copy_split=0):
    if copy_split > 1:
        effective_len = len(recordings) // copy_split
    else:
        effective_len = len(recordings)
    if split is None:
        slices = train_test_slice(recordings, spec_length, stride, train_load, split=split)
        rd_dataset = RadarDataset(recordings, slices=slices, ignore_dims=ignore_dims)
        return rd_dataset
    elif isinstance(split, dict):
        slices = train_test_slice(recordings[:effective_len], spec_length, stride, train_load, verbose=False,
                                  split=split, return_segments=False)
        if copy_split > 1:
            slices = [copy_slices(sl, effective_len, copy_split) for sl in slices]
        rd_datasets = [RadarDataset(recordings, slices=s, ignore_dims=ignore_dims) for s in slices]
        # assert set(index for index, sl in slices[0]) == set(range(len(recordings))), \
        #    "Slices do not include all recordings"
        return rd_datasets
    else:
        slices = train_test_slice(recordings[:effective_len], spec_length, stride, train_load, verbose=False,
                                  split=split, return_segments=True)
        segments = slices.pop(-1)
        if copy_split > 1:
            slices = [copy_slices(sl, effective_len, copy_split) for sl in slices]
        rd_datasets = [RadarDataset(recordings, slices=s, ignore_dims=ignore_dims) for s in slices]
        # assert set(index for index, sl in slices[0]) == set(range(len(recordings))), \
        return rd_datasets, segments


def copy_slices(slices, num_recs, repeat):
    new_slices = []
    for n in range(repeat):
        new_slices += [(index + num_recs * n, sl) for index, sl in slices]
    return new_slices


# ============== eval
def evaluate(model_instance, input_loader, gpu=True, predict_info=False):
    ori_train_state = model_instance.training
    model_instance.eval()
    first_test = True
    all_probs, all_labels = None, None

    for data in input_loader:
        inputs = data[0]
        labels = data[1]
        if gpu:
            inputs = [inp.cuda() for inp in inputs]
            labels = labels.cuda()

        probabilities = model_instance.predict(inputs)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    predict = torch.squeeze(predict).float()
    correct = predict == all_labels
    accuracy = torch.sum(correct) / float(all_labels.size()[0])

    model_instance.train(ori_train_state)

    if predict_info:
        predictions = {"predict": predict, "real": all_labels, "correct": correct}
        return accuracy, predictions
    else:
        return accuracy


def confusion_matrix(predicted, true_label, nb_classes):
    cm = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for t, p in zip(true_label.view(-1), predicted.view(-1)):
            cm[t.long(), p.long()] += 1
    print(cm)


def save_params(parameters):
    log_file = f"log/{parameters['log']}_params.json"
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_file, "w") as wf:
        json.dump(parameters, wf, indent=4)


def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')
    parser.add_argument("--log", type=str, default=None, help="ID to mark output files and logs. Default to timestamp")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train-transfer", help="train a model to do style transfer")
    train_parser.add_argument("--plot", action='store_true', help="Plot and save a training report")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs for training")
    train_parser.add_argument("--tv-weight", type=float, default=TV_WEIGHT, help="Weight of TV regularization")
    train_parser.add_argument("--model", type=str, default=None, help="Path to a saved model to use "
                                                                      "for perceptual loss. VGG16 as default.")
    train_parser.add_argument("--recordings", type=str, default=None, help="Select to use 'first' or 'last' recordings")
    train_parser.add_argument("--segments", type=str, default=None, help="path to a segment file")
    train_parser.add_argument("--visualize", type=int, default=None, help="Set to 1 if you want to visualize training")
    train_parser.add_argument("--input", type=str, default=REAL_PATH, help="Path to input training dataset")
    train_parser.add_argument("--output", type=str, default=SYNTH_PATH, help="Path to output training dataset")
    train_parser.add_argument("--config", type=str, nargs='*', default=["F"], help="Radar configurations to train with")
    train_parser.add_argument("--range", action='store_true', help="Use range information alongside doppler")
    train_parser.add_argument("--classes", type=int, nargs='*', default=None,
                              help="Classes to train with, default to all")

    classify_parser = subparsers.add_parser("train-classify", help="train a model to classify human activity")
    classify_parser.add_argument("--plot", action='store_true', help="Plot and save a training report")
    classify_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    classify_parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs for training")
    classify_parser.add_argument("--min-acc", type=float, default=0.0, help="Retrain until min. accuracy is reached")
    classify_parser.add_argument("--recordings", type=str, default=None,
                                 help="Select to use 'first' or 'last' recordings")
    classify_parser.add_argument("--dataset", type=str, default=SYNTH_PATH, help="Path to training dataset")
    classify_parser.add_argument("--config", type=str, nargs='*', default=["E", "F"],
                                 help="Radar configurations to train with")
    classify_parser.add_argument("--range", action='store_true', help="Use range information alongside doppler")
    classify_parser.add_argument("--no-split", action='store_true', help="Do not split data into test/train")

    test_parser = subparsers.add_parser("test", help="test a model to apply human activity classification")
    test_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    test_parser.add_argument("--visualize", type=int, default=0,
                             help="Number of misclassified spectrograms to show")
    test_parser.add_argument("--transformer", type=str, default=None, help="Path to a saved model to use "
                                                                           "for image transform")
    test_parser.add_argument("--classifier", type=str, required=True, help="Path to a saved model to use "
                                                                           "for classification.")
    test_parser.add_argument("--recordings", type=str, default=None, help="Select to use 'first' or 'last' recordings")
    test_parser.add_argument("--segments", type=str, default=None, help="path to a segment file")
    test_parser.add_argument("--dataset", type=str, default=REAL_PATH, help="Path to the dataset "
                                                                            "to feed the transformer")
    test_parser.add_argument("--config", type=str, nargs='*', default=["E"], help="Radar configurations to test with")
    test_parser.add_argument("--range", action='store_true', help="Use range information alongside doppler")

    args = parser.parse_args()

    params = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"Process started at {params['timestamp']}")
    if args.log is None:
        args.log = params["timestamp"].replace(':', '')
    params.update(vars(args))
    save_params(params)

    # command
    if args.subcommand == "train-transfer":
        print("Training image transfer!")
        train_transfer(args)
    elif args.subcommand == "train-classify":
        print("Training classifier!")
        train_classify(args)
        acc = train_classify(args)
        while acc < args.min_acc:
            acc = train_classify(args)
    elif args.subcommand == "test":
        print("Testing!")
        acc = test(args)
        params["accuracy"] = acc
        save_params(params)
    else:
        print("invalid command")


if __name__ == '__main__':
    main()
