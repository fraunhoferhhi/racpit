import torch
import torch.nn as nn
from torchvision import models
from networks.img_transf import ImageTransformNet, MultiTransformNet


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        # Convert 1-channel image to 3-channel
        model = models.vgg16(pretrained=True)
        first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        first_conv_layer.extend(list(model.features))
        model.features = nn.Sequential(*first_conv_layer)
        features = model.features

        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


class RDConv(nn.Module):
    """
    Convolutional layer(s) for radar spectrograms
    """
    def __init__(self, num_branches):
        super(RDConv, self).__init__()
        self.branches = nn.ModuleList([self._branch() for _ in range(num_branches)])

    @staticmethod
    def _branch():
        conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 7), padding=(3, 3))
        relu1 = nn.ReLU()
        max1 = nn.MaxPool2d(4, 3)
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=(2, 2))
        relu2 = nn.ReLU()
        max2 = nn.MaxPool2d(3, 2)
        conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        relu3 = nn.ReLU()
        max3 = nn.MaxPool2d(3, 2)

        return nn.ModuleList([conv1, relu1, max1, conv2, relu2, max2,
                              conv3, relu3, max3])

    def _branch_forward(self, x, branch=0, flatten=True):
        for layer in self.branches[branch]:
            x = layer(x)
        if flatten:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, inputs):
        if len(self.branches) > 1 or any(isinstance(inputs, _type) for _type in (list, tuple)):
            features = []
            for i, x in enumerate(inputs):
                features.append(self._branch_forward(x, i))
            return torch.cat(features, dim=1)
        else:
            return self._branch_forward(inputs)

    def output_num(self, input_shapes):
        inputs = [torch.zeros(1, *s[1:]) for s in input_shapes]
        if len(inputs) == 1:
            inputs = inputs[0]
        y = self.forward(inputs)
        return y.size()[-1]


class RDNet(nn.Module):
    def __init__(self, input_shapes=None, class_num=5):
        super(RDNet, self).__init__()
        # set base network

        branches = len(input_shapes)
        self.conv_layers = RDConv(branches)
        features_num = self.conv_layers.output_num(input_shapes)
        self.classifier_layer_list = [nn.Linear(features_num, 16), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(16, 16), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(16, 8), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(8, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.softmax = nn.Softmax(dim=1)

        # initialization
        for dep in range(4):
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)

    def forward(self, inputs):
        features = self.conv_layers(inputs)
        outputs = self.classifier_layer(features)
        return outputs

    def predict(self, inputs):
        outputs = self.forward(inputs)
        return self.softmax(outputs)


class RDPerceptual(RDNet):
    """
    Use a trained RD classifier for perceptual loss
    """

    def __init__(self, model_path, input_shapes=None, class_num=5):
        super(RDPerceptual, self).__init__(input_shapes=input_shapes, class_num=class_num)
        self.load_state_dict(torch.load(model_path))

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        features = self.conv_layers(inputs)
        outputs = self.classifier_layer(features)
        return outputs, features


class RACPIT(RDNet):
    """
        RD classifier including image transformer
    """
    def __init__(self, trans_path, model_path, input_shapes=None, class_num=5, train_classifier=False):
        super(RACPIT, self).__init__(input_shapes=input_shapes, class_num=class_num)
        self.load_state_dict(torch.load(model_path))

        if len(input_shapes) > 1:
            self.transformer = MultiTransformNet(num_inputs=len(input_shapes), num_channels=1)
        else:
            self.transformer = ImageTransformNet(num_channels=1)
        self.transformer.load_state_dict(torch.load(trans_path))

        frozen_params = self.transformer.parameters() if train_classifier else self.parameters()
        for param in frozen_params:
            param.requires_grad = False

    def forward(self, x):
        if len(x) == 1:
            x = x[0]
        x_trans = self.transformer(x)
        return super(RACPIT, self).forward(x_trans)
