"""
Implementation of VGG-Face model in PyTorch.
"""

from __future__ import annotations
from collections import OrderedDict
from torch import nn

class VGGFace(nn.Module):
    """
    VGGFace class.

    This is an reimplementation of the original `VGG-Face` model in `PyTorch`.

    *Source: https://github.com/chi0tzp/PyVGGFace/blob/master/lib/vggface.py.*
    """

    def __init__(self, save_layer_output: bool = False) -> None:
        """
        Initialize VGGFace model.

        :param save_layer_output: If True, save the output of each layer in a list.
        :return: None
        """
        super().__init__()

        self.save_layer_output = save_layer_output
        self.layer_output = []
        self._layer_names = []

        self.features = nn.ModuleDict(
            OrderedDict(
                {
                    # === Block 1 ===
                    "conv_1_1": nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                    "relu_1_1": nn.ReLU(inplace=True),
                    "conv_1_2": nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    "relu_1_2": nn.ReLU(inplace=True),
                    "maxp_1_2": nn.MaxPool2d(kernel_size=2, stride=2),
                    # === Block 2 ===
                    "conv_2_1": nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                    "relu_2_1": nn.ReLU(inplace=True),
                    "conv_2_2": nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                    "relu_2_2": nn.ReLU(inplace=True),
                    "maxp_2_2": nn.MaxPool2d(kernel_size=2, stride=2),
                    # === Block 3 ===
                    "conv_3_1": nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                    "relu_3_1": nn.ReLU(inplace=True),
                    "conv_3_2": nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                    "relu_3_2": nn.ReLU(inplace=True),
                    "conv_3_3": nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                    "relu_3_3": nn.ReLU(inplace=True),
                    "maxp_3_3": nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                    # === Block 4 ===
                    "conv_4_1": nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                    "relu_4_1": nn.ReLU(inplace=True),
                    "conv_4_2": nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    "relu_4_2": nn.ReLU(inplace=True),
                    "conv_4_3": nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    "relu_4_3": nn.ReLU(inplace=True),
                    "maxp_4_3": nn.MaxPool2d(kernel_size=2, stride=2),
                    # === Block 5 ===
                    "conv_5_1": nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    "relu_5_1": nn.ReLU(inplace=True),
                    "conv_5_2": nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    "relu_5_2": nn.ReLU(inplace=True),
                    "conv_5_3": nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                    "relu_5_3": nn.ReLU(inplace=True),
                    "maxp_5_3": nn.MaxPool2d(kernel_size=2, stride=2),
                }
            )
        )

        self.fc = nn.ModuleDict(
            OrderedDict(
                {
                    "fc6": nn.Linear(in_features=512 * 7 * 7, out_features=4096),
                    "fc6-relu": nn.ReLU(inplace=True),
                    "fc6-dropout": nn.Dropout(p=0.5),
                    "fc7": nn.Linear(in_features=4096, out_features=4096),
                    "fc7-relu": nn.ReLU(inplace=True),
                    "fc7-dropout": nn.Dropout(p=0.5),
                    "fc8": nn.Linear(in_features=4096, out_features=2622),
                }
            )
        )

    def reset_layer_output(self):
        """Reset the layer output list (i.e., set it to an empty list)."""
        self.layer_output = []

    @property
    def layer_names(self):
        """Return list of layer names in `VGGFace`."""
        if self._layer_names:
            return self._layer_names

        for child in self.children():
            for layer in child:
                self._layer_names.append(str(layer))
        return self._layer_names

    def forward(self, x):
        """Run forward pass through the model `VGGFace`."""
        if self.save_layer_output:
            self.reset_layer_output()

        # Forward through feature layers
        for layer in self.features.values():
            x = layer(x)
            # Append layer output to list
            if self.save_layer_output:
                self.layer_output.append(x)

        # Flatten convolution outputs
        x = x.view(x.size(0), -1)

        # Forward through FC layers
        if hasattr(self, "fc"):  # check for later cut-off models
            for layer in self.fc.values():
                x = layer(x)
                if self.save_layer_output:
                    self.layer_output.append(x)

        return x


# Rebuild a new model that ends at fc7
class VGGFaceTruncated(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        self.features = vgg_model.features
        self.fc = nn.Sequential(
            vgg_model.fc["fc6"],
            vgg_model.fc["fc6-relu"],
            vgg_model.fc["fc6-dropout"],
            vgg_model.fc["fc7"],
            vgg_model.fc["fc7-relu"],
        )

    def forward(self, x):
        for layer in self.features.values():
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x