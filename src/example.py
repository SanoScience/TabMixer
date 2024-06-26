import torch

from src.models.inceptioni3d import InceptionI3d
from src.tabular_data import get_tabular_config

conf = get_tabular_config(input_shape=(1, 1, 16, 128, 128), model_name="Inception", module="TabMixer", tab_dim=6)
x = torch.randn(1, 1, 16, 128, 128)
y = torch.randn(1, 6)
model = InceptionI3d(num_classes=1, in_channels=1, tabular_config=[conf[-1]])
print(model.forward(x, y).shape)
