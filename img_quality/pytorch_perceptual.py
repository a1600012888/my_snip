import torch
import numpy as np
from pvgg import vgg19

def ensure_torch_shape(tensor):
    if len(tensor.size()) == 3:
        return tensor.unsqueeze(dim = 0)
    return tensor

def compute_perceptual_loss(img1, img2):
    img1 = torch.tensor(img1, dtype = torch.float32).cuda()
    img2 = torch.tensor(img2, dtype=torch.float32).cuda()

    img1 = ensure_torch_shape(img1)
    img2 = ensure_torch_shape(img2)
    assert len(img1.size()) == 4
    assert len(img2.size()) == 4

    vgg = vgg19(pretrained = True)
    vgg.cuda()
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False

    contents, styles = vgg(img1, img2)

    return contents.cpu().squeeze(dim = 0), styles.cpu().squeeze(dim = 0)


def get_content(img1, img2, layer = 4):
    assert layer in [0, 1, 2, 3, 4]

    contents, _ = compute_perceptual_loss(img1, img2)

    return contents[layer].item()

def get_style(img1, img2, layer = 4):
    assert layer in [0, 1, 2, 3, 4]

    _, styles = compute_perceptual_loss(img1, img2)

    return styles[layer].item()


def test():

    img1 = np.random.randn(3, 224, 224) * 125
    img2 = np.random.randn(3, 224, 224) * 125

    style = get_style(img1, img2)
    content = get_content(img1, img2)

    print(style, content)


if __name__ == '__main__':

    test()