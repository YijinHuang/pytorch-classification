import sys
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, '..')
from utils.func import load_config
from modules.builder import generate_model


def define_preprocess(cfg):
    input_size = cfg.data.input_size
    mean = cfg.data.mean
    std = cfg.data.std
    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return preprocess


def predict(model, preprocess, device, imgs):
    model.eval()
    torch.set_grad_enabled(False)

    preds = []
    for img in tqdm(imgs):
        img = Image.open(img).convert('RGB')
        img = preprocess(img).unsqueeze(0).to(device)
        preds.append(model(img).detach())

    return preds


def print_weights(model):
    weights = model.state_dict()
    for var_name in weights:
        print(var_name, "\t", weights[var_name])


if __name__ == "__main__":
    # specify train.checkpoint (trained weights to load) in the config file
    config_file = '../configs/default.yaml'
    cfg = load_config(config_file)
    device = 'cuda'

    imgs = ['img1.jpg', 'img2.jpg']
    model = generate_model(cfg)
    preprocess = define_preprocess(cfg)
    result = predict(model, preprocess, device, imgs)
    print(result)
