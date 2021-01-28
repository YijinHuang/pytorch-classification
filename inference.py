import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from config import *
from modules import generate_model
from utils import select_out_features


def define_preprocess():
    input_size = DATA_CONFIG['input_size']
    mean = DATA_CONFIG['mean']
    std = DATA_CONFIG['std']
    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return preprocess


def load_model(weights_path, device):
    net_name = BASIC_CONFIG['network']
    backbone = NET_CONFIG[net_name]
    device = BASIC_CONFIG['device']
    criterion = TRAIN_CONFIG['criterion']
    num_classes = BASIC_CONFIG['num_classes']
    out_features = select_out_features(num_classes, criterion)
    model = generate_model(
        net_name,
        backbone,
        out_features,
        device,
        checkpoint=weights_path,
    )
    return model


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
    imgs = ['img1.jpg', 'img2.jpg']

    weights_path = '/path/to/weights'
    device = 'cuda:0'
    model = load_model(weights_path, device)
    preprocess = define_preprocess()
    result = predict(model, preprocess, device, imgs)
    print(result)
