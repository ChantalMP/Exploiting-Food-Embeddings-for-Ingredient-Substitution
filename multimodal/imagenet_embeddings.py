'''
Generate embeddings for all ingredients with pretrained ImageNet and save to embedding_dict.pth
'''

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


class ImageFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.eval()
        self.model.to(self.device)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def forward(self, images):
        images = torch.stack([self.transforms(image) for image in images])
        images = images.to(self.device)
        outputs = self.model(images).squeeze(3).squeeze(2)
        outputs = outputs.mean(dim=0)
        return outputs


if __name__ == '__main__':
    model = ImageFeatureExtractor()
    model.eval()
    embedding_dict = {}

    with open('foodbert_embeddings/data/used_ingredients_clean.json') as f:
        ingredients = json.load(f)

    for ingredient in tqdm(ingredients, desc='Generating Image Embeddings'):
        images_path = Path(f"multimodal/data/downloads/{ingredient}")
        image_paths = list(images_path.glob('*.jpg'))
        images = [Image.open(path) for path in image_paths]
        images = [elem for elem in images if transforms.ToTensor()(elem).shape[0] == 3][:5]
        if len(images) == 0:
            print("STOPP ", ingredient)
            continue
        embedding = model(images)
        embedding_dict[ingredient] = embedding

    with open("multimodal/data/embedding_dict.pth", "wb") as f:
        torch.save(embedding_dict, f)
