import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnet_b0 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         # transforms.ToTensor(),
         transforms.Normalize([0.485], [0.229])])

    # load image
    img_path = "/Users/chenjinsheng/Downloads/XinLong_2013_class/other"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    image_paths = [os.path.join(img_path, i) for i in os.listdir(img_path)
                   if os.path.splitext(i)[-1] in supported]
    # img = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    # img = data_transform(img)
    # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "./model-29.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    for img_path in image_paths:
        img = Image.open(img_path)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            if(predict_cla == 1):
                print(img_path)

        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        # print(print_res)

if __name__ == '__main__':
    main()
