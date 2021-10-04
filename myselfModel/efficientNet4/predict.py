import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from shutil import copyfile

from model import efficientnet_b0 as create_model

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def filter_image_with_dl(src_name, dst_name):
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
         transforms.ToTensor(),
         transforms.Normalize([0.485], [0.229])])

    # load image
    # img_path = "/Users/chenjinsheng/PycharmProjects/qihuiProject/1231"

    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "./weights/model-29.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    if (os.path.exists(dst_name) == False):
        os.mkdir(dst_name)
    else:
        os.rmdir(dst_name)
        os.mkdir(dst_name)
    for root, dirs, files in os.walk(src_name):
        print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        if (len(dirs) > 0):
            for dir in dirs:
                os.mkdir(dst_name + os.sep + dir)
        # print(files)  # 当前路径下所有非目录子文件
        if (len(files) > 0):
            dst_path = dst_name + os.sep + root[-4:]
            image_paths = [os.path.join(root, i) for i in files
                           if os.path.splitext(i)[-1] in supported]
            dst_paths = [os.path.join(dst_path, i) for i in files
                         if os.path.splitext(i)[-1] in supported]
            img_open = [data_transform(Image.open(path)) for path in image_paths]
            img_tensor = torch.stack(img_open, dim=0).to(device)

            try:
                with torch.no_grad():
                    output = torch.squeeze(model(img_tensor)).cpu()
                    predict = torch.softmax(output, dim=1)
                    predict_cla = torch.argmax(predict, dim=1).numpy()
                    chose_img_index_list = np.argwhere(predict_cla == 0)
                    img_chose = np.array(image_paths)[chose_img_index_list.reshape(-1)]
                    dst_cp_path = np.array(dst_paths)[chose_img_index_list.reshape(-1)]
                    for src, dst in zip(img_chose, dst_cp_path):
                        copyfile(src, dst)
            except:
                pass


if __name__ == '__main__':
    filter_image_with_dl("/home/chenjs/nfsdata/ASA_airglow_image/XinLong_2013_cut",
                         "/home/chenjs/nfsdata/ASA_airglow_image/XinLong_2013_cut_filter")
