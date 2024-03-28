import os
import numpy as np

from PIL import Image
import csv
import onnxruntime
from .wordcloud2frame import WordCloud2Frame


class CustomTransforms:
    def __init__(self, mean, std, resize_size=256, crop_size=224):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.resize_size = resize_size
        self.crop_size = crop_size

    def __call__(self, img):
        # Resize
        img = img.resize((self.resize_size, self.resize_size), Image.BILINEAR)

        # CenterCrop
        width, height = img.size
        left = (width - self.crop_size) / 2
        top = (height - self.crop_size) / 2
        right = (width + self.crop_size) / 2
        bottom = (height + self.crop_size) / 2
        img = img.crop((left, top, right, bottom))

        # ToTensor
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1)) / 255.0

        # Normalize
        img = (img - self.mean[:, None, None]) / self.std[:, None, None]

        return img

class ObjectDetection:
    def __init__(self, image_path):
        self.image_path = image_path
        self.transform = CustomTransforms(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )

    def make_model(self):
        model_dir = os.path.join(os.path.dirname(__file__), "../../models/resnet50/resnet50.onnx")
        model = onnxruntime.InferenceSession(model_dir)
        # if torch.cuda.is_available():
        #     model.cuda()
        return model

    def object_detection(self):
        model = self.make_model()
        if self.image_path is None or self.image_path == '':
            return

        file_list = os.listdir(self.image_path+"/frame/")
        framelist = []

        with open('./src/algorithms/imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]

        for file_name in file_list:
            if os.path.splitext(file_name)[-1] in ['.jpg', '.png', '.bmp']:
                img_path = self.image_path+"/frame/"+file_name
                img_t = self.transform(Image.open(img_path))

                # if torch.cuda.is_available():
                #     batch_t = torch.unsqueeze(img_t, 0).cuda()
                # else:
                #     batch_t = torch.unsqueeze(img_t, 0)
                batch_t = np.expand_dims(img_t, axis=0)
                out = model.run(["output"], {"input": batch_t})[0]
                indices = np.argsort(out[0])[::-1]  # 按概率降序排列
                percentage = np.exp(out) / np.sum(np.exp(out), axis=1) * 100

                for idx in indices[:10]:
                    frame_id = file_name[5:-4]
                    framelist.append([frame_id, (classes[idx], percentage[0][idx].item())[0]])

        self.object_detection_csv(framelist, self.image_path)

    def object_detection_csv(self, framelist, save_path):
        csv_file = open(os.path.join(save_path, 'objects.csv'), "w+", newline='')
        name = ['FrameId', 'Top1-Objects']

        try:
            writer = csv.writer(csv_file)
            writer.writerow(name)

            for i in range(len(framelist)):
                datarow = [framelist[i][0]]
                datarow.append(framelist[i][1])
                writer.writerow(datarow)
        finally:
            csv_file.close()

        wc2f = WordCloud2Frame()
        tf = wc2f.wordfrequency(os.path.join(save_path, 'objects.csv'))
        wc2f.plotwordcloud(tf, save_path, "/objects")
