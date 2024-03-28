import os
import numpy as np
import cv2
import onnxruntime
from src.algorithms.resultsave import resultsave

class TransNetV2:

    def __init__(self,image_sav,video_path):
        self.model = onnxruntime.InferenceSession('./models/transnetv2.onnx')
        self.image_save = image_sav
        self.v_path = video_path
    def predict_video(self, video_path: str,parent):
        cap = cv2.VideoCapture(video_path)
        all_segments = self.split_and_resize_video_frames(video_path)
        print('test',len(all_segments))

        np.savetxt("videotest.txt", all_segments, fmt="%d")

        number = getFrame_number("videotest.txt")
        number.pop()

        frame_save = self.image_save + "/frame"
        # 删除旧的分镜
        if not (os.path.exists(self.image_save)):
            os.mkdir(self.image_save)
        if not (os.path.exists(frame_save)):
            os.makedirs(frame_save)
        else:
            imgfiles = os.listdir(os.getcwd() + "/" + frame_save)
            for f in imgfiles:
                os.remove(os.getcwd() + "/" + frame_save + "/" + f)
        cap = cv2.VideoCapture(self.v_path)
        # print(cap)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # print(frame_count)
        frame_len = len(str((int)(frame_count)))
        shot_len = []
        start = 0
        # 第一帧的图片
        i = 0
        _, img1 = cap.read()
        frameid = ""
        for j in range(frame_len - len(str(i))):
            frameid = frameid + "0"
        cv2.imwrite(frame_save + "/frame" + frameid + str(i) + ".png", img1)

        # 后续的分镜图片
        _, img1 = cap.read()
        for i in number:
            i = i + 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            _, img2 = cap.read()
            j = ('%0{}d'.format(frame_len)) % i
            cv2.imwrite(frame_save + "/frame" + str(j) + ".png", img2)
            shot_len.append([start, i, i - start])
            start = i
            img1 = img2
        parent.shot_finished.emit()
        print("TransNetV2 completed")  # 把画图放进来
        rs = resultsave(self.image_save + "/")
        rs.plot_transnet_shotcut(shot_len)
        rs.diff_csv(0, shot_len)
        return 0

    def split_and_resize_video_frames(self, video_path, frame_segment_size=5000, new_size=(48, 27), frameexpect = 1):#可以每两帧选一帧
        # Load the video
        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # List to hold all segments
        all_segments = np.array([])
        # Temporary list to store frames for the current segment
        current_segment_frames = []
        # Frame skip counter
        # [[0  752]
        #  [753 2463]
        # [2465
        # 5591]]
        frame_skip_counter = 0
        end_save = 0
        # Process video
        frametmp = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # if frametmp < frameexpect:
            #     frametmp = frametmp + 1
            # else:
            #     frametmp = 0
            #     continue
            # Resize the frame
            resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            current_segment_frames.append(resized_frame)

            # If the segment has reached its maximum size, add it to the list of all segments
            if len(current_segment_frames) == frame_segment_size:
                frames_array = np.array(current_segment_frames)
                assert frames_array.shape[1:] == (27, 48, 3)

                # 转换数据类型为uint8，并添加所需的额外维度
                inputs = frames_array.astype(np.uint8)
                inputs = inputs[np.newaxis, ...]  # 在数组的开始处添加一个新的轴

                single_frame_pred, all_frame_pred = self.model.run(['single_frame_pred', 'all_frame_pred'],
                                                                   {"input": inputs})
                # 应用Sigmoid函数
                single_frame_pred = sigmoid(single_frame_pred)
                scenes = predictions_to_scenes(single_frame_pred[0])
                scenes = scenes[:-1]
                # scenes = scenes * 2
                print(scenes)
                print(scenes[-1][1])
                scenes = scenes + end_save
                end_save = scenes[-1][1] + 1
                if all_segments.size == 0:  # 如果第一个数组为空
                    all_segments = scenes
                else:
                    all_segments = np.vstack((all_segments, scenes))
                print('all_segments',all_segments)
                current_segment_frames = []  # Reset for the next segment
                cap.set(cv2.CAP_PROP_POS_FRAMES, scenes[-1][1])

        # Check if there are frames in the current segment (for the last segment)
        if current_segment_frames:
            print(len(current_segment_frames))
            frames_array = np.array(current_segment_frames)
            assert frames_array.shape[1:] == (27, 48, 3)

            # 转换数据类型为uint8，并添加所需的额外维度
            inputs = frames_array.astype(np.uint8)
            inputs = inputs[np.newaxis, ...]  # 在数组的开始处添加一个新的轴

            single_frame_pred, all_frame_pred = self.model.run(['single_frame_pred', 'all_frame_pred'],
                                                               {"input": inputs})
            # 应用Sigmoid函数
            single_frame_pred = sigmoid(single_frame_pred)
            scenes = predictions_to_scenes(single_frame_pred[0])
            # scenes = scenes * 2
            scenes = scenes + end_save
            print(scenes)
            if all_segments.size == 0:  # 如果第一个数组为空
                all_segments = scenes
            else:
                all_segments = np.vstack((all_segments, scenes))
            print(all_segments)
        # Release the video capture object
        cap.release()

        print(
            f"Video split into {len(all_segments)} segments, each with up to {frame_segment_size} frames")
        return all_segments
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getFrame_number(f_path):
    f = open(f_path, 'r')
    Frame_number = []

    i = 0
    for line in f:
        NumList = [int(n) for n in line.split()]
        Frame_number.append(NumList[1])

    print(Frame_number)
    return Frame_number

def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
    predictions = (predictions > threshold).astype(np.uint8)

    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # just fix if all predictions are 1
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)


def transNetV2_run(image_save,v_path,parent):
    import sys

    # 模型跑完了生成一个分镜帧号的txt
    model = TransNetV2(image_save,v_path)

    file = v_path
    if os.path.exists(file + ".predictions.txt") or os.path.exists(file + ".scenes.txt"):
        print(f"[TransNetV2] {file}.predictions.txt or {file}.scenes.txt already exists. "
              f"Skipping video {file}.", file=sys.stderr)
    print(file)
    model.predict_video(file,parent)
    print("TransNetV2 completed")
    return 0


