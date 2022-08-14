import os
from typing import Union

import cv2
from matplotlib import pyplot as plt
import numpy as np
import onnxruntime as rt


def preprocess(img: Union[str, np.ndarray]):
    if type(img) == str:
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # permute
    img = img.transpose((2, 0, 1))

    # normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = img.astype(np.float32, copy=False)
    mean = np.array(mean)[:, np.newaxis, np.newaxis]
    std = np.array(std)[:, np.newaxis, np.newaxis]

    img = img / 255.0
    img -= mean
    img /= std

    return img


def argmax3d(heatmaps: np.ndarray, joint_num: int):
    # zyx
    heatmaps = heatmaps.reshape((-1, joint_num, 32, 32, 32))
    z = np.argmax(heatmaps.sum(axis=(3, 4)), axis=2)
    y = np.argmax(heatmaps.sum(axis=(2, 4)), axis=2)
    x = np.argmax(heatmaps.sum(axis=(2, 3)), axis=2)
    return np.stack([x, y, z], axis=-1)


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


if __name__ == "__main__":
    onnx_filepath = r"mobile_human_pose_baseline_256x256.onnx"
    # onnx_filepath = r"mobile_human_pose_working_well_256x256.onnx"
    sess = rt.InferenceSession(onnx_filepath)
    outputs = sess.get_outputs()
    output_names = list(map(lambda out: out.name, outputs))

    print(output_names)

    # input_img = np.random.randn(1, 3, 256, 256).astype(np.float32)
    # output = sess.run(output_names, {"input": input_img})
    # # output: list, output[0]: (1, 672, 32, 32) = (21 * 1, 1, 1) * 32 => 3D heatmap
    # print(output[0].shape)
    # print(argmax3d(output[0], 21).shape)

    skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

    joint_num = 21
    img_filepath = r"C:\Users\Bangwen\PycharmProjects\3DMPPE_POSENET_RELEASE\posenet3d\demo\0.jpg"
    original_img = cv2.imread(img_filepath)
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    input_img = preprocess(rgb_img)

    output = sess.run(None, {"input": input_img[np.newaxis, :]})
    kps_3d = argmax3d(output[0], joint_num)

    print(kps_3d.shape)
    kps_3d[..., 0] = kps_3d[..., 0] / 32 * 256
    kps_3d[..., 1] = kps_3d[..., 1] / 32 * 256

    person_num = len(kps_3d)
    vis_img = original_img.copy()
    for n in range(person_num):
        vis_kps = np.zeros((3,joint_num))
        vis_kps[0,:] = kps_3d[n][:,0]
        vis_kps[1,:] = kps_3d[n][:,1]
        vis_kps[2,:] = 1
        img_2d = vis_keypoints(vis_img, vis_kps, skeleton)
    cv2.imwrite("pose2d.jpg", img_2d)