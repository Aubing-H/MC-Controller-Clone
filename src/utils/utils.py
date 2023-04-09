import os
import cv2
import numpy as np

def image_desc(frame, image_length):
    rows, columns, _ = frame.shape
    if rows <= columns:
        beg = (columns - rows) // 2
        frame = frame[:, beg: beg + rows, :]
    else:
        beg = (rows - columns) // 2
        frame = frame[beg: beg + columns, :, :]
    frame = cv2.resize(frame, (image_length, image_length))
    return frame


def image2np(image):
    tp = list(range(len(image.shape)))
    tp = tp[:-3] + [tp[-1]] + tp[-3:-1]
    image = image.transpose(*tp)  # H x W x D -> D x H x W
    image = image.astype(np.float32) / 255  # uint8(0-255) -> float32(0.-1.) 
    return image  # when use cv2 image[..., ::-1] BGR -> RGB

def np2image(image):
    image = (image * 255).astype(np.uint8)  # uint8(0-255) <- float32(0.-1.) 
    tp = list(range(len(image.shape)))
    tp = tp[:-3] + tp[-2:] + [tp[-3]]
    image = image.transpose(*tp)  # H x W x D <- D x H x W
    return image  # when use cv2 image[..., ::-1] BGR <- RGB


class ImageHolder:
    ''' data shape: [H, W, BGR], dtype=np.uint8 '''
    def __init__(self) -> None:
        pass

    def save_image(self, frame, path):  # format: [H, W, BGR]
        cv2.imwrite(path, frame)
        pass

    def show_image(self, frame):  # format: [H, W, BGR]
        cv2.imshow('image', frame)
        cv2.waitKey()

    def hwc2chw(self, frame):
        return frame.transpose(2, 0, 1)
    
    def chw2hwc(self, frame):
        return frame.transpose(1, 2, 0)


class VideoHolder:
    ''' 
        write or read video frames
        data shape: [H, W, BGR], dtype=np.uint8 '''

    def __init__(self, video_path) -> None:
        self.video_path = video_path
        pass

    def init_write_frame(self, frame_height, frame_width):
        self.writer = cv2.VideoWriter(
            self.video_path, 
            cv2.VideoWriter_fourcc(*'mp4v'),
            20,  # fps
            (frame_width, frame_height))

    def write_frame(self, frame):
        try:
            self.writer.write(frame)
        except Exception as e:
            print(e)

    def read_frame(self):  # return a generator type, use it as iterator
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        while ret == True:
            yield frame
            ret, frame = cap.read()
