import cv2 as cv


class LoadWebcam:
    def __init__(self, pipe="0"):

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.2.184/1' IP camera
        # pipe = 'rtsp://username:passward@192.168.2.184/1' IP camera with login
        # pipe = 'https://www.bilibili.com/video/BV1Qo4y1X7br' IP golf camera

        self.pipe = pipe
        self.cap = cv.VideoCapture(pipe)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        return self

    def __next__(self):

        if cv.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv.destroyAllWindows()
            raise StopIteration

        # read frame
        ret, img = self.cap.read()
        return ret, img


if __name__ == '__main__':
    pipe = "https://www.bilibili.com/video/BV1Qo4y1X7br"
    webcam = LoadWebcam(pipe=pipe)
    while True:
        r, im = webcam.__next__()
        if r:
            cv.imshow('video stream', im)
        else:
            break
