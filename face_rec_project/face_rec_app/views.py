import os
import shutil
import tensorflow as tf
import numpy as np
import time
import cv2
from cv2 import VideoCapture
from django.http import StreamingHttpResponse, HttpResponse
from django.template import loader
from imutils.video import FPS
from os.path import dirname, join
from .models import UploadFiles
from .forms import ImagesUpload, VideoUpload

# confidence by which we will determine good enough predictions
CONFIDENCE = 0.7

# paths for models loading
PROTOTXT_PATH = join(dirname(__file__), "../static/deploy.prototxt.txt")
FR_MODEL_PATH = join(dirname(__file__), "../static/res10_300x300_ssd_iter_140000.caffemodel")
GENDER_MODEL_PATH = join(dirname(__file__), "../static/gender_model")
AGE_MODEL_PATH = join(dirname(__file__), "../static/age_model")

# paths for images loading and saving to hard drive
IMAGES_PATH = join(dirname(__file__), "../media/images")
OUTPUT_IMAGES_PATH = join(dirname(__file__), "../media/images_out")

# paths for images loading and saving to hard drive
VIDEO_PATH = join(dirname(__file__), "../media/video")
OUTPUT_VIDEO_PATH = join(dirname(__file__), "../media/video_out")

# constant holding image file names passed for prediction
image_filenames = []
video_filename = None

# load pre-trained models
assert isinstance(tf.keras.models, object)
genderModel = tf.keras.models.load_model(GENDER_MODEL_PATH)
ageModel = tf.keras.models.load_model(AGE_MODEL_PATH)

# load classification model
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, FR_MODEL_PATH)

# frame that will be passed to html
frame = None
vc = None

# fps counter
fps = FPS()


"""
HOME PAGE
"""


def home(request):
    template = loader.get_template('index.html')
    context = {

    }
    return HttpResponse(template.render(context, request))


"""
WEBCAM STREAM
"""


def webcam_stream(request):
    template = loader.get_template('webcam_stream.html')
    context = {
        'stream_started': False
    }
    return HttpResponse(template.render(context, request))


def stream(request):
    return StreamingHttpResponse(stream_source(), content_type="multipart/x-mixed-replace;boundary=frame")


def start_stream(request):
    template = loader.get_template('webcam_stream.html')
    context = {
        'stream_started': True
    }
    return HttpResponse(template.render(context, request))


def stop_stream(request):
    global vc, fps

    # double-checking whether the stream was launched, closing the vc
    if isinstance(vc, VideoCapture):
        vc.release()
    fps.stop()
    # print the average fpg to the console
    fps_num = fps.fps() if fps.fps() > 0 else 0
    print("Average number of fps: ", fps_num)
    template = loader.get_template('webcam_stream.html')
    context = {
        'stream_started': False,
        'fps_number': fps_num
    }
    return HttpResponse(template.render(context, request))


# function which will start generating webcam stream in our website
def stream_source():
    global vc, fps
    # start the video stream
    vc = VideoCapture(0)
    # allow our camera sensor to warm up
    time.sleep(2.0)
    # check whether camera is valid
    if not vc.isOpened():
        print("Cannot open camera. Try restarting the server.")
        return
    # start counting fps
    fps.start()
    # precaution variable to exit the loop in special cases. this may not be even needed, because exiting is handled by
    # the variable 'stream_started' passed to the view's context
    exit_counter = 0
    # loop over the frames from the video stream
    while True:
        # read frame from camera
        ret, frame_in = vc.read()

        # checks whether the frame is received properly
        if not frame_in.shape or not ret:
            print("Can't get any frames. Closing in ", 5 - exit_counter)
            exit_counter += 1

        # exits when we receive at least 5 faulty frames
        if exit_counter >= 5:
            vc.release()
            return

        # output predictions
        frame_out = process_frame(frame_in)

        # update fps counter
        fps.update()

        # convert image to jpg
        (flag, encodedImage) = cv2.imencode(".jpg", frame_out)

        # then to byte array, yield the result. this is so that it may be displayed on the website
        if flag:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')


"""
VIDEO PROCESSING
"""


# function which processes the video, returns view with changed variables. the video may be later seen by the user
def process_video(request):
    # load the video from 'video' folder
    vid_name = os.listdir(VIDEO_PATH)[0]
    video_in = cv2.VideoCapture(VIDEO_PATH + "/" + vid_name)
    # define codec - mp4 by default
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output video will be the same resolution and fps, mp4 codec
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH + "/out_" + vid_name, fourcc, int(video_in.get(cv2.CAP_PROP_FPS)),
                                (int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    # check whether opening of the video that is passed to the program succeeded
    if not video_in.isOpened():
        print("Error opening the supplied video file.")
    while video_in.isOpened():
        # read the frame from the input video
        retval, frame_in = video_in.read()

        # check validity, else process frame
        if not retval:
            break
        else:
            frame_out = process_frame(frame_in)

            # write the processed frame to the output video
            out_video.write(frame_out)

    video_in.release()

    if request.method == 'POST':
        remove_files(VIDEO_PATH)
        remove_files(OUTPUT_VIDEO_PATH)
        form = VideoUpload(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    else:
        form = VideoUpload()
    template = loader.get_template('video_processing.html')
    context = {
        'form': form,
        'uploaded': True,
        'started': False
    }
    return HttpResponse(template.render(context, request))


def video_stream(request):
    return StreamingHttpResponse(play_video(), content_type="multipart/x-mixed-replace;boundary=frame")


def play_video():
    vid_name = os.listdir(VIDEO_PATH)[0]
    video_in = cv2.VideoCapture(OUTPUT_VIDEO_PATH + "/out_" + vid_name)
    if not video_in.isOpened():
        print("Error opening the supplied video file.")
    while video_in.isOpened():
        retval, frame_in = video_in.read()
        if not retval:
            break
        else:
            # Convert image to jpg
            (flag, encodedImage) = cv2.imencode(".jpg", frame_in)

            # Then to byte array
            if flag:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')
    video_in.release()


def start_displaying(request):
    if request.method == 'POST':
        remove_files(VIDEO_PATH)
        remove_files(OUTPUT_VIDEO_PATH)
        form = VideoUpload(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    else:
        form = VideoUpload()
    template = loader.get_template('video_processing.html')
    context = {
        'form': form,
        'uploaded': True,
        'started': True
    }
    return HttpResponse(template.render(context, request))


def home_vp(request):
    uploaded = False
    if request.method == 'POST':
        remove_files(VIDEO_PATH)
        remove_files(OUTPUT_VIDEO_PATH)
        form = VideoUpload(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            uploaded = True
    else:
        form = VideoUpload()
        uploaded = False
    template = loader.get_template('video_processing.html')
    context = {
        'form': form,
        'uploaded': uploaded,
        'started': False
    }
    return HttpResponse(template.render(context, request))


"""
UPLOADING IMAGES
"""


def home_ic(request):
    global image_filenames
    uploaded = False
    if request.method == 'POST':
        remove_files(IMAGES_PATH)
        remove_files(OUTPUT_IMAGES_PATH)
        image_filenames = []
        form = ImagesUpload(request.POST, request.FILES)
        files = request.FILES.getlist('images')
        if form.is_valid() and files:
            for f in files:
                file_instance = UploadFiles(images=f)
                file_instance.save()
            uploaded = True
    else:
        form = ImagesUpload()
        uploaded = False
    template = loader.get_template('image_collage.html')
    context = {
        'form': form,
        'images': image_filenames,
        'uploaded': uploaded,
        'started': False
    }
    return HttpResponse(template.render(context, request))


# process image by image, input predictions
def process_images(request):
    global image_filenames
    for file in os.listdir(IMAGES_PATH):
        image = cv2.imread(os.path.join(IMAGES_PATH, file))
        if image is not None:
            image_filenames.append("/images_out/" + "out_" + file)
            cv2.imwrite(os.path.join(OUTPUT_IMAGES_PATH, "out_" + file), process_frame(image))
        else:
            print("Could not process file: ", file)

    if request.method == 'POST':
        image_filenames = []
        remove_files(IMAGES_PATH)
        remove_files(OUTPUT_IMAGES_PATH)
        form = ImagesUpload(request.POST, request.FILES)
        files = request.FILES.getlist('images')
        if form.is_valid() and files:
            for f in files:
                file_instance = UploadFiles(images=f)
                file_instance.save()
    else:
        form = ImagesUpload()
    template = loader.get_template('image_collage.html')
    context = {
        'form': form,
        'images': image_filenames,
        'uploaded': True,
        'started': False
    }
    return HttpResponse(template.render(context, request))


def image_collage(request):
    if request.method == 'POST':
        remove_files(IMAGES_PATH)
        remove_files(OUTPUT_IMAGES_PATH)
        form = ImagesUpload(request.POST, request.FILES)
        files = request.FILES.getlist('images')
        if form.is_valid() and files:
            for f in files:
                file_instance = UploadFiles(images=f)
                file_instance.save()
    else:
        form = ImagesUpload()
    template = loader.get_template('image_collage.html')
    context = {
        'form': form,
        'images': image_filenames,
        'uploaded': True,
        'started': True
    }
    return HttpResponse(template.render(context, request))


"""
UTILITY
"""


# utility function used to remove files from specified directory
def remove_files(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# utility function used to process a frame and output predictions on it.
def process_frame(frame_in):
    (h, w) = frame_in.shape[:2]

    # convert to blob (needed for detection for the Caffe model)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_in, (300, 300), interpolation=cv2.INTER_AREA), 1.0, (300, 300), (104.0, 117.0, 123.0)
    )

    # pass for detection
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # ignore weak detections
        if confidence < CONFIDENCE:
            continue

        # rectangle box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # compute height and width of the frame
        width = endX - startX
        height = endY - startY

        # extract 224x224 and 100x100 face image that we will use to pass for the predictions,
        # normalize the frame so that the image is not distorted in any way
        try:
            crop_x = crop_y = 0
            if height > width:
                crop_y = height - width
            elif height < width:
                crop_x = width - height
            face_image_gender = cv2.resize(frame_in[int(startY + 0.8 * crop_y):int(endY - 0.2 * crop_y),
                                           int(startX + 0.5 * crop_x):int(endX - 0.5 * crop_x)], (100, 100),
                                           interpolation=cv2.INTER_AREA)
            face_image_age = cv2.resize(frame_in[int(startY + 0.8 * crop_y):int(endY - 0.2 * crop_y),
                                        int(startX + 0.5 * crop_x):int(endX - 0.5 * crop_x)], (224, 224),
                                        interpolation=cv2.INTER_AREA)
        except Exception as e:
            print("Caught exception: ", e)
            continue

        # expand frame dimensions to fit it into models
        face_image_age = np.expand_dims(face_image_age, axis=0)
        face_image_gender = np.expand_dims(face_image_gender, axis=0)

        # analyze frame in the gender model
        pred_gender = genderModel.predict(x=face_image_gender, verbose=0)[0][0]
        prob_man = (1 - pred_gender) * 100
        prob_woman = 100 - prob_man

        # analyze frame in age model
        pred_age = ageModel.predict(x=face_image_age, verbose=0)[0][0]

        # print out text
        (tFace, tScale, tColor, tThickness) = (cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1)
        text1 = "C:{certainty:.2f}%  A:{age:.2f}".format(certainty=confidence * 100, age=pred_age)
        text2 = "M:{man:.2f}% F:{woman:.2f}%".format(man=prob_man, woman=prob_woman)
        (_, offset), _ = cv2.getTextSize(text1, tFace, tScale, tThickness)
        y1 = startY - offset if startY - offset > offset else startY + offset
        y2 = endY - offset if endY + 2 * offset > frame_in.shape[0] else endY + offset

        cv2.rectangle(frame_in, (startX, startY), (endX, endY), tColor, 2)
        cv2.putText(
            frame_in, text1, (startX, y1), tFace, tScale, tColor, tThickness
        )
        cv2.putText(
            frame_in, text2, (startX, y2), tFace, tScale, tColor, tThickness
        )

    return frame_in
