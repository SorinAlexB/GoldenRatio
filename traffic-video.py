from imageai.Detection import VideoObjectDetection
import os
className={
    1:'person',3:'car',2:'bicycle',4:'motorcycle',6:'bus',8:'truck'
}

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "video.mp4"), output_file_path=os.path.join(execution_path, "traffic_detected"), frames_per_second=20, log_progress=True)
print(video_path)