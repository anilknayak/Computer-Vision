# Created by anilnayak
# Creation Information
# Date 10/18/17 
# Year 2017 
# Time 3:55 AM
# For the Project TensorFlow
# created from PyCharm Community Edition

import cv2 as cv
import numpy as np
from PIL import Image
import MergeBoxes as mg
import time
class IdentifyObjects():
    def __init__(self, background_subtractor_history=50,
                 background_subtractor_varThreshold=150,
                 background_subtractor_learning_rate=0.05,
                 background_subtractor_detectShadows=True,
                 median_blur_window_size=7,
                 area_threshold=200,
                 morphology_window_size=3,
                 video_file = None,
                 draw_bounding_box_flag=False,
                 save_image_flag=True,
                 background_substraction_method="MOG",
                 video_or_camera = 1):
        if video_or_camera==99:
            self.video_capture = cv.VideoCapture(video_file)
        else:
            self.video_capture = cv.VideoCapture(video_or_camera)

        self.capture_ready_flag, self.currentframe = self.video_capture.read()

        self.gray_frame = cv.cvtColor(self.currentframe, cv.COLOR_RGB2GRAY)
        self.height, self.width, d = self.currentframe.shape
        self.average_frame = np.zeros((self.height, self.width))
        self.surface = self.width * self.height
        self.current_contour_area = 0
        self.currentcontours = None
        self.background_substraction_method = background_substraction_method
        self.draw_bounding_box_flag = draw_bounding_box_flag
        self.save_image_flag = save_image_flag
        self.background_subtractor_varThreshold = background_subtractor_varThreshold
        self.background_subtractor_history = background_subtractor_history
        self.background_subtractor_learning_rate = background_subtractor_learning_rate
        self.detectShadows = background_subtractor_detectShadows
        self.median_blur_window_size = median_blur_window_size
        self.area_threshold = area_threshold
        self.morphology_window_size = morphology_window_size
        self.change_parameters_flag = False
        self.merge_box = mg.MergeBoxes()
        self.background_substraction_method_decision()

        return None

    def background_substraction_method_decision(self):
        if self.background_substraction_method == "MOG":
            # It is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm
            # Paper: An improved adaptive background mixture model for real-time tracking with shadow detection
            # It uses a method to model each background pixel by a mixture of K Gaussian distributions (K = 3 to 5)

            self.background = cv.bgsegm.createBackgroundSubtractorMOG(history=self.background_subtractor_history)

        elif self.background_substraction_method == "MOG2":
            # It is also a Gaussian Mixture-based Background/Foreground Segmentation Algorithm
            # Paper: Improved adaptive Gausian mixture model for background subtraction
            # Paper: Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction
            # One important feature of this algorithm is that it selects the appropriate number of gaussian distribution for each pixel
            # detectShadows = True (which is so by default), it detects and marks shadows, but decreases the speed. Shadows will be marked in gray color.
            self.background = cv.createBackgroundSubtractorMOG2(history = self.background_subtractor_history,
                                                                varThreshold = self.background_subtractor_varThreshold,
                                                                detectShadows = self.detectShadows)
        elif self.background_substraction_method == "GMG":
            # This algorithm combines statistical background image estimation and per-pixel Bayesian segmentation
            # Paper: Visual Tracking of Human Visitors under Variable-Lighting Conditions for a Responsive Audio Art Installation
            # It employs probabilistic foreground segmentation algorithm that identifies possible foreground objects using Bayesian inference
            # The estimates are adaptive; newer observations are more heavily weighted than old observations to accommodate variable illumination
            # Several morphological filtering operations like closing and opening are done to remove unwanted noise. You will get a black window during first few frames
            self.background = cv.bgsegm.createBackgroundSubtractorGMG(initializationFrames = self.background_subtractor_history,
                                                                      decisionThreshold = self.background_subtractor_varThreshold)
        elif self.background_substraction_method == "CNT":
            self.background = cv.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15,
                                                                      useHistory=True,
                                                                      maxPixelStability=15 * 60,
                                                                      isParallel=True)

    def capture_background_substraction(self):
        morphology_kernel = np.ones((self.morphology_window_size,
                                     self.morphology_window_size), np.uint8)
        frame_rate = 1
        frames = -1
        carmax = 0
        carmin = 9999999999
        humanmax = 0
        humanmin = 9999999999
        while True:
            frames = frames + 1
            if frames%frame_rate == 0:
                self.capture_ready_flag, self.currentframe = self.video_capture.read()
                self.motion_rectangles_frame = np.copy(self.currentframe)
                self.blurred_frame = cv.medianBlur(self.currentframe, self.median_blur_window_size)
                self.motion_frame_mask = self.background.apply(self.blurred_frame, learningRate=.005)
                self.morphed_motion_frame_mask = cv.morphologyEx(self.motion_frame_mask,cv.MORPH_CLOSE, morphology_kernel)
                self.motion_frame_mask = cv.morphologyEx(self.motion_frame_mask,cv.MORPH_OPEN, morphology_kernel)
                _, self.currentcontours, _ = cv.findContours(np.copy(self.morphed_motion_frame_mask),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                self.current_contour_area = 0
                large_contours = []
                bounding_boxes_found = []
                boxes = []
                if self.currentcontours:
                    for contour in self.currentcontours:
                        # if cv.contourArea(contour) > self.area_threshold:
                            # large_contours.append(contour)
                        x, y, w, h = cv.boundingRect(contour)
                        bounding_boxes_found.append([x, y, w, h,0])

                boxes = self.merge_box.merge_boxes(bounding_boxes_found)
                actual_feature_boxes = []
                for box in boxes:
                    x1 = box[0]
                    y1 = box[1]
                    w1 = box[2]
                    h1 = box[3]

                    if self.merge_box.area(box) > self.area_threshold:

                        if w1 > h1*2 and (not w1 >= h1*4) and w1*h1>1000:
                            actual_feature_boxes.append([x1, y1, w1, h1, "car"])
                            a = self.merge_box.area(box)

                            if carmax < a:
                                carmax = a
                            if carmin > a:
                                carmin = a

                            if self.draw_bounding_box_flag:
                                cv.rectangle(self.currentframe, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                                cv.putText(self.currentframe, "C", (x1, y1), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (200, 200, 250), 1)
                        elif h1 > w1*2:
                            a = self.merge_box.area(box)
                            if humanmax < a:
                                humanmax = a
                            if humanmin > a:
                                humanmin = a
                            actual_feature_boxes.append([x1, y1, w1, h1, "human"])

                            if self.draw_bounding_box_flag:
                                cv.rectangle(self.currentframe, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                                cv.putText(self.currentframe, "H", (x1, y1), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (200, 200, 250), 1)

                # print("Number of Boxes : ",len(actual_feature_boxes))
                # if self.currentcontours:
                #     for contour in self.currentcontours:
                #         # how many contours
                #         if cv.contourArea(contour) > self.area_threshold:
                #             large_contours.append(contour)
                #             x, y, w, h = cv.boundingRect(contour)
                #
                #             if w > h:
                #                 boxes.append([x,y,w,h,"car"])
                #             elif h > w:
                #                 boxes.append([x, y, w, h, "human"])
                #
                #             if self.draw_bounding_box_flag:
                #                 cv.rectangle(self.currentframe, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if self.save_image_flag:
                    self.prepare_xml_and_save_files(actual_feature_boxes, self.currentframe)

                cv.imshow('Background Substraction', self.currentframe)
                # cv.drawContours(self.currentwiframe, large_contours, -1, (255, 255, 0), 3)
                # self.currentframe Has contours
                # self.motion_rectangles_frame has rectrangle Areas
            if cv.waitKey(1) & 0xFF == ord('q'):
                print("Human : ", humanmin , humanmax)
                print("Car : ", carmin, carmax)
                break

    def prepare_xml_and_save_files(self, boxes, currentframe):
        file_name = 'image_' + str(time.time()).replace(".","")
        xml_file_name = "./images/" + file_name + ".xml"
        image_file_name = file_name + '.jpg'
        height, width, depth = np.shape(currentframe)
        objects = ''
        for box in boxes:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            label = box[4]


            objects = objects + "<object>"\
                "<name>"+label+"</name>"\
                "<pose>Unspecified</pose>"\
                "<truncated>0</truncated>"\
                "<difficult>0</difficult>"\
                "<bndbox>"\
                    "<xmin>"+str(x)+"</xmin>"\
                    "<ymin>"+str(y)+"</ymin>"\
                    "<xmax>"+str(x+w)+"</xmax>"\
                    "<ymax>"+str(y+h)+"</ymax>"\
                "</bndbox>"\
            "</object>"\


        annotation = "<annotation>" \
            "<folder>images</folder>" \
            "<filename>" + image_file_name + "</filename>" \
            "<path>/images/" + image_file_name + "</path>" \
            "<source>" \
                "<database>Unknown</database>" \
            "</source>" \
            "<size>" \
                "<width>" + str(width) + "</width>" \
                "<height>" + str(height) + "</height>" \
                "<depth>" + str(depth) + "</depth>" \
            "</size>" \
            "<segmented>0</segmented>" + objects + "</annotation>"

        print("Writing Image",image_file_name)
        cv.imwrite('./images/' + image_file_name, currentframe)
        file = open(xml_file_name,"w")
        file.write(annotation)
        file.close()


if __name__ == "__main__":
    # background_substraction_method "MOG" , "MOG2" , "GMG", "CNT"
    # mog2 : th=200
    OBJIND = IdentifyObjects(area_threshold=100,
                             draw_bounding_box_flag=True,
                             save_image_flag=False,
                             background_substraction_method="MOG2",
                             background_subtractor_varThreshold=200,
                             video_or_camera=99,
                             video_file='video2.avi')
    OBJIND.capture_background_substraction()