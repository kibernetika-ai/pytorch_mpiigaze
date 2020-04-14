import sys
import os
from gaze_estimation import get_default_config
from gaze_estimation import GazeEstimationMethod, GazeEstimator
from gaze_estimation.gaze_estimator.common import (Face, FacePartsName,
                                                   Visualizer)

import cv2
import numpy as np
from ml_serving.utils import helpers
import logging

LOG = logging.getLogger(__name__)

class GazeModel:
    def __init__(self, **params):
        self.cfg = get_default_config()
        self.cfg.merge_from_file(os.path.join(params.get('model'),'config.yaml'))
        self.cfg.merge_from_list(['face_detector.dlib.model',
                                  os.path.join(os.environ['DLIB_FACE_DIR'],'shape_predictor_68_face_landmarks.dat')])
        self.cfg.merge_from_list(['gaze_estimator.checkpoint', os.path.join(params.get('model'),'checkpoint.pth')])
        self.cfg['gaze_estimator']['normalized_camera_params'] = os.path.join(params.get('model'),'normalized_camera_params_eye.yaml')
        self.cfg['gaze_estimator']['camera_params'] = os.path.join(params.get('model'),'sample_params.yaml')
        self.cfg['device'] = 'cpu'
        self.gaze_estimator = GazeEstimator(self.cfg)
        self.visualizer = Visualizer(self.gaze_estimator.camera)

    def _draw_face_bbox(self,face: Face) -> None:
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self,face: Face) -> None:
        length = self.cfg.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)
        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)

    def _draw_landmarks(self,face: Face) -> None:
        self.visualizer.draw_points(face.landmarks,
                               color=(0, 255, 255),
                               size=1)

    def _draw_face_template_model(self,face: Face) -> None:
        self.visualizer.draw_3d_points(face.model3d,
                                  color=(255, 0, 525),
                                  size=1)

    def _draw_gaze_vector(self,face: Face) -> None:
        length = self.cfg.demo.gaze_visualization_length
        if self.cfg.mode == GazeEstimationMethod.MPIIGaze.name:
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
        elif self.cfg.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))

        else:
            raise ValueError

    def process(self,frame):
        if frame.shape[2]>3:
            frame = frame[:,:,0:3]
        frame = cv2.resize(frame, (640, 480))
        undistorted = cv2.undistort(
            frame, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)
        self.visualizer.set_image(frame.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)
        return self.visualizer.image

def init_hook(**params):
    LOG.info('Loaded. {}'.format(params))
    return GazeModel(**params)

def process(inputs, ctx, **kwargs):
    img, is_video = helpers.load_image(inputs, 'image', rgb=False)
    img = ctx.global_ctx.process(img)
    img = img[:, :, ::-1]
    if not is_video:
        img = cv2.imencode('.jpg', img)[1].tostring()
    return {'output': img}