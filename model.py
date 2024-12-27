# common dependencies
from typing import Union
import argparse
import json
import os
from tqdm import tqdm

# 3rd party dependencies
import numpy as np
from deepface.DeepFace import find, represent
from deepface.commons import functions
from deepface.commons.functions import load_image 
from deepface.detectors import FaceDetector

class Model():
    def __init__(self,
            db_path : str = "database",
            model_name : str = "ArcFace",
            threshold : float = 0.4,
            distance_metric : str = "cosine",
            enforce_detection : bool = True,
            detector_backend : str = "retinaface",
            align : bool = True,
            normalization : str = "base",
            silent : bool = False,
    ):
        self.db_path = db_path
        self.model_name = model_name
        self.threshold = threshold
        self.distance_metric = distance_metric
        self.enforce_detection = enforce_detection
        self.detector_backend = detector_backend
        self.align = align
        self.normalization = normalization
        self.silent = silent

    def find(self, img_path : Union[str, np.ndarray]) -> list[str]:
        dfs = find(img_path, self.db_path, self.model_name, self.distance_metric, self.enforce_detection, self.detector_backend, self.align, self.normalization, self.silent)

        results = []

        if len(dfs) == 0:
            print("Can not recognize the face(s)...")
        else:
            for df, source_region in dfs:
                info = {
                    "x":source_region["x"],
                    "y":source_region["y"],
                    "w":source_region["w"],
                    "h":source_region["h"],
                    "found": False,
                    "have_infomation": False
                }

                if df.empty or df.iloc[0][-1] > self.threshold: # Empty result or distance > threshold
                    print("\nThis person is not found in database...")
                    results.append(info)
                    continue

                print("\nFound...")
                print(df.iloc[0][-1])

                info["found"] = True
                path = df.iloc[0]["identity"]
                origin_path = os.path.dirname(path)
                
                about = None
                
                with open(f"{origin_path}/about.json", 'r') as f:
                    about = json.loads(f.read())
                
                if about is not None:
                    info.update(about)

                results.append(info)
        
        return results
    
    def detect_faces(self, img):
        # img, img_name = load_image(img)
        face_detector = FaceDetector.build_model(self.detector_backend)
        face_objs = FaceDetector.detect_faces(face_detector, self.detector_backend, img, self.align)
        return face_objs

    def represent(self, imgs, imgs_paths):
        representations = []

        target_size = functions.find_target_size(model_name=self.model_name)

        # for employee in employees:
        pbar = tqdm(
            range(0, len(imgs)),
            desc="Finding representations",
        )
        for index in pbar:
            img = imgs[index]

            img_objs = functions.extract_faces(
                img=img,
                target_size=target_size,
                detector_backend=self.detector_backend,
                grayscale=False,
                enforce_detection=self.enforce_detection,
                align=self.align,
            )

            for img_content, img_region, _ in img_objs:
                embedding_obj = represent(
                    img_path=img_content,
                    model_name=self.model_name,
                    enforce_detection=self.enforce_detection,
                    detector_backend="skip",
                    align=self.align,
                    normalization=self.normalization,
                )

                img_representation = embedding_obj[0]["embedding"]

                instance = []
                instance.append(imgs_paths[index])
                instance.append(img_representation)
                instance.append(img_region["x"])
                instance.append(img_region["y"])
                instance.append(img_region["w"])
                instance.append(img_region["h"])
                representations.append(instance)

        return representations

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help='Enter image to recognize')
    opt = parser.parse_args()
    return opt

def main(opt):
    model = Model()
    model.run(opt.img_path)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
