import numpy as np
import cv2
from ultralytics import FastSAM
import clip
import open_clip
import os
from PIL import Image
import torch
from tqdm import tqdm



class SmartAttention:
    def __init__(self, device='cuda:0', initial_dictionary_path='image_dictionary/images'):
        self.initial_dictionary_path = initial_dictionary_path
        self.fastsam = FastSAM('./FastSAM-s.pt')
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device, jit=False)
        self.device = device
        self.dictionary_features, self.dictionary_classes, self.dictionary_filenames = self.create_dictionary(dictionary_path=initial_dictionary_path)
        self.interaction_dictionary = self.create_interactions_dictionary(dictionary_path=initial_dictionary_path)
    
    def extract_clip_features(self, img):
        img = Image.fromarray(img)
        img = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(img)
        return features


    def create_dictionary(self, dictionary_path="image_dictionary/images"):
        features = []
        classes = []
        filenames = []
        for _class in os.listdir(dictionary_path):
            for image in os.listdir(os.path.join(dictionary_path, _class)):
                img = cv2.imread(os.path.join(dictionary_path, _class, image))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                features.append(self.extract_clip_features(img).cpu().numpy())
                classes.append(_class)
                filenames.append(image)

        return np.array(features).squeeze(), np.array(classes), np.array(filenames)

    def create_interactions_dictionary(self, dictionary_path="image_dictionary/images"):
        dict = {}
        for _class in os.listdir(dictionary_path):
            dict[_class] = 0
        dict["unknown"] = 0
        return dict 
    
    def update_interactions_dictionary(self, dictionary_path="image_dictionary/images"):
        for _class in os.listdir(dictionary_path):
            if _class not in self.interaction_dictionary.keys():
                self.interaction_dictionary[_class] = 0
        return self.interaction_dictionary


    def extract_regions_with_sam(self, frame, show=False):
        results = self.fastsam.track(frame, stream=False, show=show, conf=0.9, iou=0.6, mode="track", persist=True)
        #print(self.fastsam.device)
        if show:
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

        boxes = results[0].boxes.xyxy.cpu()
        """masks_from_sam = results[0].masks.cpu()
        masks = []
        for mask in masks_from_sam:
            resize_mask = cv2.resize(mask.data.numpy().astype(np.uint8).transpose(1,2,0), (frame.shape[1], frame.shape[0]))
            masks.append(resize_mask)"""
        

        return boxes
    
    def apply_smart_attention(self, frame, boxes, unknown_threshold=0.6):
        
        #unknown_images = []
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for box in boxes:
            img_box = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            #img_box_plain = np.copy(img_box)
            img_box_features = self.extract_clip_features(img_box)

            distances = np.dot(self.dictionary_features, img_box_features.cpu().numpy().T) 
            denom = (np.linalg.norm(self.dictionary_features, axis=1) * np.linalg.norm(img_box_features.cpu().numpy(), axis=1))
            distances = distances.squeeze() / denom

           
            
            if np.max(distances) < unknown_threshold:
                class_name = "unknown"
                #unknown_images.append(img_box_plain) 
            else:
                class_name = self.dictionary_classes[np.argmax(distances)]
            
            #print(np.mean(distances), np.std(distances), np.max(distances), np.min(distances), np.argmax(distances), self.dictionary_filenames[np.argmax(distances)])
            
            # draw rectangle and label
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # add the similarity as well
            cv2.putText(frame, str(np.max(distances)), (int(box[0]), int(box[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Interactions {}".format(self.interaction_dictionary[class_name]), (int(box[0]), int(box[1]) + 60),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        annotated_frame = frame

        return annotated_frame
    
       
    def find_unkown(self, frame, boxes, unknown_threshold=0.6):
        
        unknown_images = []
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for box in boxes:
            img_box = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            img_box_features = self.extract_clip_features(img_box)

            distances = np.dot(self.dictionary_features, img_box_features.cpu().numpy().T) 
            denom = (np.linalg.norm(self.dictionary_features, axis=1) * np.linalg.norm(img_box_features.cpu().numpy(), axis=1))
            distances = distances.squeeze() / denom

           
            
            if np.max(distances) < unknown_threshold:
                unknown_images.append(img_box) 

        return unknown_images
    
    def add_interactions(self, _class, interactions = 1):
        if _class in self.interaction_dictionary.keys():
            self.interaction_dictionary[_class] = self.interaction_dictionary[_class] + interactions
        else:
            self.interaction_dictionary[_class] =  interactions
    
    def add_view(self, img, class_name=None, new=False):
        if new:
            class_list = os.listdir(self.initial_dictionary_path)
            new_objects = [int(obj.split('_')[1]) for obj in class_list if 'object_' in obj]
            if new_objects == []:
                new_idx = 0
            else:
                new_idx = max(new_objects)+1

            os.mkdir(f"{self.initial_dictionary_path}/object_{new_idx}")
            cv2.imwrite(f"{self.initial_dictionary_path}/object_{new_idx}/view_0.png", img)
            print("Added an extra object!")

        else:
            views_list = os.listdir(f"{self.initial_dictionary_path}/{class_name}")
            instance = max([int(view.split("_")[1]) for view in views_list])
            cv2.imwrite(f"{self.initial_dictionary_path}/{class_name}/view_{instance}.png", img)
    
    def database_update(self):
        self.dictionary_features, self.dictionary_classes, self.dictionary_filenames = self.create_dictionary(dictionary_path=self.initial_dictionary_path)
        self.interaction_dictionary = self.update_interactions_dictionary(dictionary_path=self.initial_dictionary_path)
        print("Database up to date.")
    
    def unkown_handler(self, ):
        return

if __name__ == "__main__":
    sam = SmartAttention(device='cuda:0', initial_dictionary_path='modules/SmartAttentionModule/image_dictionary_2/images')
    frame = cv2.imread('image_dictionary_2/images/object_0/view_0.png')
    cv2.imshow('frame', frame)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
       
    boxes = sam.extract_regions_with_sam(frame, show=True)
    print(boxes)
    
    """for mask in masks:
        thing = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('frame', thing)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break"""