import os, cv2, torch, numpy as np
from glob import glob
from tqdm import tqdm
from torch.autograd import Variable
from models import SimpleGenerator, U2NET, normPRED

class VideoMaker():
    
    def __init__(self, video_path, frame_save_path, sketch_save_path, device, video_save_path, model_name, save_name):
        
        self.video_path, self.frame_save_path, self.sketch_save_path, self.device = video_path, frame_save_path, sketch_save_path, device
        self.video_save_path, self.model_name, self.save_name = video_save_path, model_name, save_name
        
    def get_video_frames(self):

        if os.path.isdir(self.frame_save_path): print("Video frames are already obtained! Skipping this step..."); pass
        else: 
            # Create directory to save images
            os.makedirs(self.frame_save_path, exist_ok = True)
            # Video capture
            vidcap = cv2.VideoCapture(f"{self.video_path}")
            # Read the capture
            suc, im = vidcap.read()

            count = 0
            print("Obtaining frames from the video...")
            while suc:
                # Save the image
                cv2.imwrite(f"{self.frame_save_path}/frame{count}.jpg", im) 
                # Read the next capture
                suc, im = vidcap.read()
                count += 1

            print(f"Total number of frames -> {count}")

    def get_model(self):
        
        if self.model_name == "u2":
            
            checkpoint = "saved_models/u2net_portrait/u2net_portrait.pth"
            model = U2NET(3,1)
            state_dict = {}
            torch_dict_sk = torch.load(checkpoint, map_location='cpu')
            for key in torch_dict_sk:
                if key.startswith('module'):
                    state_dict[key[7:]] = torch_dict_sk[key]
                else:
                    state_dict[key] = torch_dict_sk[key]

            model.load_state_dict(state_dict)
            model.eval().to(self.device)
            
        elif self.model_name == "cartoon":
            
            checkpoint = "saved_models/weight.pth"
            weight = torch.load(checkpoint, map_location = 'cpu')
            model = SimpleGenerator()
            model.load_state_dict(weight)
            model.eval()
        
        return model
    
    def get_images(self, path):
        
        im_files = glob(f"{path}/*.jpg")
        im_files.sort(key = os.path.getctime)
        
        return im_files
    
    def preprocess_im(self, im_file):
        
        if self.model_name == "u2":
        
            im = cv2.imread(im_file)

            image = np.zeros((im.shape[0],im.shape[1],3))

            im = im/np.max(im)

            image[:,:,0] = (im[:,:,2]-0.406)/0.225
            image[:,:,1] = (im[:,:,1]-0.456)/0.224
            image[:,:,2] = (im[:,:,0]-0.485)/0.229

            # convert BGR to RGB
            image = image.transpose((2, 0, 1))
            image = image[np.newaxis,:,:,:]
            image = torch.from_numpy(image)

            # convert numpy array to torch tensor
            image = image.type(torch.FloatTensor)
            image = Variable(image.to(self.device))
        

        elif self.model_name == "cartoon":
            
            im = cv2.imread(im_file)
            image = im/127.5 - 1
            image = image.transpose(2, 0, 1)
            image = torch.tensor(image).unsqueeze(0)

        return image
    
    def get_sketch_im(self, model, im):
        
        if self.model_name == "u2":
            
            pred1,_,_,_,_,_,_ = model(im)

            # normalsization
            pred = 1.0 - pred1[:,0,:,:]
            pred = normPRED(pred)

            # convert torch tensor to numpy array
            pred = pred.squeeze()
            pred = pred.cpu().data.numpy()

            img = (pred*255).astype(np.uint8)
            #대비 조정
            a = 1
            factor = 1.0 - a*0.1
            avg = np.mean(img) / 2.0
            avg = 255*0.5

            dim = 255/(255*factor - avg)
            img = img*factor - avg #306 - 200 = 106

            img[img < 0] = 0
            img[img > 255] = 255
            img *= dim
            output = img.astype(np.uint8)

        elif self.model_name == "cartoon":
        
            output = model(im.float())
            output = output.squeeze(0).detach().numpy()
            output = output.transpose(1, 2, 0)
            output = (output + 1) * 127.5
            output = np.clip(output, 0, 255).astype(np.uint8)

        return output
        
    
    def get_sketches(self):

        if os.path.isdir(self.sketch_save_path): print("Sketch images are already obtained! Skipping this step..."); pass
        else:
            # Create directory to save images
            os.makedirs(self.sketch_save_path, exist_ok = True)

            # Get model
            model = self.get_model() 
            # Get image files
            im_files = self.get_images(self.frame_save_path)

            print("Converting images to sketches...")
            for idx, im_file in tqdm(enumerate(im_files)):
                
                # Get an image
                im = self.preprocess_im(im_file)

                # Get a sketch
                sketch = self.get_sketch_im(model, im)

                # Save the file to the folder
                cv2.imwrite(f"{self.sketch_save_path}/{os.path.basename(im_file)}", sketch)

            print("Sketches are obtained!")
            
    def create_video(self):

        os.makedirs(self.video_save_path, exist_ok = True)

        ims_array = []
        
        im_files = self.get_images(self.sketch_save_path)

        print("Creating video from the sketch frames...")
        for idx, filename in tqdm(enumerate(im_files)):
            im = cv2.imread(filename)
            height, width, layers = im.shape
            ims_array.append(im) 

        video = cv2.VideoWriter(f"{self.video_save_path}/video_{self.save_name}", cv2.VideoWriter_fourcc(*'DIVX'), 25, (ims_array[0].shape[1], ims_array[0].shape[0]))

        for idx, im in enumerate(ims_array): video.write(im)
        video.release()

        print("Video is successfully saved!")
