
def get_video_frames(video_path, save_path):
    
    import os, cv2
    if os.path.isdir(save_path): print("Video frames are already obtained! Skipping this step..."); pass
    else: 
        # Create directory to save images
        os.makedirs(save_path, exist_ok = True)
        # Video capture
        vidcap = cv2.VideoCapture(f"{video_path}")
        # Read the capture
        suc, im = vidcap.read()

        count = 0
        print("Obtaining frames from the video...")
        while suc:
            # Save the image
            cv2.imwrite(f"{save_path}/frame{count}.jpg", im) 
            # Read the next capture
            suc, im = vidcap.read()
            count += 1

        print(f"Total number of frames -> {count}")


def get_sketches(path, save_path, device, checkpoint):
    
    import cv2, os, torch, numpy as np
    from glob import glob
    from tqdm import tqdm
    from torch.autograd import Variable
    from model import U2NET, normPRED
    
    if os.path.isdir(save_path): print("Sketch images are already obtained! Skipping this step..."); pass
    else:
        # Create directory to save images
        os.makedirs(save_path, exist_ok = True)

        net = U2NET(3,1)
        state_dict = {}
        torch_dict_sk = torch.load(checkpoint, map_location='cpu')
        for key in torch_dict_sk:
            if key.startswith('module'):
                state_dict[key[7:]] = torch_dict_sk[key]
            else:
                state_dict[key] = torch_dict_sk[key]

        net.load_state_dict(state_dict)
        net.eval().to(device)

        # Get image files
        im_files = glob(f"{path}/*.jpg")
        im_files.sort(key = os.path.getctime)

        print("Converting images to sketches...")
        for idx, im_file in tqdm(enumerate(im_files)):
            # input = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2GRAY)
            input = cv2.imread(im_file)

            tmpImg = np.zeros((input.shape[0],input.shape[1],3))

            input = input/np.max(input)

            tmpImg[:,:,0] = (input[:,:,2]-0.406)/0.225
            tmpImg[:,:,1] = (input[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (input[:,:,0]-0.485)/0.229

            # convert BGR to RGB
            tmpImg = tmpImg.transpose((2, 0, 1))
            tmpImg = tmpImg[np.newaxis,:,:,:]
            tmpImg = torch.from_numpy(tmpImg)

            # convert numpy array to torch tensor
            tmpImg = tmpImg.type(torch.FloatTensor)
            tmpImg = tmpImg.to(device)
        #     if torch.cuda.is_available():
        #         tmpImg = Variable(tmpImg.cuda())
        #     else:
            tmpImg = Variable(tmpImg)

            # inference
            d1,d2,d3,d4,d5,d6,d7= net(tmpImg)

            # normalization
            pred = 1.0 - d1[:,0,:,:]
            pred = normPRED(pred)

            # convert torch tensor to numpy array
            pred = pred.squeeze()
            pred = pred.cpu().data.numpy()

            del d1,d2,d3,d4,d5,d6,d7

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
            dst = img.astype(np.uint8)

            # Save the file to the folder
            cv2.imwrite(f"{save_path}/{os.path.basename(im_file)}", dst)

        print("Sketches are obtained!")
    
def create_video(path, save_path, save_name):
    
    import cv2, os, numpy as np
    from glob import glob
    from pathlib import Path
    from tqdm import tqdm

    os.makedirs(save_path, exist_ok = True)

    ims_array = []
    files = glob(f"{path}/*.jpg")
    files.sort(key = os.path.getctime)
    
    print("Creating video from the sketch frames...")
    for idx, filename in tqdm(enumerate(files)):
        im = cv2.imread(filename)
        height, width, layers = im.shape
        ims_array.append(im) 

    video = cv2.VideoWriter(f"{save_path}/video_{save_name}", cv2.VideoWriter_fourcc(*'DIVX'), 25, (ims_array[0].shape[1], ims_array[0].shape[0]))

    for idx, im in enumerate(ims_array): video.write(im)
    video.release()
    
    print("Video is successfully saved!")
