import os, argparse
from utils import VideoMaker
from time import time

def run(args):    

    start_time = time()
    folder_name = f"{os.path.basename(args.root).split('.')[0]}_u2"
    
    vid = VideoMaker(args.root, f"{args.frames}/{folder_name}", f"{args.sketches}/{folder_name}",
                     device = "cuda", checkpoint = args.checkpoint, video_save_path = args.save_path,
                     save_name = f"{folder_name}.mp4")
    
    vid.get_video_frames(); vid.get_sketches(); vid.create_video()
    print(f"The process is done in {(time() - start_time):.3f} seconds!")
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = "Sketch Video Create Arguments")
    
    # Add arguments to the parser
    parser.add_argument("-r", "--root", type = str, default = "potato.mp4", help = "Path to the video")
    parser.add_argument("-cp", "--checkpoint", type = str, default = "saved_models/u2net_portrait/u2net_portrait.pth", help = "Path to trained model checkpoint")
    parser.add_argument("-f", "--frames", type = str, default = "frames", help = "Path to save the frames")
    parser.add_argument("-s", "--sketches", type = str, default = "sketches", help = "Path to save the sketches")
    parser.add_argument("-sp", "--save_path", type = str, default = "videos", help = "Path to save the video")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
    
