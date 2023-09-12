import os, argparse
from utils import VideoMaker
from time import time

def run(args):    

    start_time = time()
    folder_name = f"{os.path.basename(args.root).split('.')[0]}_{args.model_name}"
    
    vid = VideoMaker(args.root, f"{args.frames}/{folder_name}", f"{args.sketches}/{folder_name}",
                     device = "cuda", video_save_path = args.save_path,
                     model_name = args.model_name, save_name = f"{folder_name}.mp4")
    
    vid.get_video_frames(); vid.get_sketches(); vid.create_video()
    print(f"The process is done in {(time() - start_time):.3f} seconds!")
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = "Sketch Video Create Arguments")
    
    # Add arguments to the parser
    parser.add_argument("-r", "--root", type = str, default = "potato.mp4", help = "Path to the video")
    parser.add_argument("-mn", "--model_name", type = str, default = "u2", help = "Model name to convert images to sketches")
    parser.add_argument("-f", "--frames", type = str, default = "frames", help = "Path to save the frames")
    parser.add_argument("-s", "--sketches", type = str, default = "sketches", help = "Path to save the sketches")
    parser.add_argument("-sp", "--save_path", type = str, default = "videos", help = "Path to save the video")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
    
