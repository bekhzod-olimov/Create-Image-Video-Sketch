import os, argparse
from utils import get_video_frames, get_sketches, create_video
from time import time

def run(args):    

    start_time = time()
    folder_name = f"{os.path.basename(args.root).split('.')[0]}_u2"
    get_video_frames(video_path = args.root, save_path = f"{args.frames}/{folder_name}")
    get_sketches(path = f"{args.frames}/{folder_name}", save_path = f"{args.sketches}/{folder_name}", device = "cuda")
    create_video(path = f"sketches/{folder_name}", save_path = "videos", save_name = f"{folder_name}.mp4")
    print(f"The process is done in {(time() - start_time):.3f} seconds!")
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = "Sketch Video Create Arguments")
    
    # Add arguments to the parser
    parser.add_argument("-r", "--root", type = str, default = "../pencil_sketches/potato.mp4", help = "Path to the video")
    parser.add_argument("-f", "--frames", type = str, default = "frames", help = "Path to save the frames")
    parser.add_argument("-s", "--sketches", type = str, default = "sketches", help = "Path to save the sketches")
    parser.add_argument("-sp", "--save_path", type = str, default = "videos", help = "Path to save the video")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
    
