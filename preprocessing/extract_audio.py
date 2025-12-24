import os
import csv
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip
from pathlib import Path

video_pth = Path('/home/prj/data/AVE_Dataset/AVE/')
mode ='val'
save_pth = '/home/prj/data/AVE_Dataset/preprocess/audio'
train_sep_file = f'/home/prj/DAVIS/data/AVE/{mode}.csv'

os.makedirs(save_pth, exist_ok=True)

with open(train_sep_file) as f:
    csv_reader = csv.reader(f)
    for item in csv_reader:
        mp4 = os.path.join(video_pth, item[1])
        audio_pth = os.path.join(save_pth, item[0])
        print(audio_pth)
        if not os.path.exists(os.path.join(save_pth, item[1])):
            os.makedirs(os.path.join(save_pth, item[1]))
        video = VideoFileClip(mp4)
        audio = video.audio
        if os.path.exists(audio_pth):
            print("already exist!")
            continue
        audio.write_audiofile(audio_pth, fps=11025)
        print("finish video id: " + item[1])
    