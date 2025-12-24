import os
import random
import numpy as np
import csv
from .base import BaseDataset
import clip
import torch


class MuddyMixDataset(BaseDataset):
    def __init__(self, root_dir, list_sample, opt, **kwargs): # opt : args
        super(MuddyMixDataset, self).__init__(
            list_sample, opt, **kwargs) # list sample에 csv 파일넣으면 됨
        self.fps = opt.frameRate
        #self.num_mix = opt.num_mix NOTE 의미 없음
        self.audLen = opt.audLen
        self.num_frames = opt.num_frames
        self.model_type = opt.arch_frame
        self.root_dir = root_dir


    def make_stft(self, audio_raw, audio_sep):
        amp_mix, phase_mix = self._stft(audio_raw)
        ampN, _ = self._stft(audio_sep)
        mag = ampN.unsqueeze(0)
        audio_sep = torch.from_numpy(audio_sep)

        return amp_mix.unsqueeze(0), mag, phase_mix.unsqueeze(0), torch.from_numpy(audio_raw)

    def __getitem__(self, index):
        info = self.list_sample[index]
        video_name = info['Video_Name']
        sub_video_name = info['SubVideo_Name']
        video_path = os.path.join(self.root_dir, video_name, 'sub_video', sub_video_name)
        raw_audio_path = os.path.join(video_path, 'audio_raw', f'{sub_video_name}.wav')
        audio_sep_path = os.path.join(video_path, 'separated', 'speech.wav') # NOTE 여기 바꿔야 함 나중에
        text = clip.tokenize('an image of human speaking')
        frame_files = sorted([f for f in os.listdir(os.path.join(video_path, 'frames'))])
        path_frames = [os.path.join(video_path,'frames', f) for f in frame_files]
        num_frames = len(path_frames)
        assert num_frames !=0, f"이상해 뭔가 : {path_frames}"
        center = num_frames // 2
        half = self.num_frames // 2
        start = max(0, center - half)
        end = min(num_frames, center + half + self.num_frames % 2)
        selected = path_frames[start:end]

        # 길이가 부족하면 앞/뒤 프레임을 반복해서 채움
        while len(selected) < self.num_frames:
            if start > 0:
                start -= 1
                selected.insert(0, path_frames[start])
            elif end < num_frames:
                selected.append(path_frames[end])
                end += 1
            else:  # 정말 짧은 영상이면 마지막 프레임 반복
                selected.append(selected[-1])

        # 더 길면 여기서 잘라줌
        selected = selected[:self.num_frames]
        path_frames = selected
        # load frames and audios, STFT
        try:
            if self.model_type != 'clip':
                frames = self._load_frames(path_frames)
            else:
                frames = self._load_frames_clip(path_frames)
                
            # jitter audio
            center_timeN = num_frames // 2
            audio_raw = self._load_audio(raw_audio_path, center_timeN)
            audio_sep = self._load_audio(audio_sep_path, center_timeN)
            mag_mix, mag, phase_mix, audio_mix = self.make_stft(audio_raw, audio_sep)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            raise e
            #return None
            # mag_mix, mags, frames, audios, phase_mix = \
            #     self.dummy_mix_data(N)
            # audio_mix = audios[0]

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mag, 
        'audio_mix': audio_mix}
        ret_dict['audios'] = audio_sep
        # ret_dict['class'] = classes
        # ret_dict['text'] = texts

        if self.split != 'train':
            # ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = info

        # return ret_dict # mix 한거 : 원본, mags: sep한 speech (아니면 foley나 뭐 등등 ?)
        return ret_dict