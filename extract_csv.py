import os
from pathlib import Path
import librosa
import numpy as np
import pandas as pd

def analyze_audio_amplitudes(root_path, amplitude_ratio_threshold=1000):
    """
    Analyze and filter the maximum amplitude of speech.wav compared to the original audio in the Muddy_Mix dataset.

    Args:
        root_path: Root path of the Muddy_Mix dataset
        amplitude_ratio_threshold: Amplitude ratio threshold compared to original (default: 1000x)

    Returns:
        dict: Analysis results
    """
    root = Path(root_path)

    if not root.exists():
        print(f"Path does not exist: {root_path}")
        return

    valid_audios = []
    invalid_audios = []
    error_audios = []

    total_subvideos = 0
    processed_subvideos = 0

    # Check all folders in the root directory
    for video_folder in root.iterdir():
        if video_folder.is_dir():
            video_name = video_folder.name
            print(f"\nVideo: {video_name}")

            # Find sub_video folders in each video folder
            sub_video_path = video_folder / "sub_video"
            if sub_video_path.exists() and sub_video_path.is_dir():
                subvideos = [d for d in sub_video_path.iterdir() if d.is_dir()]

                for subvideo in subvideos:
                    total_subvideos += 1
                    subvideo_name = subvideo.name

                    # speech.wav file path
                    speech_path = subvideo / "separated" / "speech.wav"

                    # Find original audio in audio_raw folder
                    audio_raw_path = subvideo / "audio_raw"
                    raw_audio_file = None

                    if audio_raw_path.exists() and audio_raw_path.is_dir():
                        # Find the first audio file in audio_raw folder
                        audio_files = [f for f in audio_raw_path.iterdir() if f.is_file() and f.suffix.lower() in ['.wav', '.mp3', '.mp4', '.m4a']]
                        if audio_files:
                            raw_audio_file = audio_files[0]

                    if not speech_path.exists():
                        error_audios.append({
                            'video': video_name,
                            'subvideo': subvideo_name,
                            'error': 'speech.wav not found'
                        })
                        continue

                    if not raw_audio_file:
                        error_audios.append({
                            'video': video_name,
                            'subvideo': subvideo_name,
                            'error': 'raw audio not found'
                        })
                        continue

                    try:
                        # Load audio files and measure maximum amplitude
                        speech_audio, speech_sr = librosa.load(str(speech_path), sr=None)
                        speech_duration = len(speech_audio) / speech_sr
                        speech_max_amplitude = np.max(np.abs(speech_audio))

                        raw_audio, raw_sr = librosa.load(str(raw_audio_file), sr=None)
                        raw_duration = len(raw_audio) / raw_sr
                        raw_max_amplitude = np.max(np.abs(raw_audio))

                        # Calculate amplitude ratio (original vs speech volume)
                        if speech_max_amplitude > 0:
                            amplitude_ratio = raw_max_amplitude / speech_max_amplitude
                        else:
                            amplitude_ratio = float('inf')

                        # Also measure file sizes for reference
                        speech_size = speech_path.stat().st_size
                        raw_size = raw_audio_file.stat().st_size

                        audio_info = {
                            'video': video_name,
                            'subvideo': subvideo_name,
                            'speech_path': str(speech_path),
                            'raw_path': str(raw_audio_file),
                            'speech_size_mb': speech_size / (1024 * 1024),
                            'raw_size_mb': raw_size / (1024 * 1024),
                            'amplitude_ratio': amplitude_ratio,
                            'speech_duration': speech_duration,
                            'raw_duration': raw_duration,
                            'speech_max_amp': speech_max_amplitude,
                            'raw_max_amp': raw_max_amplitude
                        }

                        # If ratio > threshold, invalid (speech volume too small), else valid
                        if amplitude_ratio > amplitude_ratio_threshold:
                            invalid_audios.append(audio_info)
                        else:
                            valid_audios.append(audio_info)

                        processed_subvideos += 1

                    except Exception as e:
                        error_audios.append({
                            'video': video_name,
                            'subvideo': subvideo_name,
                            'error': str(e)
                        })
                        print(f"  ⚠️ {subvideo_name}: Error occurred - {e}")

    # Results summary
    print("\n" + "="*60)
    print("Analysis Results Summary (Amplitude Criteria):")
    print(f"Total subvideos: {total_subvideos}")
    print(f"Processed subvideos: {processed_subvideos}")
    print(f"Valid audios: {len(valid_audios)} (usable)")
    print(f"Invalid audios: {len(invalid_audios)} (speech volume {amplitude_ratio_threshold}x smaller than original)")
    print(f"Errors: {len(error_audios)}")

    if valid_audios:
        print(f"\nValid audio statistics:")
        amplitude_ratios = [audio['amplitude_ratio'] for audio in valid_audios]
        speech_amps = [audio['speech_max_amp'] for audio in valid_audios]
        raw_amps = [audio['raw_max_amp'] for audio in valid_audios]

        print(f"  Average amplitude ratio: {np.mean(amplitude_ratios):.2f}x")
        print(f"  Median amplitude ratio: {np.median(amplitude_ratios):.2f}x")
        print(f"  Max amplitude ratio: {np.max(amplitude_ratios):.2f}x")
        print(f"  Min amplitude ratio: {np.min(amplitude_ratios):.2f}x")
        print(f"  Average speech max amplitude: {np.mean(speech_amps):.4f}")
        print(f"  Average raw max amplitude: {np.mean(raw_amps):.4f}")

    if invalid_audios:
        print(f"\nInvalid audio examples (top 5 by amplitude ratio):")
        invalid_sorted = sorted(invalid_audios, key=lambda x: x['amplitude_ratio'], reverse=True)
        for i, audio in enumerate(invalid_sorted[:5]):
            print(f"  {i+1}. {audio['video']}/{audio['subvideo']}: {audio['amplitude_ratio']:.1f}x")
            print(f"      Raw amplitude: {audio['raw_max_amp']:.4f}, Speech amplitude: {audio['speech_max_amp']:.4f}")

    return {
        'total_subvideos': total_subvideos,
        'processed_subvideos': processed_subvideos,
        'valid_audios': valid_audios,
        'invalid_audios': invalid_audios,
        'error_audios': error_audios,
        'valid_count': len(valid_audios),
        'invalid_count': len(invalid_audios),
        'error_count': len(error_audios)
    }

def create_valid_audio_csv(audio_analysis_result, output_path="/home/prj/data/valid_muddy_mix_audios.csv"):
    """
    Extract valid audios from audio_analysis results and save to CSV file.

    Args:
        audio_analysis_result: Result from analyze_audio_amplitudes function
        output_path: Path to save the CSV file

    Returns:
        pandas.DataFrame: DataFrame of valid audios
    """

    if not audio_analysis_result or 'valid_audios' not in audio_analysis_result:
        print("No valid audio_analysis result found.")
        return None

    valid_audios = audio_analysis_result['valid_audios']

    if not valid_audios:
        print("No valid audios found.")
        return None

    # Create DataFrame
    df = pd.DataFrame(valid_audios)

    # Reorder columns and rename
    df_clean = df[[
        'video', 'subvideo', 'speech_path', 'raw_path',
        'speech_duration', 'raw_duration',
        'speech_max_amp', 'raw_max_amp', 'amplitude_ratio',
        'speech_size_mb', 'raw_size_mb'
    ]].copy()

    # Rename columns
    df_clean.columns = [
        'Video_Name', 'SubVideo_Name', 'Speech_Path', 'Raw_Audio_Path',
        'Speech_Duration_Sec', 'Raw_Duration_Sec',
        'Speech_Max_Amplitude', 'Raw_Max_Amplitude', 'Amplitude_Ratio',
        'Speech_Size_MB', 'Raw_Size_MB'
    ]

    # Round numeric columns
    df_clean['Speech_Duration_Sec'] = df_clean['Speech_Duration_Sec'].round(2)
    df_clean['Raw_Duration_Sec'] = df_clean['Raw_Duration_Sec'].round(2)
    df_clean['Speech_Max_Amplitude'] = df_clean['Speech_Max_Amplitude'].round(4)
    df_clean['Raw_Max_Amplitude'] = df_clean['Raw_Max_Amplitude'].round(4)
    df_clean['Amplitude_Ratio'] = df_clean['Amplitude_Ratio'].round(2)
    df_clean['Speech_Size_MB'] = df_clean['Speech_Size_MB'].round(2)
    df_clean['Raw_Size_MB'] = df_clean['Raw_Size_MB'].round(2)

    # Sort by amplitude ratio (lower ratio = better quality)
    df_clean = df_clean.sort_values('Amplitude_Ratio').reset_index(drop=True)

    # Save to CSV
    df_clean.to_csv(output_path, index=False)

    print(f"✅ Valid audio CSV file created: {output_path}")
    print(f"📊 Total {len(df_clean)} valid subvideos saved")

    # Print basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"  - Average Speech duration: {df_clean['Speech_Duration_Sec'].mean():.2f} sec")
    print(f"  - Average amplitude ratio: {df_clean['Amplitude_Ratio'].mean():.2f}x")
    print(f"  - Amplitude ratio range: {df_clean['Amplitude_Ratio'].min():.2f} ~ {df_clean['Amplitude_Ratio'].max():.2f}x")

    # Statistics by video
    video_counts = df_clean['Video_Name'].value_counts()
    print(f"\n📹 Valid subvideos per video (top 10):")
    for i, (video, count) in enumerate(video_counts.head(10).items()):
        print(f"  {i+1:2d}. {video[:20]:20s}: {count:3d}")

    return df_clean

if __name__ == "__main__":
    # Execute
    muddy_mix_path = "/home/prj/data/Muddy_Mix"
    audio_analysis = analyze_audio_amplitudes(muddy_mix_path, amplitude_ratio_threshold=1000)

    if audio_analysis:
        valid_df = create_valid_audio_csv(audio_analysis)

        # Also create a detailed CSV
        if valid_df is not None:
            # Simple version (without paths)
            simple_df = valid_df[['Video_Name', 'SubVideo_Name', 'Speech_Duration_Sec',
                                 'Speech_Max_Amplitude', 'Amplitude_Ratio']].copy()
            simple_df.to_csv("/home/prj/data/valid_muddy_mix_simple.csv", index=False)

            print(f"\n📋 Simple version CSV also created: /home/prj/data/valid_muddy_mix_simple.csv")

            # Print first 5 samples
            print(f"\n🔍 CSV preview (first 5):")
            print(simple_df.head())