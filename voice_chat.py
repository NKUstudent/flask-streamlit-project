# 导包
import wave

import pyttsx3
import speech_recognition as sr
from webrtcvad import Vad
import pyaudio
import numpy as np
import audioop

def text_to_voice(text, language, rate, volume, sayit):
    # 初始化语音引擎
    engine = pyttsx3.init()
    # 设置语速、音量
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    voices = engine.getProperty('voices')
    for voice in voices:
        print('id = {} \nname = {} \n'.format(voice.id, voice.name))
    if language == 0:
        # 中文
        engine.setProperty('voice', voices[0].id)
    elif  language == 1:
        # 英文
        engine.setProperty('voice', voices[1].id)
    if sayit == 1:
        engine.say(text)
    elif sayit == 0:
        print('我不念咯')
    # 结束语句
    engine.runAndWait()
    engine.stop()

# 将WAV文件中的音频转为文字
def voice_to_text():
    # 初始化识别器
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('说些什么吧...')
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language='zh-CN')
            print('识别结果：' + text)
        except sr.UnknownValueError:
            print('Bing Speech Recognition无法理解你的语音')
        except sr.RequestError as e:
            print('无法从Bing Speech Recognition服务中获取数据; {0}'.format(e))

class AudioFrame(object):
    # 简单的音频帧容器，用于存储从麦克风读取的原始音频数据。
    def __init__(self, raw_data, timestamp, sample_rate, channels):
        self.raw_data = raw_data
        self.timestamp = timestamp
        self.sample_rate = sample_rate
        self.channels = channels

    # 使用webrtcvad检测静音。
    def is_silence(self, vad, aggressiveness=3):
        # 确保音频是单声道的16位PCM
        if self.channels != 1:
            raise ValueError("Audio must be mono.")
        if self.raw_data.dtype != np.int16:
            raise ValueError("Audio must be 16-bit PCM.")

        frame_length, frame_step = 30 * self.sample_rate // 1000, 10 * self.sample_rate // 1000  # 10ms和30ms
        num_padding = (frame_length - len(self.raw_data) % frame_step) if len(self.raw_data) % frame_step != 0 else 0
        padding = np.zeros((num_padding,), dtype=np.int16)
        frame_data = np.pad(self.raw_data, (0, num_padding), mode='constant', constant_values=(0, 0))

        # 将音频数据分割成帧
        num_frames = (len(frame_data) - frame_length) // frame_step + 1
        frames = [frame_data[i * frame_step:(i + 1) * frame_step] for i in range(num_frames)]

        # 使用VAD检测每帧
        for frame in frames:
            if frame.ndim != 1:
                raise ValueError("音频帧必须是单通道的")

                # 确保是16位整数
            if frame.dtype != np.int16:
                raise ValueError("音频帧必须是int16类型")

                # 检查帧长度是否在合理范围内（这里以16 kHz采样率为例）
            if not (100 <= len(frame) <= 1000):  # 假设合理的帧长度在10ms到100ms之间
                raise ValueError("音频帧长度不合理")

            if not vad.is_speech(frame.tobytes(), sample_rate = self.sample_rate):
                return True
        return False

# 将numpy帧保存为WaV文件。
def save_frames_to_wav(frames, file_path, sample_rate, num_channels):
    with wave.open(file_path, 'wb') as wav_file:
        n_bytes = 2 # 16-bit PCM
        sampwidth = n_bytes * num_channels
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b''.join(frame.tobytes() for frame in frames))

def main():
    # 初始化Vad
    vad = Vad(mode=3)

    # 初始化语音识别器
    r = sr.Recognizer()

    # 设置麦克风参数
    CHUNK = 1024  # 每次读取的字节数
    FORMAT = pyaudio.paInt16  # 音频格式
    CHANNELS = 1  # 声道数
    RATE = 44100  # 采样率

    # 使用pyaudio打开麦克风
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("开始录音...")

    # 缓存非静音音频数据
    non_silent_frames = []

    try:
        while True:
            # 读取音频数据
            data = stream.read(CHUNK)
            if not data:
                break

            # 将字节数据转换为numpy数组
            frame = np.frombuffer(data, dtype=np.int16)

            # 检测静音
            if not AudioFrame(frame, None, RATE, CHANNELS).is_silence(vad):
                non_silent_frames.append(frame)

                # 这里可以添加逻辑来定期处理非静音帧（例如，当缓存达到一定大小时）
            # 或者当检测到长时间的静音时停止录音

            # 注意：为了简化示例，我们没有在这里实现完整的语音到文本的转换
            # 你需要将非静音帧合并成一个完整的音频块，并使用speech_recognition进行识别
        if non_silent_frames:
            temp_wav_file = 'temp_audio.wav'
            save_frames_to_wav(non_silent_frames, temp_wav_file, RATE, CHANNELS)
            voice_to_text(temp_wav_file)

    except KeyboardInterrupt:
        print("录音已停止")

    finally:
        # 停止录音并关闭流
        stream.stop_stream()
        stream.close()
        p.terminate()


#text_to_voice(text = '你好，我是一个基于langchain的一个语音助手，有什么可以帮你的吗？', language = 0, rate = 200, volume = 100, sayit = 1)
voice_to_text()
#if __name__ == '__main__':
#    main()