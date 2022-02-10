from scipy.io import wavfile
import sounddevice as sd

print("hello!")
samplerate, data = wavfile.read('./voice.wav')
print("Read wav file with sample rate", samplerate, "and length", len(data))



# for samp in data:
# 	print(samp)


