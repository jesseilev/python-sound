from scipy.io import wavfile
from scipy.io.wavfile import write
import numpy as np
from scipy.integrate import quad
import math

#samplerate, data = wavfile.read('./drum_roll_heavy.wav')
samplerate, data = wavfile.read('./wind-quintet.wav')  #int16 format

print("Read wav file with sample rate", samplerate, "and length", len(data))
# The input should be a uint8 wav file with two uint8 channels.  
# Wav is not standardized! Won't work with 16bit, 24bit, or mono

length = min(samplerate*60,len(data))  # truncate to 60 seconds of audio
data = np.resize(data,length,2)

signal = data.astype(np.int32)  #convert to signed 32-bit int

print(type(data[1000][0]))
print(type(signal[1000][0]))


#trans = 32768  # use for uint8 wav files
#prec = 65536

trans = 0  # set to 127 for int16 wav files
prec = 16 #set to 65536 for uint8 files


signal -= trans
# signal is now an array of pairs of 32-bit ints between -127 and 128
# we need the extra bits to do math without wrapping around

#signal *= 1024    # signal is now a signed array of pairs of ints between -127*1024 and 128*1024

# left = [0]*length
# right = [0]*length
# for n in range(0,length):
# 	left = signal[n][0]
# 	right = signal[n][1]



left = []   
right = []
for s in signal:   #extract left and right channels
	left.append(s[0])
	right.append(s[1])

for n in range(100000,100100):
	print(data[n], signal[n], left[n], right[n])	


#def gaussian(x,sigma):
#	return np.exp(-x*x/(2*sigma*sigma))/(np.sqrt(2*np.pi)*sigma)

def gaussian(x):  #gaussian density evaluated at real number x
	return np.exp(-x*x/2)/np.sqrt(2*np.pi)

def gauss_int(x,y):   #integral of the gaussian density from x to y
	return quad(gaussian, x, y)[0]


def gauss_weight(sigma):  # weights for gaussian smoothing at scale sigma
	cutoff = math.ceil(5*sigma)
	factor = math.sqrt(sigma) #yes that's square root of the std dev!
	weight = [0]* (cutoff*2 + 1)
	for k in range(-cutoff, +cutoff):
		weight[k+cutoff] = prec * factor* gauss_int((k-0.5)/sigma,(k+0.5)/sigma)
	return np.trim_zeros((np.array(weight)).astype(np.int32))
# returning an array of positive ints, symmetric, of odd length, scaled to sum to about prec*sqrt(sigma)
# the factor of sqrt(sigma) should restore the original volume since smoothing will damp it by sqrt(sigma)

#print(gauss_weight(10))
#print(gauss_weight(100))


def subsample(sample, sigma):  
# returns Gaussian smoothing of the array vals at scale sigma, subsampled at rate sigma, rounded to nearest int

	weight = gauss_weight(sigma)  # load the array of gaussian weights, for smoothing
	window = len(weight)  # size of the smoothing window

#	length = len(sample) #length of the original sample
	sublength = math.ceil(length/sigma)  # an upper bound for the length of the subsample
	subsample =[0]*sublength

	n=0
	t=0
	while t<length-window: # smooth the sample and write it to subsample
		total=0
		for k in range(0,window):  # convolve with gaussian weights
			total += sample[t+k] * weight[k]    # note these are all ints!
		subsample[n] = total
		n+=1
		t=math.floor(n*sigma)

	return np.trim_zeros(np.array(subsample))


def scaledown(sample,factor):

	offset = math.floor(factor/2)
	return np.floor_divide(sample+offset, factor)  #divides by factor and rounds to the nearest int



sigma = 1

subsample_left = subsample(left,sigma)
subsample_left = (scaledown(subsample_left,256)+127).astype(np.uint8)

subsample_right = subsample(right,sigma)
subsample_right = (scaledown(subsample_right,256)+127).astype(np.uint8)

for n in range(3000,3100):
	print(subsample_left[n])


#subsample_left = (scaledown(subsample(left,sigma),2*prec) -127 ).astype(np.uint8)
#subsample_right = (scaledown(subsample(right,sigma),2*prec) -127).astype(np.uint8)

subsample=[]  #combine the left and right channels into an array of pairs
L = len(subsample_left)
for n in range(0,L):
	subsample.append([subsample_left[n],subsample_right[n]])
# subsample = np.array(subsample)

# l = math.floor((len(gauss_weight(10))-1)/2)
#for n in range(3000,3100):
# 	print(left[10*n+l],subsample_left[n])

#smoothed_left = np.array(smooth(left, sig)).astype(np.uint8)
#smoothed_right = np.array(smooth(right, sig)).astype(np.uint8)

# smoothed_length = round(len(data) / sig)
# smoothed = [[0,0]]*smoothed_length
# for n in range(0,smoothed_length):
# 	orig_index = n * sig
# 	smoothed[n] = [smoothed_left[orig_index],smoothed_right[orig_index]]

# for n in range(1000,1100):
# 	print("Left channel",left[n],"Smoothed pair",smoothed[n])

# print(type(data))
# print(type(np.array(smoothed_right)))
# print(type(np.array(smoothed_right).astype(np.uint8)))


# for n in range(1000, 1100):
# 	print("data = ", data[n])
# 	print("new_data = ", new_data[n])

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
write("output.wav", samplerate, np.array(subsample))
print("Wrote ouptut.wav with sample rate ",samplerate," of length",len(subsample))


