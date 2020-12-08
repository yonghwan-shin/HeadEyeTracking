#%%

from plotly.subplots import make_subplots
from analysing_functions import *
from IIRfilter import *
# IF you are using Pycharm
import plotly.io as pio
from scipy import fftpack, signal, stats



target = 3
env = "W"
block = 3
subject = 301
## Bring the result into pandas dataframe
holo, imu, eye = bring_data(target, env, block, subject)
## Get the delayed time between hololens - laptop
shift, corr, shift_time = synchronise_timestamp(imu, holo, show_plot=False)

## filter out the low-confidene eye data
# eye = eye[eye.confidence > 0.8]

## match the delayed timestamp into hololens' timestamp
eye.timestamp = eye.timestamp - shift_time

def butter_lowpass_filter(data,cutoff,fs,order=2):
    nyq = fs/2
    normal_cutoff = cutoff/nyq
    b,a = signal.butter(order,normal_cutoff,bytpe='low',analog=False)
    y=signal.filtfilt(b,a,data)
    return y

#%%
from scipy import fftpack,fft
from scipy.fft import rfft,rfftfreq,irfft

f_s = 60
x = np.array(holo.head_rotation_y)
X = fftpack.fft(x)
freqs = fftpack.fftfreq(len(x),1/f_s)

fig,ax = plt.subplots()
ax.stem(freqs,np.abs(X))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel("Frequency Domain (Spectrum) Magnitude")
ax.set_xlim(0,f_s/15)
plt.show()

yf = rfft(x)
xf = rfftfreq(len(x),1/f_s)
# plt.plot(xf,np.abs(yf))
# plt.show()


plt.plot(holo.timestamp,x)
plt.plot(holo.timestamp,holo.Phi)
filtered_x = butter_lowpass_filter(x,0.8,60)
plt.plot(holo.timestamp,filtered_x)
plt.show()
y= np.array(holo.head_rotation_x)
plt.plot(holo.timestamp,y)
plt.plot(holo.timestamp,-holo.Theta)
filtered_y = butter_lowpass_filter(y,1.8,60)
plt.plot(holo.timestamp,filtered_y)
plt.show()
# cut=20
# yf[cut:]=0
# dominant_frequency_indices = np.argsort(-yf)
# ncnt = 15
# plti=0
# for i in range(ncnt+1):
#     plti+=1
#     plt.subplot(4,4, plti)
#     plt.title("N={}".format(i+1))
#     yf[dominant_frequency_indices[i]]=0
#     plt.plot(irfft(yf))
#     plt.plot(x)
# plt.show()


#%%

# x = holo.head_rotation_x
holo.Theta = holo.Theta + holo.head_rotation_x.mean()
holo.head_rotation_x=holo.head_rotation_x-holo.head_rotation_x.mean()

from scipy.interpolate import splrep,splev
spl = splrep(holo.timestamp,holo.head_rotation_x)
fs = 60*2
dt=1/fs
spl=splrep(holo.timestamp,holo.head_rotation_x)
newt=np.arange(0,holo.timestamp.values[-1],dt)
newx=splev(newt,spl)

plt.plot(newt,newx)
plt.plot(holo.timestamp,holo.head_rotation_x)
plt.plot(holo.timestamp,-holo.Theta)
plt.show()

nfft = len(newt)
df = fs/nfft
k=np.arange(nfft)
f=k*df

nfft_half=math.trunc(nfft/2)
f0=f[range(nfft_half)]
y=np.fft.fft(newx)/nfft *2
y0=y[range(nfft_half)]
amp=abs(y0)

plt.plot(f0,amp)
plt.xlim(0,30)
plt.show()

ampsort = np.sort(amp)
q1,q3 = np.percentile(ampsort,[25,75])
iqr=q3-q1
upper_bound = q3+1.5*iqr
outer=ampsort[ampsort>upper_bound]
topn=len(outer)
print('outer count:' ,len(outer))

idxy=np.argsort(-amp)
# for i in range(topn):
#     print()
newy=np.zeros((nfft,))
arfreq=[]
arcoec=[]
arcoes=[]
for i in range(topn):
    freq=f0[idxy[i]]
    yx=y[idxy[i]]
    coec=yx.real
    coes=yx.imag*-1
    newy += coec * np.cos(2 * np.pi * freq * newt) + coes * np.sin(2 * np.pi * freq * newt)
    arfreq.append(freq)
    arcoec.append(coec)
    arcoes.append(coes)
plt.plot(holo.timestamp,holo.head_rotation_x,c='r',label='original')
plt.plot(newt,newy,c='b',label='fft')
plt.legend()
plt.show()

plti=0
ncnt=15

newy = np.zeros((nfft,))
for i in range(ncnt+1):
    freq = f0[idxy[i]]
    yx = y[idxy[i]]
    coec = yx.real
    coes = yx.imag * -1
    print('freq=', freq, 'coec=', coec, ' coes', coes)
    newy += coec * np.cos(2 * np.pi * freq * newt) + coes * np.sin(2 * np.pi * freq * newt)
    plti+=1
    plt.subplot(4,4, plti)
    plt.title("N={}".format(i+1))
    plt.plot(newt, newy)
    plt.plot(holo.timestamp,-holo.Theta)
    # plt.savefig('fft02_5.jpg')
plt.show()