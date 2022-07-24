import torch
import torchaudio
import matplotlib.pyplot as plt

filepath = "./sounds/a_higher.wav" 
waveform, sample_rate = torchaudio.load(filepath)
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

SAMPLING_RATE = 44800
filename = filepath.split("/")[-1]

sk = "waveform"
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(1.6180 * 4, 4*2))
lns1=ax1.plot(waveform.t().numpy(),"red",label = "waveform[0]")
lns2=ax2.plot(waveform.t().numpy(),"red",label = "waveform[0]")
lns3=ax3.plot(waveform.t().numpy(),"blue",label = "waveform[0]")
ax1.legend(loc=0)
ax2.legend(loc=0)
ax3.legend(loc=0)
ax1.set_title(sk)
ax2.set_xlim(50000,50000+SAMPLING_RATE*0.0625) #0,44100*0.25
ax3.set_xlim(3*SAMPLING_RATE,SAMPLING_RATE*3.0625)
plt.pause(1)
#plt.savefig('./saved_image/{}_{}_double_.png'.format(filename.split(".")[0], sk)) 
plt.close()

specgram = torchaudio.transforms.Spectrogram(n_fft=1024)(waveform)
print("Shape of spectrogram: {}".format(specgram.size()))

sk = "specgram"
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(1.6180 * 4, 4*2))
lns1=ax1.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray') 
lns2=ax2.imshow(specgram.log2()[0,:,:].numpy(), cmap='hsv') 
lns3=ax3.imshow(specgram.log2()[0,:,:].numpy(), cmap='hsv') 

ax2.set_ylim(250,0)
ax3.set_ylim(125,0)
ax1.set_title(sk)

plt.pause(1)
plt.savefig('./saved_images/{}_{}_double.png'.format(filename.split(".")[0], sk)) 
