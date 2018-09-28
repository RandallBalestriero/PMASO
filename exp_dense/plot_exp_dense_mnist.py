import cPickle
from pylab import *
import glob
import matplotlib as mpl
label_size = 13
mpl.rcParams['xtick.labelsize'] = label_size+10
mpl.rcParams['ytick.labelsize'] = label_size

fs=15



def plotclasses(classes,samplesclass1):
    for i,k in zip(range(len(classes)),classes):
	print shape(samplesclass1[k])
        for j in xrange(10):
            subplot(len(classes),10,1+i*10+j)
            imshow(samplesclass1[k][j,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])





def doit(sig,l,doplot=1):
    f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_neurons_'+sig+'_'+str(l)+'.pkl','rb')
    LOSSES1,reconstruction1,x1,samplesclass01,samplesclass11,W1=cPickle.load(f)
    f.close()
    if(doplot==0): return concatenate(LOSSES1)
    for i in xrange(150):
        figure(figsize=(2,2))
        imshow(x1[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/neurons/x_'+sig+'_'+str(l)+'_'+str(i)+'.png')
        figure(figsize=(2,2))
        imshow(reconstruction1[-1][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
        tight_layout()
        savefig('BASE_EXP/neurons/reconstruction_x_'+sig+'_'+str(l)+'_'+str(i)+'.png')
        close()
    #
    for ii in xrange(4):
        figure(figsize=(20,4))
        for i in xrange(10):
            subplot(2,10,1+i)
            imshow(x1[i+10*ii,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            subplot(2,10,1+i+10)
            imshow(reconstruction1[-1][i+10*ii,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
        tight_layout()
        savefig('BASE_EXP/neurons/reconstruction_all_x_'+sig+'_'+str(l)+'_'+str(ii)+'.png')
        close()
    #
    figure(figsize=(20,20))
    for c in xrange(10):
	for i in xrange(10):
            imshow(samplesclass01[-1][c][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
    tight_layout()
    savefig('BASE_EXP/neurons/samplesall_0_'+sig+'_'+str(l)+'.png')
    close()
    #
    figure(figsize=(20,20))
    for c in xrange(10):
	for i in xrange(10):
            imshow(samplesclass11[-1][c][i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
    tight_layout()
    savefig('BASE_EXP/neurons/samplesall_1_'+sig+'_'+str(l)+'.png')
    close()
    #
    for ii in xrange(40):
        for i in xrange(10):
            figure(figsize=(2,2))
            imshow(samplesclass01[-1][i][ii,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('BASE_EXP/neurons/sample_0_'+sig+'_'+str(l)+'_'+str(ii)+'_'+str(i)+'.png')
            close()
            figure(figsize=(2,2))
            imshow(samplesclass11[-1][i][ii,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
            tight_layout()
            savefig('BASE_EXP/neurons/sample_1_'+sig+'_'+str(l)+'_'+str(ii)+'_'+str(i)+'.png')
            close()
    return concatenate(LOSSES1)



#loss1 = unique(doit('local',1,0))
#loss2 = unique(doit('local',2,0))
#loss3 = unique(doit('local',3,0))
#loss4 = unique(doit('global',1,0))
#loss5 = unique(doit('global',2,0))
loss6 = unique(doit('global',3,1))


figure(figsize=(4,4))
plot(loss6,c='k',ls='-',linewidth=4)
plot(range(len(loss6))[3::20],loss6[3::30],c='k',ls='None',marker='x',markersize=8)
tight_layout()
savefig('BASE_EXP/neurons/losses.png')
close()





