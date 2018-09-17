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





def doit(sig,l):
    f=open('/mnt/project2/rb42Data/PMASO/BASE_EXP/exp_neurons_'+sig+'_'+str(l)+'.pkl','rb')
    LOSSES1,reconstruction1,x1,samplesclass01,samplesclass11,W1=cPickle.load(f)
    f.close()
    figure(figsize=(14,1.5))
    for i in xrange(10):
        subplot(1,10,1+i)
        imshow(x1[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
        xticks([])
        yticks([])
    tight_layout()
    savefig('BASE_EXP/neurons/reconstruction_x'+sig+'_'+str(l)+'.png')
    close()
    for reconstruction,j in zip(reconstruction1,range(len(reconstruction1))):
        figure(figsize=(14,1.5))
        for i in xrange(10):
            subplot(1,10,1+i)
            imshow(reconstruction[i,:,:,0],aspect='auto',cmap='Greys',interpolation='nearest')
            xticks([])
            yticks([])
        tight_layout()
        savefig('BASE_EXP/neurons/reconstruction_x'+sig+'_'+str(l)+'_'+str(j)+'.png')
        close()
    for samplesclass,j in zip(samplesclass11,range(len(samplesclass11))):
        figure(figsize=(14,15))
        classes=range(10)
        plotclasses(classes,samplesclass)
        tight_layout()
        savefig('BASE_EXP/neurons/tenclass1_'+sig+'_'+str(l)+'_'+str(j)+'.png')
        close()
        for c in xrange(10):
            figure(figsize=(14,1.5))
            classes=[c]
            plotclasses(classes,samplesclass)
            tight_layout()
            savefig('BASE_EXP/neurons/twoclass1_'+sig+'_'+str(l)+'_'+str(c)+'_'+str(j)+'.png')
            close()
    for samplesclass,j in zip(samplesclass01,range(len(samplesclass01))):
        figure(figsize=(14,15))
        classes=range(10)
        plotclasses(classes,samplesclass)
        tight_layout()
        savefig('BASE_EXP/neurons/tenclass0_'+sig+'_'+str(l)+'_'+str(j)+'.png')
        close()
	for c in xrange(10):
            figure(figsize=(14,1.5))
            classes=[c]
            plotclasses(classes,samplesclass)
            tight_layout()
            savefig('BASE_EXP/neurons/twoclass0_'+sig+'_'+str(l)+'_'+str(c)+'_'+str(j)+'.png')
            close()
    return concatenate(LOSSES1)



loss1 = unique(doit('local',1))
loss2 = unique(doit('local',2))
loss3 = unique(doit('local',3))
loss4 = unique(doit('global',1))
loss5 = unique(doit('global',2))
loss6 = unique(doit('global',3))

figure(figsize=(15,15))
plot(loss1,c='b',ls='--',linewidth=5)
plot(range(len(loss1))[3::20],loss1[3::20],c='b',ls='None',marker='o',markersize=10)
plot(loss4,c='b',ls='-',linewidth=5)
plot(range(len(loss4))[3::20],loss4[3::20],c='b',ls='None',marker='o',markersize=10)

plot(loss2,c='b',ls='--',linewidth=5)
plot(range(len(loss2))[3::20],loss2[3::20],c='r',ls='None',marker='x',markersize=10)
plot(loss5,c='b',ls='-',linewidth=5)
plot(range(len(loss5))[3::20],loss5[3::20],c='r',ls='None',marker='x',markersize=10)

plot(loss3,c='k',ls='--',linewidth=5)
plot(range(len(loss3))[3::20],loss3[3::20],c='k',ls='None',marker='s',markersize=10)
plot(loss6,c='k',ls='-',linewidth=5)
plot(range(len(loss6))[3::20],loss6[3::20],c='k',ls='None',marker='s',markersize=10)


tight_layout()
savefig('BASE_EXP/neurons/losses.png')
close()



