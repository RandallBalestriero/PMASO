from pylab import *
import cPickle
import glob




DATASET = 'CIFAR'
l_r     = '0.001'
model   = 'smallCNN'


data0 = [[],[]]
data1 = [[],[]]

for b in [0,1]:
	for std in linspace(0,1,5):
		DATA0 = []
		DATA1 = []
		files = glob.glob(DATASET+'_'+model+'_lr'+l_r+'_run*_std'+str(std)+'_b'+str(b))
		print "GLOB",files
		for ff in files:
			print ff
			f=open(ff,'rb')
			train_accu,test_accu = cPickle.load(f)
			f.close()
			print train_accu
			DATA0.append(mean(train_accu[-5:]))
			DATA1.append(mean(test_accu[-5:]))
		data0[b].append(asarray(DATA0).mean())
		data1[b].append(asarray(DATA1).mean())


subplot(121)
plot(data0[0])
plot(data0[1])
subplot(122)
plot(data1[0])
plot(data1[1])

show()



