from train_net import *
import time

# ['sgd', 'momentum', 'adadelta', 'rmsprop', 'adam', 'nesterov']
for pum in ['rmsprop']:
    print ' ----------------------------- '
    print ' ----- Parameter update method:', pum
    print str(time.localtime())
    train_net(update_method=pum)
    print str(time.localtime())


