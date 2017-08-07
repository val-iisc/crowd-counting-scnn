# utils.py
# Some useful functions


import datetime
import cPickle
import lasagne


# Log details
def log(f, txt, do_print = 1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')

    
# Load saved network weights
def load_nets(files, networks):
    print 'Loading nets...'
    for file_name, net in zip(files, networks):
        fp = open(file_name, 'rb')
        params = cPickle.load(fp)
        lasagne.layers.set_all_param_values(net.output_layer, params)
        print ' >', net.name
        fp.close()
    print ' > Done.'


# Save network weights
def save_nets(files, networks):
    print 'Saving nets...'
    for file_name, net in zip(files, networks):
        fp = open(file_name, 'wb')
        params = lasagne.layers.get_all_param_values(net.output_layer)
        cPickle.dump(params, fp, protocol = cPickle.HIGHEST_PROTOCOL)
        print ' >', net.name
        fp.close()
    print ' > Done.'


