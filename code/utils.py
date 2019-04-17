import numpy as np

def get_log_ticks(start, stop, style='e'):

    def smart_strip(str):
        for i, x in enumerate(str):
            str[i] = x.rstrip('0')
            if str[i][-1] == '.':

                str[i] += '0'

        return str
    ticks = np.array([])
    ints = np.arange(np.floor(start),np.ceil(stop)+1)
    for int_now in ints:
        ticks = np.append(ticks, np.arange(0.1,1.0,0.1)*10.0**int_now)
        #ticks = np.unique(ticks)
    ticks = np.append(ticks, 1.0*10.0**ints[-1])

    if style=='e':
        labels = ['%1.1e' % tick for tick in ticks]
        ticks_to_keep = ['%1.1e' % 10**int for int in ints]
    elif style=='f':
        labels = ['%1.12f' % tick for tick in ticks]
        ticks_to_keep = ['%1.12f' % 10**int for int in ints]

    ticks_to_keep = np.unique(ticks_to_keep)

    ticks = np.log10(ticks)


    for i, label in enumerate(labels):

        if label in ticks_to_keep:
            #print('Not removing label')
            continue
        else:
            #print('Removing label')
            labels[i] = ' '

    labels = smart_strip(labels)
    return ticks, labels
