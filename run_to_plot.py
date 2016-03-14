from matplotlib import pyplot as plt

FILENAME = 'all_updates.txt'

def get_coords(filename=FILENAME, Type='val_loss'):
    linecount = 0
    f = open('./'+filename, 'r')
    l = f.readline()
    vals = []
    corrs = []
    curves = [[]]
    count = 0
    while len(l)>0:
        linecount += 1
        if Type=='val_loss':
            if str(l[:17]) == '  validation loss':
                curves[-1].append(float(l[20:].strip()))
        if Type=='acc':
            if str(l[:17]) == '  validation accu':
                curves[-1].append(float(l[22:-2].strip()))
        if str(l[:10]) == ' ----- Par':
            curves.append([])
            corrs.append(l)
            count += 1
            vals = []
        l = f.readline()
    f.close()

    corrs.append(l)

    x = [len(y) for y in curves]

    curves = curves[1:]
    corrs = corrs[:-1]

    return curves, corrs


def plot(curves, corrs=None, Type='val_loss'):
    # min_epochs:  in case different tests ran for different numbers of epochs 
    min_epochs = 60000 # ie much larger than the number of epochs would be
    fig = plt.figure()
    for c in curves:
        min_epochs = min(min_epochs, len(c))
    r = list(range(min_epochs))
    plt.xlabel('Epoch')
    '''if Type=='acc':
        for j,c in enumerate(curves):
            for i in range(len(c)):
                curves[j][i] = 1-curves[j][i]'''
    for i,c in enumerate(curves):
        plt.plot(r,c[:min_epochs], label=corrs[i][32:-1].strip())

    if Type=='val_loss':
        plt.title('Training Losses Standard Task')
        plt.ylabel('Training Loss')
        plt.legend(loc=3)
        txt = 'Figure 1.  Training losses for each kind of parameter update are \n\
                given.  Each was trained for 20 epochs with typical hyperparameters. '
    if Type=='acc':
        plt.title('Validation Accuracy for the Standard Task')
        plt.ylabel('Validation Error %')
        plt.legend(loc=4)
        txt = 'Figure 2.  Validation accuracies on test sets for each kind of \n\
                parameter update are given.  Each was trained for 20 epochs \n\
                with typical hyperparameters. '
    plt.gca().set_position((.1, .3, .8, .6))
    fig.text(.1,.1,txt)
    plt.show()


if __name__=='__main__':
    A,B = get_coords()
    plot(A,B)
 
