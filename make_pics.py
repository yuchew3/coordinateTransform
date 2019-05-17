import numpy as np
import pandas
import ast
import json
import matplotlib.pyplot as plt
import ca_data_utils
import skimage.io

def window_size_pic():
    names = np.load('../data/clf_results/clf_names.npy')
    test_accu = np.zeros((3,12))
    train_accu = np.zeros((3,12))
    for i in range(2,12):
        test_entry = np.load('../data/clf_results/test_accuracy_'+str(i)+'.npy')
        test_accu[:,i] = test_entry
        train_entry = np.load('../data/clf_results/train_accuracy_'+str(i)+'.npy')
        train_accu[:,i] = train_entry
    
    window_sizes = range(2,12)
    figure = plt.figure()
    ax = plt.subplot(111)
    ax.plot(window_sizes, test_accu[0,2:], 'b-', label=names[0]+' test accuracy')
    ax.plot(window_sizes, train_accu[0,2:], 'b--', label=names[0]+' train accuracy')
    ax.plot(window_sizes, test_accu[1,2:], 'r-', label=names[1]+' test accuracy')
    ax.plot(window_sizes, train_accu[1,2:], 'r--', label=names[1]+' train accuracy')
    ax.plot(window_sizes, test_accu[2,2:], 'g-', label=names[2]+' test accuracy')
    ax.plot(window_sizes, train_accu[2,2:], 'g--', label=names[2]+' train accuracy')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('window size')
    ax.set_ylabel('accuracy')

    plt.show()
    # plt.savefig(figure, 'window_size')

def gridCV():
    with open('../data/results.txt', 'r') as f:
        s = f.read().replace('\n','')
        f.close()
    entries = s.split('\t')
    df = dict()
    for e in entries:
        line = e.split(':', 2)
        key = line[0].replace('\'', '')
        v = line[1].strip().replace('array([', '[').replace('])',']')
        try:
            df[key] = ast.literal_eval(v)
        except:
            df[key] = v
    test = 0
    train = 0
    test_accu = np.zeros(28)
    train_accu = np.zeros(28)
    for k in df.keys():
        if 'test_score' in k and 'split' in k:
            test_accu += np.array(df[k])
            test += 1
        if 'train_score' in k and 'split' in k:
            train_accu += np.array(df[k])
            train += 1
    test_accu /= test
    train_accu /= train

    gamma = [9.9999999999999995e-07,1.0000000000000001e-05,0.0001,0.001,0.01,0.10000000000000001,1.0]
    C = [0.10000000000000001, 1.0, 10., 100.]

    # scattor plot over two variables
    test_accu = test_accu.reshape((4,7))
    train_accu = train_accu.reshape((4,7))
    fig = plt.figure()
    ax1 = plt.subplot(1,2,1)
    im = ax1.imshow(test_accu)
    ax1.set_xticks(np.arange(len(gamma)))
    ax1.set_yticks(np.arange(len(C)))
    ax1.set_xticklabels(gamma)
    ax1.set_yticklabels(C)
    for i in range(len(C)):
        for j in range(len(gamma)):
            text = ax1.text(j, i, round(test_accu[i, j],4),
                        ha="center", va="center", color="w")
    ax1.set_xlabel('gamma (kernel coefficient)')
    ax1.set_ylabel('C (penalty)')
    ax1.set_title('test accuracy')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax2 = plt.subplot(1,2,2)
    im2 = ax2.imshow(train_accu)
    ax2.set_xticks(np.arange(len(gamma)))
    ax2.set_yticks(np.arange(len(C)))
    ax2.set_xticklabels(gamma)
    ax2.set_yticklabels(C)
    for i in range(len(C)):
        for j in range(len(gamma)):
            text = ax2.text(j, i, round(train_accu[i, j],4),
                        ha="center", va="center", color="w")
    ax2.set_xlabel('gamma (kernel coefficient)')
    ax2.set_ylabel('C (penalty)')
    ax2.set_title('train accuracy')
    plt.tight_layout()
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.show()

    # figure = plt.figure()
    # test_accu = test_accu.reshape((4,7))
    # train_accu = train_accu.reshape((4,7))
    # # print(np.average(test_accu, axis=0)) # gamma

    # ax1 = plt.subplot(2,1,1)
    # ax1.plot(gamma, test_accu[2], label='test accuracy')
    # ax1.plot(gamma, train_accu[2], label='train accuracy')
    # ax1.legend()
    # ax1.set_ylabel('mean accuracy')
    # ax1.set_xlabel('gamma (C = 10)')
    # ax1.set_xticks(gamma)


    # ax2 = plt.subplot(2,1,2)
    # ax2.plot(C, test_accu[:,3], label='test accuracy')
    # ax2.plot(C, train_accu[:,3], label='train accuracy')
    # ax2.legend()
    # ax2.set_ylabel('mean accuracy')
    # ax2.set_xlabel('C (gamma = 0.001)')
    # ax2.set_xticks(C)
    # plt.tight_layout()
    # plt.show()

def make_video():
    vid = skimage.io.imread('../data/vid.tif')[9:39992]
    labels = ca_data_utils.load_labels()[9:39992]
    preds = np.load('../data/clf_results/y_pred.npy')
    i = 0
    for frame, label, pred in zip(vid, labels, preds):
        print('startint ', i)
        print(frame.shape)
        f, (ax1,ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]})
        ax1.imshow(frame)
        color='g'
        if (label != pred):
            color='r'
        circle = plt.Circle((0.5, 0), 0.2, color=color)
        ax2.add_artist(circle)
        fname = '../data/clf_results/video/image_{0:05d}'.format(i)
        i += 1
        print(fname)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        f.savefig(fname)


if __name__ == '__main__':
    # window_size_pic()
    make_video()