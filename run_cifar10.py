import argparse
import cv2
import json
import numpy as np
import os

import matplotlib.pyplot as plt
from collections import defaultdict
from os.path import exists, join
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, BatchNormalization, ActivityRegularization
from keras.callbacks import (
    LearningRateScheduler, ModelCheckpoint, TensorBoard
)
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.utils import Sequence


class CifarSequence(Sequence):
    """
    Sequence class to handle batch creation
    """

    def __init__(self, X, regions, Y, batch_size, steps_per_epoch):
        self.x = X
        self.regions = regions
        self.y = Y
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.on_epoch_end()

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index:index+self.batch_size]
        x = {'image':self.x[indexes, ...],'region':self.regions[indexes, ...]}
        return x, self.y[indexes, ...]

    def __len__(self):
        return self.steps_per_epoch

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.x.shape[0])
        np.random.shuffle(self.indexes)


def saveim(im, fname, cmap=None):
    """ Helper function for saving images """
    if im.ndim == 3 and im.shape[2] == 3:
        plt.imsave(fname, im)
    else:
        plt.imsave(fname, im, cmap=cmap)


def save_train_examples(X_train, Y_train, regions_train, num_examples, max_regions, res_dir):
    """ Save some training examples, including images, f_p, regions, and F_p"""
    r = RegionAccumulator(max_regions)
    for i in range(num_examples):
        img_I = (X_train[i].transpose(1,2,0)+1.)/2.
        saveim(img_I, join(res_dir, 'cifar_img_train{}.png'.format(i)), cmap=plt.cm.Greens)
        f_p = Y_train[i:i+1]
        saveim(f_p.squeeze(), join(res_dir, 'f_p_train{}.png'.format(i)))
        region = regions_train[i:i+1]
        saveim(region.squeeze(), join(res_dir, 'regions_train{}.png'.format(i)), cmap=plt.cm.Paired)
        # Use RegionAccumulator to show ground truth F(r)
        region_sums = K.eval(r([K.constant(f_p), K.constant(region, dtype=np.int64)]))
        sum_image = np.zeros_like(region, dtype=np.float32)
        for ind, r_sum in enumerate(region_sums.squeeze()):
            num_region = np.sum(region == ind)
            sum_image[region == ind] = r_sum / float(num_region + 1e-7)
        saveim(sum_image.squeeze(), join(res_dir, 'F_r_train{}.png'.format(i)), cmap=plt.cm.Greens)


def create_random_voronoi(size, num_regions, rng):
    """
    Creates a random voronoi image of a certain width and height
    by sampling num_region random points. Returns int64 single channel image.

    Code adapted from:
            https://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/
    """
    w = h = size  # For this project, assume images are square
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for rp in rng.randint(0, w, size=(num_regions, 2)):
        subdiv.insert((rp[0], rp[1]))
    facets, centers = subdiv.getVoronoiFacetList([])
    img = np.zeros((w, h), dtype=np.uint8)
    c = 0
    for i in xrange(0,len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)
        ifacet = np.array(ifacet_arr, np.int)
        cv2.fillConvexPoly(img, ifacet, c, cv2.INTER_NEAREST, 0)  # Do not use another interpolation!!
        c += 1
    return img.astype(np.int64)


class RegionAccumulator(Layer):
    """
    Layer that performs the region accumulation.
    """

    def __init__(self, max_num_regions, **kwargs):
        self.max_num_regions = max_num_regions
        super(RegionAccumulator, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RegionAccumulator, self).build(input_shape)

    def call(self, x):
        # x has two inputs, the per-pixel input and the fixed regions
        bs = K.shape(x[0])[0]
        func = K.reshape(x[0],(bs,1,-1))
        region = K.one_hot(K.flatten(x[1]), self.max_num_regions)
        region = K.reshape(region,(bs,-1,self.max_num_regions))
        return K.squeeze(K.batch_dot(func,region),1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.max_num_regions)

    def get_config(self):
        return {'max_num_regions': self.max_num_regions}


def prep_cifar():
    (X_train, _), (X_test, _) = cifar10.load_data()
    # Merge train and test for now, but will split them back apart later
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X = np.concatenate((X_train,X_test),axis=0).astype(np.float32) / 127.5 - 1.
    assert X.shape[0] == (n_train + n_test)
    return X, n_train


def prep_y_sparse(X):
    """ Creates target Y tensor that is sparse"""
    N,d,h,w = X.shape
    color_vecs = X.transpose(0,2,3,1).reshape(-1,3)
    n_bins = 16
    bins = np.linspace(-1, 1, n_bins)
    disc_color_vecs = np.digitize(color_vecs, bins)
    color_image_inds = [i / (h*w) for i in range(color_vecs.shape[0])]
    c_to_ims = defaultdict(set)
    c_sums = defaultdict(int)

    for color, image_ind in zip(disc_color_vecs, color_image_inds):
        c_to_ims[tuple(color)].add(image_ind)
        c_sums[tuple(color)] += 1
    # Create avg times in images
    c_avg_in_images = {c: c_sums[c] / float(len(c_to_ims[c])) for c in c_sums.keys()}
    # Collect colors/inds in far right/bottom range
    xmin = 29000
    ymax = 10
    # Create x/y points which correspond to # of images and avg times in iamge
    xs = []
    ys = []
    for c in c_sums.keys():
        xs.append(len(c_to_ims[c]))
        ys.append(c_avg_in_images[c])
    p_inds = [i for i in range(len(xs)) if xs[i] > xmin and ys[i] < ymax]
    sparse_colors = [c_sums.keys()[i] for i in p_inds]
    Y = np.zeros((N*h*w))
    for i in range(disc_color_vecs.shape[0]):
            Y[i] = tuple(disc_color_vecs[i]) in sparse_colors
    Y = Y.reshape(N,1,h,w).astype(np.float32)
    return Y


def prep_y_real(X):
    """ Creates target Y tensor that contains real values """
    N,d,h,w = X.shape
    num_centroids = 20
    seed = 0
    rng = check_random_state(seed)
    color_vecs = X.transpose(0,2,3,1).reshape(-1,3)
    source_colors = color_vecs[rng.choice(color_vecs.shape[0],num_centroids),:]
    tmp = cdist(source_colors, color_vecs)
    tmp = np.sort(tmp, axis=0)
    tmp = tmp[0, ...] / tmp[1, ...]
    Y = (tmp).reshape(N,1,h,w).astype(np.float32)
    return Y


def prep_y_step(X):
    """ Creates target Y tensor that is a step function"""
    N,d,h,w = X.shape
    num_centroids = 20
    threshold = 0.4
    seed = 0
    rng = check_random_state(seed)
    color_vecs = X.transpose(0,2,3,1).reshape(-1,3)
    source_colors = color_vecs[rng.choice(color_vecs.shape[0],num_centroids),:]
    tmp = cdist(source_colors, color_vecs)
    tmp = np.sum(tmp < threshold, axis=0)
    Y = (tmp).reshape(N,1,h,w).astype(np.float32)
    return Y


def prep_y_simple(X):
    """ Creates a simple target Y tensor, in this case, binary."""
    N,d,h,w = X.shape
    num_centroids = 15
    threshold = 0.2
    seed = 0
    rng = check_random_state(seed)
    color_vecs = X.transpose(0,2,3,1).reshape(-1,3)
    source_colors = color_vecs[rng.choice(color_vecs.shape[0],num_centroids),:]
    tmp = cdist(source_colors, color_vecs)
    tmp = np.min(tmp, axis=0) < threshold
    Y = (tmp).reshape(N,1,h,w).astype(np.float32)
    return Y


def prep_regions_and_sums(Y, max_regions):
    N,d,h,w = Y.shape
    seed = 0
    rng = check_random_state(seed)
    regions = np.stack([create_random_voronoi(w, max_regions, rng) for _ in range(N)], axis=0)
    regions = regions.reshape(N,1,h,w)
    # Integrate yield functions over regions
    sums = []
    for region in range(max_regions):
        tmp = Y * (regions == region).astype(np.float32)
        sums.append(tmp.sum(axis=(1,2,3)))
    sums = np.asarray(sums).transpose()
    return regions, sums


def get_ex_name(y_fun, disag, activation, act_norm, n_tr_examples, max_iters, max_regions):
    ex_name = 'ex'
    ex_name += '_samples{}'.format(n_tr_examples)
    if y_fun != 'y_sparse':
        ex_name += '_{}'.format(y_fun)
    ex_name += '_disag' if disag else '_unif'
    ex_name += '_{}'.format(activation)
    ex_name += '_actL1Norm' if act_norm else ''
    ex_name += '_e{}_r{}'.format(max_iters, max_regions)
    return ex_name


def train_experiment(X_train,
                     regions_train,
                     sums_train,
                     train_dir,
                     y_fun,
                     disag,
                     activation,
                     act_norm,
                     n_tr_examples,
                     max_iters,
                     max_regions,
                     n_val,
                     steps_per_epoch,
                     decay_lr_iters):
    epochs = max_iters / steps_per_epoch
    decay_lr_epochs = decay_lr_iters / steps_per_epoch
    N,d,h,w = X_train.shape
    # VAL SPLIT
    X_train, X_val = np.split(X_train, (N - n_val,))
    regions_train, regions_val = np.split(regions_train, (N - n_val,))
    sums_train, sums_val = np.split(sums_train, (N - n_val,))

    N,d,h,w = X_train.shape

    # Params
    lr_init = 0.01
    reg = l2(.0001)
    loss = 'mae'
    batch_size = 64

    # Subset of n_tr_examples
    X_train_ex = X_train[:n_tr_examples, ...]
    regions_train_ex = regions_train[:n_tr_examples, ...]
    sums_train_ex = sums_train[:n_tr_examples, ...]

    ex_name = get_ex_name(y_fun, disag, activation, act_norm, n_tr_examples,
                          max_iters, max_regions)
    print 'Running experiment {}'.format(ex_name)
    ex_dir = join(train_dir, ex_name)
    ex_model_path = join(ex_dir, 'model.keras')
    if exists(ex_model_path):
        return

    def step_decay(epoch):
        initial_lrate = lr_init
        drop = 0.5
        epochs_drop = decay_lr_epochs
        lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
        return lrate

    callback_list = [TensorBoard(log_dir=ex_dir),
                     ModelCheckpoint(join(ex_dir, 'model.keras'), save_best_only=True),
                     LearningRateScheduler(step_decay, verbose=True)]

    im = Input(shape=(d,h,w),name='image')
    region = Input(shape=(1,h,w),name='region',dtype='int32')

    out = Conv2D(64,(1,1),activation='relu',kernel_regularizer=reg,name='agg/conv1')(im)
    out = BatchNormalization(axis=1,scale=False,name='agg/bn1')(out)
    out = Conv2D(32,(1,1),activation='relu',kernel_regularizer=reg,name='agg/conv2')(out)
    out = BatchNormalization(axis=1,scale=False,name='agg/bn2')(out)
    out = Conv2D(16,(1,1),activation='relu',kernel_regularizer=reg,name='agg/conv3')(out)
    out = BatchNormalization(axis=1,scale=False,name='agg/bn3')(out)
    Y_pred = Conv2D(1,(1,1),activation=activation,kernel_regularizer=reg,name='agg/output')(out)
    if act_norm:
        Y_pred = ActivityRegularization(0.0001)(Y_pred)

    if disag:
        agg = RegionAccumulator(max_regions, name='agg/accum')([Y_pred,region])
        model_agg = Model(inputs=(im,region), outputs=agg)
        model_agg.compile(Adam(lr=lr_init, amsgrad=True), loss=loss, metrics=[loss])
        generator = CifarSequence(X_train_ex, regions_train_ex, sums_train_ex,
                                  batch_size, steps_per_epoch)
        val_generator = CifarSequence(X_val, regions_val, sums_val,
                                      batch_size, steps_per_epoch)
        model_agg.fit_generator(generator=generator, epochs=epochs, verbose=True,
                                validation_data=val_generator,
                                workers=1, use_multiprocessing=False, callbacks=callback_list)
    else:
        unif_region_sum = np.zeros(regions_train.shape)
        for ix in range(max_regions):
            region_cur = regions_train == ix
            region_cur_area = region_cur.sum(axis=(1,2,3))
            per_pixel_yield = sums_train[:,ix] / (1e-10 + region_cur_area)
            np.copyto(unif_region_sum,per_pixel_yield.reshape(-1,1,1,1), where=region_cur)
        unif_region_sum_val = np.zeros(regions_train.shape)
        for ix in range(max_regions):
            region_cur = regions_train == ix
            region_cur_area = region_cur.sum(axis=(1,2,3))
            per_pixel_yield = sums_train[:,ix] / (1e-10 + region_cur_area)
            np.copyto(unif_region_sum_val,per_pixel_yield.reshape(-1,1,1,1), where=region_cur)

        model_unif = Model(inputs=im, outputs=Y_pred)

        model_unif.compile(Adam(lr=lr_init, amsgrad=True), loss=loss, metrics=[loss])
        generator = CifarSequence(X_train_ex, regions_train_ex, unif_region_sum,
                                  batch_size, steps_per_epoch)
        val_generator = CifarSequence(X_val, regions_val, unif_region_sum_val,
                                      batch_size, steps_per_epoch)
        model_unif.fit_generator(generator=generator, epochs=epochs, verbose=True,
                                 validation_data=val_generator,
                                 workers=1, use_multiprocessing=False, callbacks=callback_list)


def evaluate_experiment(X_test,
                        Y_test,
                        regions_test,
                        sums_test,
                        train_dir,
                        y_fun,
                        result_dir,
                        num_viz_examples,
                        disag,
                        activation,
                        act_norm,
                        n_tr_examples,
                        max_iters,
                        max_regions):
    r = RegionAccumulator(max_regions)
    # Collect experiment name, load model
    ex_name = get_ex_name(y_fun, disag, activation, act_norm, n_tr_examples,
                          max_iters, max_regions)
    ex_dir = join(train_dir, ex_name)
    ex_model_path = join(ex_dir, 'model.keras')
    print 'Evaluating experiment {}'.format(ex_name)
    assert exists(ex_model_path), 'Unable to fine trained model: {}'.format(ex_model_path)
    model = load_model(ex_model_path, custom_objects={'RegionAccumulator': RegionAccumulator})

    if disag:
        model = Model(inputs=model.inputs[0],outputs=model.get_layer('agg/output').output)
        model.compile(SGD(lr=.0), 'mae', metrics=['mae'])  # Only necessary to run model.evaluate()

    _Y_pred = None  # Only run predictions if needed

    for i in range(num_viz_examples):
        i_path = join(result_dir, 'cifar_img_test{}.png'.format(i))
        if not exists(i_path):
            img_I = (X_test[i].transpose(1,2,0)+1.)/2.
            saveim(img_I, i_path)
        f_p_path = join(result_dir, 'f_p_{}_test{}.png'.format(y_fun, i))
        if not exists(f_p_path):
            f_p = Y_test[i:i+1]
            saveim(f_p.squeeze(), f_p_path, cmap=plt.cm.Greens)
        region_path = join(result_dir, 'regions_test{}.png'.format(i))
        if not exists(region_path):
            region = regions_test[i:i+1]
            saveim(region.squeeze(), region_path, cmap=plt.cm.Paired)
        sum_image_path = join(result_dir, 'F_r_{}_test{}.png'.format(y_fun, i))
        if not exists(sum_image_path):
            f_p = Y_test[i:i+1]
            region = regions_test[i:i+1]
            # Use RegionAccumulator to show ground truth F(r)
            region_sums = K.eval(r([K.constant(f_p), K.constant(region, dtype=np.int64)]))
            sum_image = np.zeros_like(region, dtype=np.float32)
            for ind, r_sum in enumerate(region_sums.squeeze()):
                num_region = np.sum(region == ind)
                sum_image[region == ind] = r_sum / float(num_region + 1e-7)
            saveim(sum_image.squeeze(), sum_image_path, cmap=plt.cm.Greens)
        f_hat_p_path = join(result_dir, 'f_hat_p_test_{}_{}.png'.format(ex_name, i))
        if not exists(f_hat_p_path):
            if _Y_pred is None:
                # Get predictions for batch
                _Y_pred = model.predict_on_batch(X_test[0:num_viz_examples,...])
            saveim(_Y_pred[i].squeeze(), f_hat_p_path, cmap=plt.cm.Greens)

    # Load / save results
    try:
        results_json = join(result_dir, 'results.json')
        with open(results_json) as f:
            res_json = json.load(f)
    except IOError:
        res_json = {}

    loss, mae = model.evaluate(x=X_test, y=Y_test.astype(np.float32), batch_size=64)
    res_json[ex_name] = dict(loss=loss,
                             mae=mae,
                             disag=disag,
                             activation=activation,
                             act_norm=act_norm,
                             n_tr_examples=n_tr_examples)
    with open(results_json, 'w') as f:
        json.dump(res_json, f)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run synthetic data CIFAR10 example')
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Directory for output results.')
    parser.add_argument('--train_dir', type=str, default='./training',
                        help='Directory for training outputs and logs.')
    parser.add_argument('--steps_per_epoch', type=int, default=1000,
                        help='Number of steps/iterations to complete an epoch.')
    parser.add_argument('--max_iters', type=int, default=120000,
                        help='Maximum number of iterations.')
    parser.add_argument('--decay_iters', type=int, default=40000,
                        help='Number of iterations after which to decay the '
                        'learning rate')
    parser.add_argument('--y_fun', type=str,
                        choices=['y_simple', 'y_real', 'y_step', 'y_sparse'],
                        default='y_simple',
                        help='Target (Y) function to train')
    parser.add_argument('--max_regions', type=int, default=10,
                        help='Maximum number of regions to accumulate.')
    parser.add_argument('--num_val_examples', type=int, default=2500,
                        help='Number of examples to use as validation')
    parser.add_argument('--pred_activation', type=str, choices=['sigmoid', 'softplus'],
                        default='softplus',
                        help='Activation to use after per-pixel prediction layer.')
    parser.add_argument('--activation_reg', action='store_true',
                        help='If used, will add a regularization penalty to predication layer')
    parser.add_argument('--num_viz_examples', type=int, default=50,
                        help='Number of output images to save')

    args = parser.parse_args()

    # Set up directories
    if not exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not exists(args.train_dir):
        os.makedirs(args.train_dir)

    y_fun = args.y_fun

    args = parser.parse_args()
    print 'Loading Data...'
    X, n_train = prep_cifar()
    if y_fun == 'y_sparse':
        if not exists('Ysparse.npy'):
            Y_sparse = prep_y_sparse(X)
            np.save('Ysparse.npy', Y_sparse)
        else:
            Y_sparse = np.load('Ysparse.npy')
        Y = Y_sparse
    elif y_fun == 'y_real':
        if not exists('Yreal.npy'):
            Y_real = prep_y_real(X)
            np.save('Yreal.npy', Y_real)
        else:
            Y_real = np.load('Yreal.npy')
        Y = Y_real
    elif y_fun == 'y_step':
        if not exists('Ystep.npy'):
            Y_step = prep_y_step(X)
            np.save('Ystep.npy', Y_step)
        else:
            Y_step = np.load('Ystep.npy')
        Y = Y_step
    elif y_fun == 'y_simple':
        if not exists('Ysimple.npy'):
            Y_simple = prep_y_simple(X)
            np.save('Ysimple.npy', Y_simple)
        else:
            Y_simple = np.load('Ysimple.npy')
        Y = Y_simple

    regions_path = 'regions_{}.npy'.format(y_fun)
    sums_path = 'sums_{}.npy'.format(y_fun)
    if not exists(regions_path) or not exists(sums_path):
        regions, sums = prep_regions_and_sums(Y, args.max_regions)
        np.save(regions_path, regions)
        np.save(sums_path, sums)
    else:
        assert exists(regions_path) and exists(sums_path)
        regions = np.load(regions_path)
        sums = np.load(sums_path)
    # Create splits
    X_train, X_test = np.split(X, (n_train,))
    Y_train, Y_test = np.split(Y, (n_train,))
    regions_train, regions_test = np.split(regions, (n_train,))
    sums_train, sums_test = np.split(sums, (n_train,))

    del X, Y, regions, sums

    save_train_examples(X_train, Y_train, regions_train,
                        num_examples=15, max_regions=args.max_regions, res_dir=args.result_dir)

    num_train = int(X_train.shape[0]-args.num_val_examples)
    for disag in [True, False]:
        train_experiment(X_train=X_train,
                         regions_train=regions_train,
                         sums_train=sums_train,
                         train_dir=args.train_dir,
                         y_fun=y_fun,
                         disag=disag,
                         activation=args.pred_activation,
                         act_norm=args.activation_reg,
                         n_tr_examples=num_train,
                         max_iters=args.max_iters,
                         max_regions=args.max_regions,
                         n_val=args.num_val_examples,
                         steps_per_epoch=args.steps_per_epoch,
                         decay_lr_iters=args.decay_iters)
        evaluate_experiment(X_test=X_test,
                            Y_test=Y_test,
                            regions_test=regions_test,
                            sums_test=sums_test,
                            train_dir=args.train_dir,
                            y_fun=y_fun,
                            result_dir=args.result_dir,
                            num_viz_examples=args.num_viz_examples,
                            disag=disag,
                            activation=args.pred_activation,
                            act_norm=args.activation_reg,
                            n_tr_examples=num_train,
                            max_iters=args.max_iters,
                            max_regions=args.max_regions)


if __name__ == '__main__':
    main()
