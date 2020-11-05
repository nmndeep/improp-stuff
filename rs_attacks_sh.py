from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import cv2

import torch
import time
import math
import torch.nn.functional as F

import numpy as np

from utils import Logger


class RSAttack():
    """
    Sparse-RS

    :param predict:           forward pass function
    :param norm:              type of the attack
    :param n_restarts:        number of random restarts
    :param n_queries:         max number of queries (each restart)
    :param eps:               bound on the sparsity of perturbations
    :param seed:              random seed for the starting point
    :param alpha_init:        parameter to control alphai
    :param loss:              loss function optimized ('margin', 'ce' supported)
    :param resc_schedule      adapt schedule of alphai to n_queries
    :param device             specify device to use
    :param log_path           path to save logfile.txt
    :param constant_schedule  use constant alphai
    """

    def __init__(
            self,
            predict,
            attack,
            norm='L0',
            n_queries=5000,
            eps=None,
            alpha_init=.8,
            n_restarts=1,
            seed=0,
            verbose=False,
            targeted=False,
            loss='margin',
            resc_schedule=True,
            device=None,
            log_path=None,
            constant_schedule=False,
            init_patches='stripes',
            frame_updates = 'squares_inside_frame',
            data_loader=None):
        """
        Sparse-RS implementation in PyTorch
        """

        self.predict = predict
        self.attack = attack
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = alpha_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.targeted = targeted
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.device = device
        self.logger = Logger(log_path)
        self.constant_schedule = constant_schedule
        self.init_patches = init_patches
        self.frame_updates = frame_updates
        self.data_loader = data_loader
        self.resample_loc = 10000

    def margin_and_loss(self, x, y):
        """
        :param y:        correct labels if untargeted else target labels
        """

        logits = self.predict(x)
        xent = F.cross_entropy(logits, y, reduction='none')
        u = torch.arange(x.shape[0])
        y_corr = logits[u, y].clone()
        logits[u, y] = -float('inf')
        y_others = logits.max(dim=-1)[0]

        if not self.targeted:
            if self.loss == 'ce':
                return y_corr - y_others, -1. * xent
            elif self.loss == 'margin':
                return y_corr - y_others, y_corr - y_others
        else:
            return y_others - y_corr, xent

    def init_hyperparam(self, x):
        assert self.norm in ['L0', 'patches', 'frames',
            'patches_universal', 'frames_universal']
        assert not self.eps is None
        assert self.loss in ['ce', 'margin']

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()
        if self.targeted:
            self.loss = 'ce'

    def random_target_classes(self, y_pred, n_classes):
        y = torch.zeros_like(y_pred)
        for counter in range(y_pred.shape[0]):
            l = list(range(n_classes))
            l.remove(y_pred[counter])
            t = self.random_int(0, len(l))
            y[counter] = l[t]

        return y.long().to(self.device)

    def check_shape(self, x):
        return x if len(x.shape) == (self.ndims + 1) else x.unsqueeze(0)

    def random_choice(self, shape):
        t = 2 * torch.rand(shape).to(self.device) - 1
        return torch.sign(t)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def s_selector(self, it):

        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)   ###   change to 100000 for universal frames
        tot_qr = 10000 if self.rescale_schedule else self.n_queries
        return max(2. * (float(tot_qr - it) / tot_qr  - .5) * self.eps/10*2, 0.)
        # if it <= 2000:
        #     return self.eps+8
        #
        # elif 2000 < it <= 4000:
        #     return (self.eps)+6            #l3
        #
        # elif 4000 < it <= 8000:
        #     return (self.eps)+4
        #
        # elif 8000 < it <= 12000:
        #     return  self.eps+2 #(self.eps+2)//2
        #
        # elif 12000 < it <= 16000:
        #     return  self.eps
        #
        # elif 16000 < it <= 20000:
        #     return (self.eps+2)//2
        #
        # elif 20000 < it <= 25000:
        #     return  3 #self.eps//3
        #
        # else:
        #     return self.eps//4

    def p_selection(self, it):
        """ schedule to decrease the parameter alpha """

        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)

        if 'patches' or 'frames' in self.norm:   ###   for square attack

            if 10 < it <= 50:
                p = self.p_init / 2
            elif 50 < it <= 200:
                p = self.p_init / 4
            elif 200 < it <= 500:
                p = self.p_init / 8
            elif 500 < it <= 1000:
                p = self.p_init / 16
            elif 1000 < it <= 2000:
                p = self.p_init / 32
            elif 2000 < it <= 4000:
                p = self.p_init / 64
            elif 4000 < it <= 6000:
                p = self.p_init / 128
            elif 6000 < it <= 8000:
                p = self.p_init / 256
            elif 8000 < it <= 10000:
                p = self.p_init / 512
            else:
                p = self.p_init

        if 'squares_inside_frame' in self.frame_updates:
                tot_qr = 10000 if self.rescale_schedule else self.n_queries
                return max(2. * (float(tot_qr - it) / tot_qr  - .5) * self.p_init, 0.)
        return p

    def get_init_patch(self, c, s, n_iter=1000):
        #print('using {} initialization'.format(self.init_patches))
        if self.init_patches == 'stripes':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device) + self.random_choice(
                [1, c, 1, s]).clamp(0., 1.)
        elif self.init_patches == 'uniform':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device) + self.random_choice(
                [1, c, 1, 1]).clamp(0., 1.)
        elif self.init_patches == 'random':
            patch_univ = self.random_choice([1, c, s, s]).clamp(0., 1.)
        elif self.init_patches == 'random_squares':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device)
            for _ in range(n_iter):
                size_init = torch.randint(low=1, high=math.ceil(s ** .5), size=[1]).item()
                loc_init = torch.randint(s - size_init + 1, size=[2])
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init] = 0.
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init
                    ] += self.random_choice([c, 1, 1]).clamp(0., 1.)

        return patch_univ.clamp(0., 1.)
    def sh_selection(self, it):
        """ schedule to decrease the parameter of shift """

        t = max((float(self.n_queries - it) / self.n_queries - .0) ** 1., 0) * .75

        return t

    def bias_upd(self, inn):
        int_val = int(inn[0].item()*4 + inn[1].item()*2 + inn[2].item())
        r1 = []
        a = range(10)[::-1]                       # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        b = [x for i,x in enumerate(a) if i!=3]
        a = format(int_val, '#05b')
        a = a.replace('0b', '')
        for j in np.arange(8):
            b = format(j, '#05b')
            b = b.replace('0b', '')
            r1.append(b)
        b1 = [x for _,x in enumerate(r1) if x!=a]
        # print("inn", a)
        # print("List after", b1)
        # for j in np.arange(8):
        #     a = bin(int_val)
        #     a = a.replace('0b', '')
        #     b = bin(j)
        #     b = b.replace('0b', '')
        #     r = 0
        #     if len(a)<len(b):
        #         dif = len(b) - len(a)
        #         a = ('0'*dif)+a
        #     else:
        #         dif = len(a) - len(b)
        #         b = ('0'*dif)+b
        #     for i in range(len(a)):
        #         if a[i] != b[i]:
        #             r+=1
        #     r1.append(r)
        # val = np.argmax(r1)
        # stt = '{0:03b}'.format(val)
        # return [int(stt[0]),int(stt[1]),int(stt[2])]
        val = np.random.choice(b1)
        # val = int(val)
        # print("val", val)
        # stt = '{0:03b}'.format(val)
        # print(stt)
        return [int(val[0]),int(val[1]),int(val[2])]

    def attack_single_run(self, x, y):
        with torch.no_grad():
            c, h, w = x.shape[1:]
            n_ex_total = x.shape[0]

            if self.norm == 'L0':
                eps = self.eps

                x_best = x.clone()
                n_pixels = h * w
                b_all, be_all = torch.zeros([x.shape[0], eps]).long(), torch.zeros([x.shape[0], n_pixels - eps]).long()
                for img in range(x.shape[0]):
                    ind_all = torch.randperm(n_pixels)
                    ind_p = ind_all[:eps]
                    ind_np = ind_all[eps:]
                    x_best[img, :, ind_p // w, ind_p % w] = self.random_choice([c, eps]).clamp(0., 1.)
                    b_all[img] = ind_p.clone()
                    be_all[img] = ind_np.clone()

                margin_min, loss_min = self.margin_and_loss(x_best, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)

                for it in range(1, self.n_queries):
                    # check points still to fool
                    idx_to_fool = (margin_min > 0.).nonzero().squeeze()
                    x_curr = self.check_shape(x[idx_to_fool])
                    x_best_curr = self.check_shape(x_best[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]
                    b_curr, be_curr = b_all[idx_to_fool], be_all[idx_to_fool]
                    if len(y_curr.shape) == 0:
                        y_curr.unsqueeze_(0)
                        margin_min_curr.unsqueeze_(0)
                        loss_min_curr.unsqueeze_(0)
                        b_curr.unsqueeze_(0)
                        be_curr.unsqueeze_(0)
                        idx_to_fool.unsqueeze_(0)

                    # build new candidate
                    x_new = x_best_curr.clone()
                    eps_new = max(int(self.p_selection(it) * eps), 1)
                    ind_p = torch.randperm(eps)[:eps_new]
                    ind_np = torch.randperm(n_pixels - eps)[:eps_new]

                    for img in range(x_new.shape[0]):
                        p_set = b_curr[img, ind_p]
                        np_set = be_curr[img, ind_np]
                        x_new[img, :, p_set // w, p_set % w] = x_curr[img, :, p_set // w, p_set % w].clone()
                        x_new[img, :, np_set // w, np_set % w] = self.random_choice([c, eps_new]).clamp(0., 1.)

                    # compute loss of new candidates
                    margin, loss = self.margin_and_loss(x_new, y_curr)
                    n_queries[idx_to_fool] += 1

                    # update best solution
                    idx_improved = (loss < loss_min_curr).float()
                    idx_to_update = (idx_improved > 0.).nonzero().squeeze()
                    loss_min[idx_to_fool[idx_to_update]] = loss[idx_to_update]

                    idx_miscl = (margin < -1e-6).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)
                    nimpr = idx_improved.sum().item()
                    if nimpr > 0.:
                        idx_improved = (idx_improved.view(-1) > 0).nonzero().squeeze()
                        margin_min[idx_to_fool[idx_improved]] = margin[idx_improved].clone()
                        x_best[idx_to_fool[idx_improved]] = x_new[idx_improved].clone()
                        t = b_curr[idx_improved].clone()
                        te = be_curr[idx_improved].clone()

                        if nimpr > 1:
                            t[:, ind_p] = be_curr[idx_improved][:, ind_np] + 0
                            te[:, ind_np] = b_curr[idx_improved][:, ind_p] + 0
                        else:
                            t[ind_p] = be_curr[idx_improved][ind_np] + 0
                            te[ind_np] = b_curr[idx_improved][ind_p] + 0

                        b_all[idx_to_fool[idx_improved]] = t.clone()
                        be_all[idx_to_fool[idx_improved]] = te.clone()

                    # log results current iteration
                    ind_succ = (margin_min <= 0.).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        self.logger.log(' '.join(['{}'.format(it + 1),
                            '- success rate={}/{} ({:.2%})'.format(
                            ind_succ.numel(), n_ex_total,
                            float(ind_succ.numel()) / n_ex_total),
                            '- avg # queries={:.1f}'.format(
                            n_queries[ind_succ].mean().item()),
                            '- med # queries={:.1f}'.format(
                            n_queries[ind_succ].median().item()),
                            '- loss={:.3f}'.format(loss_min.mean()),
                            '- max pert={:.0f}'.format(((x_new - x_curr).abs() > 0
                            ).max(1)[0].view(x_new.shape[0], -1).sum(-1).max()),
                            '- epsit={:.0f}'.format(eps_new),
                            ]))

                    if ind_succ.numel() == n_ex_total:
                        break

            elif self.norm == 'patches':
                '''
                assumes square images and patches
                creates image- and location-specific adversarial patches
                '''
                s = int(math.ceil(self.eps ** .5)) # size of the patches (s x s)

                # initialize patches
                x_best = x.clone()
                x_new = x.clone()
                loc = torch.randint(h - s, size=[x.shape[0], 2])
                patches_coll = torch.zeros([x.shape[0], c, s, s]).to(self.device)

                for counter in range(x.shape[0]):
                    patches_coll[counter] += self.random_choice([c, 1, s]).clamp(0., 1.)
                    x_new[counter, :, loc[counter, 0]:loc[counter, 0] + s,
                        loc[counter, 1]:loc[counter, 1] + s] = patches_coll[counter].clone()

                margin_min, loss_min = self.margin_and_loss(x_new, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)

                for it in range(1, self.n_queries):
                    idx_to_fool = (margin_min > -1e-6).nonzero().squeeze()
                    x_curr = self.check_shape(x[idx_to_fool])
                    patches_curr = self.check_shape(patches_coll[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]
                    loc_curr = loc[idx_to_fool]
                    if len(y_curr.shape) == 0:
                        y_curr.unsqueeze_(0)
                        margin_min_curr.unsqueeze_(0)
                        loss_min_curr.unsqueeze_(0)

                        loc_curr.unsqueeze_(0)
                        idx_to_fool.unsqueeze_(0)

                    s_it = int(max(self.p_selection(it) ** .5 * s, 1))
                    p_it = torch.randint(s - s_it + 1, size=[2])
                    sh_it = int(max(self.sh_selection(it) * h, 0))
                    patches_new = patches_curr.clone()
                    x_new = x_curr.clone()
                    loc_new = loc_curr.clone()
                    loc_t = 5 * (1 + it // 1000)
                    update_loc = int((it % loc_t == 0) and (sh_it > 0))
                    update_patch = 1. - update_loc
                    for counter in range(x_curr.shape[0]):
                        if update_patch == 1.:
                            patches_new[counter, :, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] += self.random_choice([c, 1, 1])
                            patches_new[counter].clamp_(0., 1.)
                        if update_loc == 1:
                            loc_new[counter] += (torch.randint(low=-sh_it, high=sh_it + 1, size=[2]))
                            loc_new[counter].clamp_(0, h - s)
                        x_new[counter, :, loc_new[counter, 0]:loc_new[counter, 0] + s,
                            loc_new[counter, 1]:loc_new[counter, 1] + s] = patches_new[counter].clone()

                    margin, loss = self.margin_and_loss(x_new, y_curr)
                    n_queries[idx_to_fool]+= 1

                    idx_improved = (loss < loss_min_curr).float()
                    idx_to_update = (idx_improved > 0.).nonzero().squeeze()
                    loss_min[idx_to_fool[idx_to_update]] = loss[idx_to_update]

                    idx_miscl = (margin < -1e-6).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)
                    nimpr = idx_improved.sum().item()
                    if nimpr > 0.:
                        idx_improved = (idx_improved.view(-1) > 0).nonzero().squeeze()
                        margin_min[idx_to_fool[idx_improved]] = margin[idx_improved].clone()
                        patches_coll[idx_to_fool[idx_improved]] = patches_new[idx_improved].clone()
                        loc[idx_to_fool[idx_improved]] = loc_new[idx_improved].clone()

                    ind_succ = (margin_min <= 0.).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        self.logger.log(' '.join(['{}'.format(it + 1),
                            '- success rate={}/{} ({:.2%})'.format(
                            ind_succ.numel(), n_ex_total,
                            float(ind_succ.numel()) / n_ex_total),
                            '- avg # queries={:.1f}'.format(
                            n_queries[ind_succ].mean().item()),
                            '- med # queries={:.1f}'.format(
                            n_queries[ind_succ].median().item()),
                            '- loss={:.3f}'.format(loss_min.mean()),
                            '- max pert={:.0f}'.format(((x_new - x_curr).abs() > 0
                            ).max(1)[0].view(x_new.shape[0], -1).sum(-1).max()),
                            '- sit={:.0f} - sh={:.0f}'.format(s_it, sh_it),
                            ]))

                    if ind_succ.numel() == n_ex_total:
                        break

                # creates images with best patches and location found
                for counter in range(x.shape[0]):
                    x_best[counter, :, loc[counter, 0]:loc[counter, 0] + s,
                        loc[counter, 1]:loc[counter, 1] + s] = patches_coll[counter].clone()

            elif self.norm == 'patches_universal':
                '''
                assumes square images and patches
                creates universal patches
                '''
                if not os.path.exists('./results_SQ/patch_black_seed_{}_{}/'.format(self.seed, y[0])):
                    os.makedirs('./results_SQ/patch_black_seed_{}_{}/'.format(self.seed, y[0]))
                s = int(math.ceil(self.eps ** .5))

                x_best = x.clone()

                # the batch is copied two times
                # x = torch.cat((x.clone(), x.clone(), x.clone()), 0)
                # y = torch.cat((y.clone(), y.clone(), y.clone()), 0)
                # n_ex_total *= 3

                x_new = x.clone()
                loc = torch.randint(h - s + 1, size=[x.shape[0], 2])  # fixed location for each image

                # initialize universal patch
                # patch_univ =   torch.load('./results_SQ/patchesnew_{}/patch_60000.pt'.format(y[0]), map_location=torch.device('cuda'))
                patch_univ = self.get_init_patch(c, s)
                #torch.zeros([1, c, s, s]).to(self.device) + self.random_choice(1, c, 1, s]).clamp(0., 1.)

                loss_batch = float(1e10)
                adv_bloss = float(1e10)
                n_succs = 0
                n_queries = torch.ones(x.shape[0]).to(self.device)
                loss_list = []
                it_list = []
                # i_sh = 2161
                # h_sh = 13
                for it in range(0, self.n_queries):
                    # create new candidate patch
                    adv1 = torch.zeros([2, c, x_best.shape[2], x_best.shape[2]]).to(self.device)
                    s_it = int(max(self.p_selection(it) ** .5 * s, 1))
                    if self.attack == 'sparse-rs':
                        p_it = torch.randint(s - s_it + 1, size=[2])

                        patch_new = patch_univ.clone()  # [1, 3, 71, 71]
                        patch_new[0, :, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] += self.random_choice([c, 1, 1])
                        patch_new.clamp_(0., 1.)
                    else:
                        dim = c * s * s
                        patch_flat_new = patch_univ.reshape(-1).clone()  # shape=15123

                        if it == 0:
                            h_sh, i_sh = 0, 0
                        chunk_len = np.ceil(dim / (2 ** h_sh)).astype(int)
                        istart = i_sh * chunk_len
                        iend = min(dim, (i_sh + 1) * chunk_len)
                        # flip 1 to 0 and 0 to 1
                        patch_flat_new[istart:iend] = 1 - patch_flat_new[istart:iend]

                        # update i and h for next iteration
                        i_sh += 1
                        if i_sh == 2 ** h_sh or iend == dim:
                            h_sh += 1
                            i_sh = 0
                            # if all pixels are exhausted, repeat again
                            if h_sh == np.ceil(np.log2(dim)).astype(int) + 1:
                                h_sh = 0

                        patch_new = patch_flat_new.reshape([1, c, s, s])

                    x_new = x.clone()

                    for counter in range(x.shape[0]):
                        loc_new = loc[counter]
                        x_new[counter, :, loc_new[0]:loc_new[0] + s, loc_new[1]:loc_new[1] + s] = 0.
                        x_new[counter, :, loc_new[0]:loc_new[0] + s, loc_new[1]:loc_new[1] + s] += patch_new[0]

                    for cout in range(2):    ###   additional
                        xx = torch.randint(224 - s, size = [1])
                        adv1[cout, :, xx[0]:xx[0] + s, xx[0]:xx[0] + s] += patch_new[0]

                    margin_run, loss_run = self.margin_and_loss(x_new, y)
                    loss_new = loss_run.sum()
                    _, loss_adv1 = self.margin_and_loss(adv1, y[:2])
                    ll = loss_adv1.sum()
                    n_succs_new = (margin_run < -1e-6).sum().item()

                    if loss_new < loss_batch and ll <= adv_bloss: # and n_succs_new >= n_succs:
                        loss_batch = loss_new + 0.
                        patch_univ = patch_new.clone()
                        n_succs = n_succs_new + 0
                        adv_bloxx = ll + 0.

                    if self.verbose:
                        self.logger.log(' '.join(['{}'.format(it + 1),
                                                  '- success rate={}/{} ({:.2%})'.format(
                                                      n_succs, n_ex_total,
                                                      float(n_succs) / n_ex_total),
                                                  '- loss={:.3f}'.format(loss_batch),
                                                  '- max pert={:.0f}'.format(((x_new - x).abs() > 0
                                                                              ).max(1)[0].view(x_new.shape[0], -1).sum(-1).max()),
                                                  '- sit={:.0f}'.format(s_it),
                                                  ]))
                    if it % 10000 == 0 or it == 100000:
                        # loss_list.append(loss_new)
                        torch.save(patch_univ, './results_SQ/patch_black_seed_{}_{}/patch_{}.pt'.format(self.seed, y[0],it))

                    if it % 1000 == 0:
                        it_list.append(it)
                        loss_list.append(loss_new)
                        f=open("./results_SQ/patch_black_seed_{}_{}/loss_over_it.txt".format(self.seed, y[0]), "a+")
                        f.write("Loss at iteration_{} is : {}\r\n".format(it, loss_new))
                        f.close()

                    if it == 99999 or n_succs ==n_ex_total:
                        torch.save(patch_univ, './results_SQ/patch_black_seed_{}_{}/patch_{}.pt'.format(self.seed, y[0],it))

                for counter in range(x_best.shape[0]):
                    loc_new = loc[counter]
                    x_best[counter, :, loc_new[0]:loc_new[0] + s, loc_new[1]:loc_new[1] + s] = 0.
                    x_best[counter, :, loc_new[0]:loc_new[0] + s, loc_new[1]:loc_new[1] + s] += patch_univ[0]

            elif self.norm == 'frames':
                mask = torch.zeros(x.shape[-2:])
                s = self.eps + 0
                mask[:s] = 1.
                mask[-s:] = 1.
                mask[:, :s] = 1.
                mask[:, -s:] = 1.

                ind = (mask == 1.).nonzero().squeeze()
                eps = ind.shape[0]

                x_best = x.clone()
                #x_new = x.clone()
                if self.init_patches =='signhunt':
                    frame_coll = torch.ones([x.shape[0], c, eps]).to(self.device)
                elif self.init_patches == 'stand_patch':
                    frame_coll =  self.random_choice([x.shape[0], c, eps]).clamp(0., 1.)
                else:
                    frame_coll = torch.zeros([x.shape[0], c, eps]).clamp(0., 1.).to(self.device)
                    frame_univ = torch.zeros([1, c, eps]).to(self.device)
                    init_as_patches = self.get_init_patch(c, w)
                    frame_coll[:,:,:] += init_as_patches[:, :, ind[:, 0], ind[:, 1]]

                it_init = 0
                loss_min = float(1e10) * torch.ones_like(y)
                margin_min = loss_min.clone()
                n_queries = torch.ones(x.shape[0]).to(self.device)
                mask_frame = torch.zeros([x.shape[0], c, h, w]).to(self.device)
                #mask_frame[:, :, ind[:, 0], ind[:, 1]] += frame_univ
                #x_new[:, :, ind[:, 0], ind[:, 1]] = 0.
                #x_new[:, :, ind[:, 0], ind[:, 1]] += frame_univ
                n_succs = 0
                n_queries = torch.zeros(x.shape[0]).to(self.device)
                s_it = torch.zeros([2]).long()   # 0 if self.frames_updates == 'standard' else
                ind_it = torch.zeros([x.shape[0], 2]).long()

                # if not self.data_loader is None:
                #     assert self.targeted
                #     new_train_imgs = []
                #     n_newimgs = math.ceil(n_ex_total / 1.)
                #     n_imgsneeded = math.ceil(self.n_queries / self.resample_loc) * n_newimgs
                #     tot_imgs = 0
                #     print('imgs updated={}, imgs needed={}'.format(n_newimgs, n_imgsneeded))
                #     while tot_imgs < min(100000, n_imgsneeded):
                #         x_toupdatetrain, _ = next(self.data_loader)
                #         new_train_imgs.append(x_toupdatetrain)
                #         tot_imgs += x_toupdatetrain.shape[0]
                #     newimgstoadd = torch.cat(new_train_imgs, axis=0)
                #     print(newimgstoadd.shape)
                #     counter_resamplingimgs = 0


                for it in range(0, self.n_queries):
                    startt = time.time()
                    idx_to_fool = (margin_min > -1e-6).nonzero().squeeze()
                    x_curr = self.check_shape(x[idx_to_fool])
                    frame_curr = frame_coll[idx_to_fool]
                    y_curr = y[idx_to_fool]
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]
                    mask_frame_curr = mask_frame[idx_to_fool]
                    ind_it_curr = ind_it[idx_to_fool]

                    if len(y_curr.shape) == 0:
                        y_curr.unsqueeze_(0)
                        margin_min_curr.unsqueeze_(0)
                        loss_min_curr.unsqueeze_(0)
                        frame_curr.unsqueeze_(0)
                        # loc_curr.unsqueeze_(0)
                        idx_to_fool.unsqueeze_(0)
                        mask_frame_curr.unsqueeze_(0)

                    eps_it = max(int(self.p_selection(it) ** 1. * eps), 1)
                    # for i in range(x_curr.shape[0]):
                    # ind_it = torch.randperm(eps)[:eps_it]
                    frame_new = frame_curr.clone()

                    if self.attack =='sparse-rs':
                        if self.frame_updates =='stand':
                            eps_it = 1 #max(int(self.p_selection(it) ** 1. * eps), 1)
                            s_it = max(3* math.ceil(self.s_selector(it) * 4), 1) #self.eps
                            mask_frame_curr[:, :, ind[:, 0], ind[:, 1]] = 0
                            mask_frame_curr[:, :, ind[:, 0], ind[:, 1]] += frame_curr
                            for xr in range(x_curr.shape[0]):
                                dir_h = self.random_choice([1]).long().cpu()
                                dir_w = self.random_choice([1]).long().cpu()
                                ind_new = torch.randperm(eps)[:eps_it]  # update locations
                                vals_new = self.random_choice([c, eps_it]).clamp(0., 1.)  # update values
                                if s_it!=1:
                                    for i_h in range(s_it):
                                        for i_w in range(s_it):
                                            hhh = (ind[ind_new, 0] + dir_h*i_h).clamp(0, h - 1)  # 326 coords
                                            www = (ind[ind_new, 1] + dir_w*i_w).clamp(0, w - 1)  # 326 coords
                                            mask_frame_curr[xr, :, hhh, www] = vals_new.clone()

                                elif s_it == 1:
                                    vall = self.bias_upd(frame_curr[xr,:,ind_new])
                                    vall = np.array(vall)
                                    # print("vall after", vall)
                                    prev = frame_curr[xr,:,ind_new]
                                    neww = vall
                                    vall = torch.from_numpy(vall).unsqueeze_(1)
                                    vall = vall.type(torch.FloatTensor)
                                    for i_h in range(s_it):
                                        for i_w in range(s_it):
                                            hhh = (ind[ind_new, 0] + dir_h*i_h).clamp(0, h - 1)  # 326 coords
                                            www = (ind[ind_new, 1] + dir_w*i_w).clamp(0, w - 1)  # 326 coords
                                            mask_frame_curr[xr, :, hhh, www] = vall.to(self.device)

                            frame_new = mask_frame_curr[:, :, ind[:, 0], ind[:, 1]]  # # [1, 3, 16320]
                            if len(frame_new.shape) == 2:
                                frame_new.unsqueeze_(0)
                            frame_new.clamp_(0., 1.)

                        elif self.frame_updates == 'squares_inside_frame':
                            eps_it_old = eps_it + 0
                            eps_it = max(math.ceil(self.p_selection(it) * self.eps), 1)
                            #print(eps_it)
                            eps_it = min(eps_it, self.eps + 0)
                            s_it = [eps_it, eps_it]
                            if False:
                                ind_it[:,0] = torch.randint(s - s_it[0] + 1, size=[x_curr.shape[0]])
                                ind_it[:,1] = torch.randint(h - s_it[1] + 1 - s, size=[x_curr.shape[0]])
                                sample_loc = random.choice([0, 1, 2, 3])
                                if sample_loc == 1:
                                    ind_it = [ind_it[0] + w - s, ind_it[1] + s]
                                elif sample_loc == 2:
                                    ind_it = [ind_it[1] + s, ind_it[0]]
                                elif sample_loc == 3:
                                    ind_it = [ind_it[1], ind_it[0] + w - s]
                            else:
                                # compute new indices for internal frame if different size of updates
                                if eps_it != eps_it_old:
                                    s_int = s - eps_it + 1
                                    mask.zero_()
                                    mask[:s_int, :-s] = 2.
                                    mask[-s:-s + s_int, s_int:-s + s_int] = 2.
                                    mask[s_int:-s + s_int, :s_int] = 2.
                                    mask[:-s, -s:-s + s_int] = 2.
                                    ind_int = (mask == 2.).nonzero().squeeze()
                                # sample point in internal frame
                                ind_it_curr = ind_int[torch.randint(ind_int.shape[0], size=[x_curr.shape[0],1])].clone().squeeze()

                            mask_frame_curr[:, :, ind[:, 0], ind[:, 1]] = 0
                            mask_frame_curr[:, :, ind[:, 0], ind[:, 1]] += frame_curr
                            if eps_it > 1:
                                for xr in range(x_curr.shape[0]):
                                    # print(xr, frame_curr.shape)
                                    mask_frame_curr[xr, :, ind_it_curr[xr,0]:ind_it_curr[xr,0] + s_it[0], ind_it_curr[xr,1]:ind_it_curr[xr,1] + s_it[1]] = 0.
                                    mask_frame_curr[xr, :, ind_it_curr[xr,0]:ind_it_curr[xr,0] + s_it[0], ind_it_curr[xr,1]:ind_it_curr[xr,1] + s_it[1]
                                        ] += self.random_choice([c, 1, 1]).clamp(0., 1.)
                            else:
                                for xr in range(x_curr.shape[0]):
                                    old_clr = mask_frame_curr[xr, :, ind_it_curr[xr,0]:ind_it_curr[xr,0] + s_it[0], ind_it_curr[xr,1]:ind_it_curr[xr,1] + s_it[1]].clone()
                                    new_clr = old_clr.clone()
                                    while (new_clr == old_clr).all().item():
                                        new_clr = self.random_choice([c, 1, 1]).clone().clamp(0., 1.)
                                    mask_frame_curr[xr, :, ind_it_curr[xr,0]:ind_it_curr[xr,0] + s_it[0], ind_it_curr[xr,1]:ind_it_curr[xr,1] + s_it[1]] = new_clr.clone()

                        frame_new = mask_frame_curr[:, :, ind[:, 0], ind[:, 1]].clone()
                        frame_new.clamp_(0., 1.)
                        if len(frame_new.shape) == 2:
                            frame_new.unsqueeze_(0)


                    else:
                        dim = c * eps
                        for xr in range(x_curr.shape[0]):
                            frame_flat_new = frame_curr[xr,:,:].reshape(-1).clone()  # shape=48960

                            if it == 0:
                                h_sh, i_sh = 0, 0
                            chunk_len = np.ceil(dim / (2 ** h_sh)).astype(int)
                            istart = i_sh * chunk_len
                            iend = min(dim, (i_sh + 1) * chunk_len)
                            # flip 1 to 0 and 0 to 1
                            frame_flat_new[istart:iend] = 1 - frame_flat_new[istart:iend]

                            # update i and h for next iteration
                            i_sh += 1
                            if i_sh == 2 ** h_sh or iend == dim:
                                h_sh += 1
                                i_sh = 0
                                # if all pixels are exhausted, repeat again
                                if h_sh == np.ceil(np.log2(dim)).astype(int) + 1:
                                    h_sh = 0
                            frame_new[xr,:,:] = frame_flat_new.reshape([1, c, eps])
                    loss_new = 0.
                    n_succs_new = 0

                    x_new = x_curr.clone()
                    x_new[:, :, ind[:, 0], ind[:, 1]] = 0.
                    x_new[:, :, ind[:, 0], ind[:, 1]] += frame_new

                    margin, loss = self.margin_and_loss(x_new, y_curr)
                    n_queries[idx_to_fool]+= 1

                    idx_improved = (loss < loss_min_curr).float()
                    idx_to_update = (idx_improved > 0.).nonzero().squeeze()
                    loss_min[idx_to_fool[idx_to_update]] = loss[idx_to_update]

                    idx_miscl = (margin < -1e-6).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)
                    nimpr = idx_improved.sum().item()
                    if nimpr > 0.:
                        idx_improved = (idx_improved.view(-1) > 0).nonzero().squeeze()
                        margin_min[idx_to_fool[idx_improved]] = margin[idx_improved].clone()
                        frame_coll[idx_to_fool[idx_improved]] = frame_new[idx_improved].clone()

                    ind_succ = (margin_min <= 0.).nonzero().squeeze()

                    # if not self.resample_loc is None and self.data_loader is None:
                    #     if (it + 1) % self.resample_loc == 0:
                    #         loc = torch.randint(h - s + 1, size=[x.shape[0], 2])
                    #         #print('locations resampled')
                    #         loss_batch = loss_batch * 0. + 1e6
                    # elif not (self.resample_loc is None or self.data_loader is None):
                    #     if (it + 1) % self.resample_loc == 0:
                    #         #print(x[:, 0, 0, 0])
                    #         newimgstoadd_it = newimgstoadd[counter_resamplingimgs * n_newimgs:(
                    #             counter_resamplingimgs + 1) * n_newimgs].clone().cuda()
                    #         new_batch = [x[n_newimgs:].clone(), newimgstoadd_it.clone()]
                    #         x = torch.cat(new_batch, dim=0)
                    #         assert x.shape[0] == n_ex_total
                    #         #print(x[:, 0, 0, 0])
                    #         #print(loc.T)
                    #         if n_newimgs < n_ex_total:
                    #             loc_newwhenresamplingimgs = [loc[n_newimgs:x.shape[0] - n_newimgs].clone(),
                    #                 torch.randint(h - s + 1, size=[2 * n_newimgs, 2])]
                    #             loc = torch.cat(loc_newwhenresamplingimgs, dim=0)
                    #         else:
                    #             loc = torch.randint(h - s + 1, size=[n_ex_total, 2])
                    #         #print(loc.T)
                    #         assert loc.shape == (n_ex_total, 2)
                    #         loss_batch = loss_batch * 0. + 1e6
                    #         counter_resamplingimgs += 1
                    if self.verbose and ind_succ.numel() != 0 and self.frame_updates =='stand':
                        self.logger.log(' '.join(['{}'.format(it + 1),
                            '- success rate={}/{} ({:.2%})'.format(
                            ind_succ.numel(), n_ex_total,
                            float(ind_succ.numel()) / n_ex_total),
                            '- avg # queries={:.1f}'.format(
                            n_queries[ind_succ].mean().item()),
                            '- med # queries={:.1f}'.format(
                            n_queries[ind_succ].median().item()),
                            '- loss={:.3f}'.format(loss_min.mean()),
                            '- max pert={:.0f}'.format(((x_new - x_curr).abs() > 0
                            ).max(1)[0].view(x_new.shape[0], -1).sum(-1).max()),
                            '- epsit={:.0f} - s_it={:.0f}'.format(eps_it, s_it),
                            ]))

                    if self.verbose and ind_succ.numel() != 0 and self.frame_updates !='stand':
                        self.logger.log(' '.join(['{}'.format(it + 1),
                            '- success rate={}/{} ({:.2%})'.format(
                            ind_succ.numel(), n_ex_total,
                            float(ind_succ.numel()) / n_ex_total),
                            '- avg # queries={:.1f}'.format(
                            n_queries[ind_succ].mean().item()),
                            '- med # queries={:.1f}'.format(
                            n_queries[ind_succ].median().item()),
                            '- loss={:.3f}'.format(loss_min.mean()),
                            '- max pert={:.0f}'.format(((x_new - x_curr).abs() > 0
                            ).max(1)[0].view(x_new.shape[0], -1).sum(-1).max()),
                            '- epsit={:.0f}'.format(eps_it),
                            ]))

                    if ind_succ.numel() == n_ex_total:
                        break

                x_best[:, :, ind[:, 0], ind[:, 1]] = 0.
                x_best[:, :, ind[:, 0], ind[:, 1]] += frame_coll



            elif self.norm == 'frames_universal':

                if not os.path.exists('./results_srs_restest2_seed_{}/frames_{}/'.format(self.seed, y[0])):
                    os.makedirs('./results_srs_restest2_seed_{}/frames_{}/'.format(self.seed, y[0]))

                mask = torch.zeros(x.shape[-2:])
                s = self.eps + 0
                # mask1 = torch.zeros([3,224,224]).to(self.device)
                mask[:s] = 1.
                mask[-s:] = 1.
                mask[:, :s] = 1.
                mask[:, -s:] = 1.

                ind = (mask == 1.).nonzero().squeeze()
                eps = ind.shape[0]
                # for _ in range(10000):
                #     l = np.random.choice([0,1,2,3])
                #     p = torch.randint(low=1, high=6, size=[1]).item()
                #
                #     size_init = torch.randint(low=20, high=100-p, size=[1]).item()
                #     if l ==0:
                #         t = torch.randint(low=0, high=224-size_init, size=[1]).item()
                #         mask1[:,:p, t:t+size_init] = 0.
                #         mask1[:,:p, t:t+size_init] += self.random_choice([c,1, 1]).clamp(0., 1.)
                #     elif l==1:
                #         t = torch.randint(low=0, high=224-size_init, size=[1]).item()
                #         mask1[:,-p:, t:t+size_init] =0.
                #         mask1[:,-p:, t:t+size_init] += self.random_choice([c,1, 1]).clamp(0., 1.)
                #     elif l==2:
                #         t = torch.randint(low=s, high=224-s-size_init, size=[1]).item()
                #         mask1[:,t:t+size_init,:s] = 0.
                #         mask1[:,t:t+size_init,:s] += self.random_choice([c,1, 1]).clamp(0., 1.)
                #     else:
                #         t = torch.randint(low=s, high=224-s-size_init, size=[1]).item()
                #         mask1[:,t:t+size_init, -s:] = 0.
                #         mask1[:,t:t+size_init, -s:] += self.random_choice([c,1, 1]).clamp(0., 1.)
                #
                # frame_univ =  torch.zeros([1,c, eps]).to(self.device) #self.random_choice([1, c, eps]).clamp(0., 1.)
                # for i in range(c):
                #     frame_univ[0,i,0:224*s] += mask1[i,:s,:].flatten()
                # ct=0
                # for ll in range(6,218):
                #     frame_univ[0, :, (224*s)+(s*ct):(224*s)+(s*ct)+s] += mask1[:,ll,:s]
                #     ct+=1
                #     frame_univ[0, :, (224*s)+(s*ct):(224*s)+(s*ct)+s] += mask1[:,ll,-s:]
                #     ct+=1
                # print(ct)
                # for i in range(c):
                #     frame_univ[0, i, 224*s+s*ct:224*s+s*ct+224*s] += mask1[i, -s:,:].flatten()
                #
                # frame_univ.clamp(0., 1.)
                x_best = x.clone()
                frame_univ =  self.random_choice([1, c, eps]).clamp(0., 1.)  # [1, 3, 16320] torch.zeros([1, c, eps]).to(self.device) torch.ones([1, c, eps]).to(self.device)
                # frame_univ1 = torch.reshape(frame_univ, (1, c, eps//109, eps//48))
                # for _ in range(50000):
                #     size_init = torch.randint(low=15, high=45, size=[1]).item()
                #     loc_init = torch.randint(eps//109 - size_init + 1, size=[1])
                #     loc_init1 = torch.randint(eps//48 - size_init + 1, size=[1])
                #     frame_univ1[0, :, loc_init[0]:loc_init[0] + size_init, loc_init1[0]:loc_init1[0] + size_init] = 0.
                #     frame_univ1[0, :, loc_init[0]:loc_init[0] + size_init, loc_init1[0]:loc_init1[0] + size_init
                #         ] += self.random_choice([c, 1, 1]).clamp(0., 1.)
                #
                # frame_univ1.clamp(0., 1.)
                # frame_univ = torch.reshape(frame_univ, (1, c, eps))

                mask_frame_new = torch.zeros([1, c, h, w]).to(self.device)  # [1, 3, 224, 224]
                # x[28, :, ind[:, 0], ind[:, 1]] = 0.
                # x[28, :, ind[:, 0], ind[:, 1]] += frame_univ.squeeze(dim = 0)
                # img2 = np.swapaxes(x[28, :,:,:].cpu().detach().numpy(),2,0)
                # img2 = np.swapaxes(img2,0,1)
                # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                # cv2.imwrite("./results_6_sc_10k60/adv_img_{}.jpg".format(y[0]), img2*255)
                # img2 = np.swapaxes(mask1.cpu().detach().numpy(),2,0)
                # img2 = np.swapaxes(img2,0,1)
                # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                # cv2.imwrite("./results_6_sch_rs/adv_img.jpg", img2*255)
                loss_batch = float(1e10)
                n_succs = 0
                n_queries = torch.ones(x.shape[0]).to(self.device)
                loss_list = []
                it_list = []

                if not self.data_loader is None:
                    assert self.targeted
                    new_train_imgs = []
                    n_newimgs = math.ceil(n_ex_total / 1.)
                    n_imgsneeded = math.ceil(self.n_queries / self.resample_loc) * n_newimgs
                    tot_imgs = 0
                    print('imgs updated={}, imgs needed={}'.format(n_newimgs, n_imgsneeded))
                    while tot_imgs < min(100000, n_imgsneeded):
                        x_toupdatetrain, _ = next(self.data_loader)
                        new_train_imgs.append(x_toupdatetrain)
                        tot_imgs += x_toupdatetrain.shape[0]
                    newimgstoadd = torch.cat(new_train_imgs, axis=0)
                    print(newimgstoadd.shape)
                    counter_resamplingimgs = 0

                for it in range(0, self.n_queries):

                    # if it == 59998 or it == 60005 or it%8000==0:
                    #     img2 = np.swapaxes(x[14,:,:,:].cpu().detach().numpy(),2,0)
                    #     img2 = np.swapaxes(img2,0,1)
                    #     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    #     cv2.imwrite("./results_6_seed7/adv_img_{}_{}.png".format(it, y[0]), img2*255)
                    eps_new = max(int(self.p_selection(it) ** 1. * eps), 1)
                    s_it = self.s_selector(it) #self.eps

                    # create new candidate frame
                    mask_frame_new[:, :, ind[:, 0], ind[:, 1]] = 0
                    mask_frame_new[:, :, ind[:, 0], ind[:, 1]] += frame_univ

                    if self.attack == 'sparse-rs':
                        dir_h = self.random_choice([1]).long().cpu()
                        dir_w = self.random_choice([1]).long().cpu()
                        ind_new = torch.randperm(eps)[:eps_new]  # update locations
                        vals_new = self.random_choice([c, eps_new]).clamp(0., 1.)  # update values
                        if s_it!=1:
                            for i_h in range(s_it):
                                for i_w in range(s_it):
                                    hhh = (ind[ind_new, 0] + dir_h*i_h).clamp(0, h - 1)  # 326 coords
                                    www = (ind[ind_new, 1] + dir_w*i_w).clamp(0, w - 1)  # 326 coords
                                    mask_frame_new[0, :, hhh, www] = vals_new.clone()

                        elif s_it == 1:
                            vall = self.bias_upd(frame_univ[0,:,ind_new])
                            vall = np.array(vall)
                            # print("vall after", vall)
                            prev = frame_univ[0,:,ind_new]
                            neww = vall
                            vall = torch.from_numpy(vall).unsqueeze_(1)
                            vall = vall.type(torch.FloatTensor)
                            for i_h in range(s_it):
                                for i_w in range(s_it):
                                    hhh = (ind[ind_new, 0] + dir_h*i_h).clamp(0, h - 1)  # 326 coords
                                    www = (ind[ind_new, 1] + dir_w*i_w).clamp(0, w - 1)  # 326 coords
                                    mask_frame_new[0, :, hhh, www] = vall.to(self.device)

                        frame_new = mask_frame_new[:, :, ind[:, 0], ind[:, 1]]  # # [1, 3, 16320]
                        if len(frame_new.shape) == 2:
                            frame_new.unsqueeze_(0)
                    else:
                        dim = c * eps
                        frame_flat_new = frame_univ.reshape(-1).clone()  # shape=48960

                        if it == 0:
                            h_sh, i_sh = 0, 0
                        chunk_len = np.ceil(dim / (2 ** h_sh)).astype(int)
                        istart = i_sh * chunk_len
                        iend = min(dim, (i_sh + 1) * chunk_len)
                        # flip 1 to 0 and 0 to 1
                        frame_flat_new[istart:iend] = 1 - frame_flat_new[istart:iend]

                        # update i and h for next iteration
                        i_sh += 1
                        if i_sh == 2 ** h_sh or iend == dim:
                            h_sh += 1
                            i_sh = 0
                            # if all pixels are exhausted, repeat again
                            if h_sh == np.ceil(np.log2(dim)).astype(int) + 1:
                                h_sh = 0
                        frame_new = frame_flat_new.reshape([1, c, eps])
                        # import ipdb;ipdb.set_trace()
                    x_new = x.clone()
                    x_new[:, :, ind[:, 0], ind[:, 1]] = 0.
                    x_new[:, :, ind[:, 0], ind[:, 1]] += frame_new

                    margin_run, loss_run = self.margin_and_loss(x_new, y)
                    if self.loss == 'ce':
                        loss_run += x_new.shape[0]

                    loss_new = loss_run.sum()
                    # loss_new = (loss_run * (margin_run > -1e-6).float()).sum()  # old version of the code
                    n_succs_new = (margin_run < -1e-6).sum().item()

                    if loss_new < loss_batch: # and n_succs_new >= n_succs:
                        loss_batch = loss_new + 0.
                        frame_univ = frame_new.clone()
                        n_succs = n_succs_new + 0
 # sample new locations
                    # if n_succs_new > n_succs:
                    #     f=open("./results_bias_srs_10k_76542/frames_{}/updates.txt".format(y[0]), "a+")
                    #     f.write("iteration_{} : {}, {}\r\n".format(it, prev, neww))
                    #     f.close()

                    # if not it>=60000:
                    if not self.resample_loc is None and self.data_loader is None:
                        if (it + 1) % self.resample_loc == 0:
                            loc = torch.randint(h - s + 1, size=[x.shape[0], 2])
                            #print('locations resampled')
                            loss_batch = loss_batch * 0. + 1e6
                    elif not (self.resample_loc is None or self.data_loader is None):
                        if (it + 1) % self.resample_loc == 0:
                            #print(x[:, 0, 0, 0])
                            newimgstoadd_it = newimgstoadd[counter_resamplingimgs * n_newimgs:(
                                counter_resamplingimgs + 1) * n_newimgs].clone().cuda()
                            new_batch = [x[n_newimgs:].clone(), newimgstoadd_it.clone()]
                            x = torch.cat(new_batch, dim=0)
                            assert x.shape[0] == n_ex_total
                            #print(x[:, 0, 0, 0])
                            #print(loc.T)
                            if n_newimgs < n_ex_total:
                                loc_newwhenresamplingimgs = [loc[n_newimgs:x.shape[0] - n_newimgs].clone(),
                                    torch.randint(h - s + 1, size=[2 * n_newimgs, 2])]
                                loc = torch.cat(loc_newwhenresamplingimgs, dim=0)
                            else:
                                loc = torch.randint(h - s + 1, size=[n_ex_total, 2])
                            #print(loc.T)
                            assert loc.shape == (n_ex_total, 2)
                            loss_batch = loss_batch * 0. + 1e6
                            counter_resamplingimgs += 1
                            '''print('locations and images resampled')
                            if counter_resamplingimgs == 1:
                                sys.exit()'''
                    if self.verbose:
                        self.logger.log(' '.join(['{}'.format(it + 1),
                            '- success rate={}/{} ({:.2%})'.format(
                            n_succs, n_ex_total,
                            float(n_succs) / n_ex_total),
                            '- loss={:.3f}'.format(loss_batch),
                            '- max pert={:.0f}'.format(((x_new - x).abs() > 0
                            ).max(1)[0].view(x_new.shape[0], -1).sum(-1).max()),
                            '- epsit={:.0f} - s_it={:.0f}'.format(eps_new, s_it),
                            ]))
                    if it % 500 == 0 or it == 100000:
                        # loss_list.append(loss_new)

                        torch.save(frame_univ, './results_srs_restest2_seed_{}/frames_{}/patch_{}.pt'.format(self.seed,y[0],it))

                    if it % 500 == 0:
                        it_list.append(it)
                        loss_list.append(loss_new)
                        f=open("./results_srs_restest2_seed_{}/frames_{}/loss_over_it.txt".format(self.seed, y[0]), "a+")
                        f.write("Loss at iteration_{} is : {}\r\n".format(it, loss_new))
                        f.close()


                    if it == 99999:
                        torch.save(frame_univ, './results_srs_restest2_seed_{}/frames_{}/patch_{}.pt'.format(self.seed, y[0],it))

                x_best[:, :, ind[:, 0], ind[:, 1]] = 0.
                x_best[:, :, ind[:, 0], ind[:, 1]] += frame_univ

        return n_queries, x_best

    def perturb(self, x, y=None):
        """
        :param x:           clean images
        :param y:           untargeted attack -> clean labels,
                            if None we use the predicted labels
                            targeted attack -> target labels, if None random classes,
                            different from the predicted ones, are sampled
        """

        self.init_hyperparam(x)

        adv = x.clone()
        qr = torch.zeros([x.shape[0]]).to(self.device)
        if y is None:
            if not self.targeted:
                with torch.no_grad():
                    output = self.predict(x)
                    y_pred = output.max(1)[1]
                    y = y_pred.detach().clone().long().to(self.device)
            else:
                with torch.no_grad():
                    output = self.predict(x)
                    n_classes = output.shape[-1]
                    y_pred = output.max(1)[1]
                    y = self.random_target_classes(y_pred, n_classes)
        else:
            y = y.detach().clone().long().to(self.device)

        if not self.targeted:
            acc = self.predict(x).max(1)[1] == y
        else:
            acc = self.predict(x).max(1)[1] != y

        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        np.random.seed(self.seed)

        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()

                qr_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)

                output_curr = self.predict(adv_curr)
                if not self.targeted:
                    acc_curr = output_curr.max(1)[1] == y_to_fool
                else:
                    acc_curr = output_curr.max(1)[1] != y_to_fool
                ind_curr = (acc_curr == 0).nonzero().squeeze()

                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                qr[ind_to_fool[ind_curr]] = qr_curr[ind_curr].clone()
                if self.verbose:
                    print('restart {} - robust accuracy: {:.2%}'.format(
                        counter, acc.float().mean()),
                        '- cum. time: {:.1f} s'.format(
                        time.time() - startt))

        return qr, adv
