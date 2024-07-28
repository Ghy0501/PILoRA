import torch
import copy
import shutil
import torch.nn as nn
import torch.optim as optim
import math
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import sys
import numpy as np
from VLT import *
from utils import *
from update import *
from tqdm import tqdm
from iCIFAR100 import iCIFAR100
from data_manager_tiny import *
from CPN import *
# from t_sneplot import *

class vitlora:
    def __init__(self, args, file_name, model, task_size, device):
        self.data_dir = '../../data'
        self.file_name = file_name
        self.args = args
        self.epochs = args.local_ep
        self.model = model
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.data_manager = DataManager()
        self.global_model = VLT(modelname='vit_base_patch16_224_dino',
                                num_classes=self.args.total_classes,
                                pretrained=True,
                                r = 4,
                                lora_layer = [0],
                                g_model=True)
        self.trans_train = transforms.Compose([
                            transforms.Resize(224),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ColorJitter(brightness=0.24705882352941178),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.trans_test = transforms.Compose([
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.trainfolder = self.data_manager.get_dataset(self.trans_train, index=list(range(200)), train=True)
        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None
        self.classes = None
        self.old_model = None
        self.list_of_testloader = list()
        self.W_aq = []
        self.W_av = []
        self.W_bq = []
        self.W_bv = []

    def inference(self, model, test_loader):
        model.eval()
        test_loss = 0.0
        correct = 0.0
        extracted_features = []
        extracted_label = []
        extracted_centers = []
        
        with torch.no_grad():
            for setp, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                extracted_label.extend(target.cpu().data.numpy())
                predict, feature, dist, centers, _ = model(data, mode='test')

                extracted_features.extend(feature.detach().cpu().numpy().tolist())
                extracted_centers = centers.T.detach().cpu().numpy()

                logits, test_loss = DCE(target, dist, type='test')
                pred = torch.max(logits, 1)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        # t_SNEplot(extracted_features, extracted_label, extracted_centers, self.numclass)
        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        return acc, test_loss
    
    def save_checkpoint(self, state, is_best):
        filename = '{}/{}/ckpt.pth.tar'.format(os.path.abspath(os.path.dirname(os.getcwd())) + '/checkpoints',
                                            self.args.store_name)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

    def map_new_class_index(self, y, order):
        return np.array(list(map(lambda x: order.index(x), y)))

    def setup_data(self, shuffle, seed):
        train_targets = self.trainfolder.labels
        order = [i for i in range(len(np.unique(train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = range(len(order))
        self.class_order = order
        if seed == 0:
            self.class_order = [i for i in range(len(np.unique(train_targets)))]
        print(100*'#')

        self.class_mask = build_continual_dataset(self.args, self.class_order)
        print(self.class_mask)

    def beforeTrain(self, current_task):
        self.model.eval()
        class_set = list(range(200))
        if current_task == 0:
            self.classes = class_set[:self.numclass]
        else:
            self.classes = class_set[self.numclass-self.task_size: self.numclass]
        self.model.centers_initial(self.classes)
        self.model.switch_lora(current_task)
        
        self.testfolder = self.data_manager.get_dataset(self.trans_test, index=class_set[:self.numclass], train=False)
        self.validfolder = self.data_manager.get_dataset(self.trans_test, index=class_set[:self.numclass], train=False, test=False)

        # self.train_loader = torch.utils.data.DataLoader(self.trainfolder, batch_size=self.args.batch_size,
        #     shuffle=True, drop_last=True, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(self.testfolder, batch_size=self.args.local_bs,
            shuffle=False, drop_last=False, num_workers=8)
        self.valid_loader = torch.utils.data.DataLoader(self.validfolder, batch_size=self.args.local_bs,
            shuffle=False, drop_last=False, num_workers=8)

        self.model.train()
        self.model.to(self.device)
        self.global_model.to(self.device)

    def train(self, current_task, old_class=0, tf_writer=None, logger_file=None):
        bst_acc = -1
        description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
        local_weights = []
        center_lr = self.args.centers_lr
        encoder_lr = self.args.encoders_lr

        for epoch in tqdm(range(self.args.epochs)):
            local_weights = []
            count_num= [ [] for i in range(0, self.task_size) ]
            feature_list = [ [] for i in range(0, self.task_size) ]
            m = self.args.client_local
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False)
            sample_num = []
            # load dataset and user groups

            for idx in idxs_users:
                train_dataset, user_groups = get_dataset(self.args, trans_train=self.trans_train, m=m,
                                                         class_set=self.classes, task_num=self.task_size)

                map_dict = {}
                for (key, value), new_key in zip(user_groups.items(), idxs_users):
                    map_dict[new_key] = value
                user_groups = map_dict

                sample_num.append(len(user_groups[idx]))

                local_model = LocalUpdate(args=self.args, dataset=train_dataset,
                                        idxs=user_groups[idx])
                w, feature_average = local_model.update_weights(
                    model=copy.deepcopy(self.model), old_model=copy.deepcopy(self.old_model), lr_c=center_lr, lr_e=encoder_lr,
                    current_task=current_task, Waq=self.W_aq, Wav=self.W_av)
                local_weights.append(copy.deepcopy(w))
                for key, values in feature_average.items():
                    feature_list[key%20].append(np.array(values))
            average_weight = [i/sum(sample_num) for i in sample_num]

            # # update global weights
            # self.model.load_state_dict(self.model)
            self.model.load_state_dict(average_weights(local_weights, self.model, self.classes,
                                                        self.args.niid_type, feature_list, average_weight))
            self.global_model = global_server(self.model, self.global_model, self.W_aq, self.W_av, self.W_bq, self.W_bv, current_task)

            valid_acc, valid_loss = self.inference(self.global_model, self.valid_loader)
            test_acc, test_loss = self.inference(self.global_model, self.test_loader)

            # center_lr, encoder_lr = center_lr*0.95, encoder_lr*0.95
            # if epoch < self.args.epochs/2:
            #     center_lr = self.args.centers_lr
            #     encoder_lr = self.args.encoders_lr
            # else:
            center_lr = self.args.centers_lr*0.5*(1 + math.cos(epoch * math.pi / self.args.epochs))
            encoder_lr = self.args.encoders_lr*0.5*(1 + math.cos(epoch * math.pi / self.args.epochs))

            tf_writer.add_scalar('test_acc', test_acc, epoch)
            tf_writer.add_scalar('test_loss', test_loss, epoch)

            output_log = 'After {} global rounds, Test acc: {}, inference loss: {}'.format(
                epoch + 1, test_acc, test_loss)
            logger_file.write(output_log + '\n')
            output_log = 'After {} global rounds, Valid acc: {}, Valid loss: {}'.format(
                epoch + 1, valid_acc, valid_loss)
            logger_file.write(output_log + '\n')
            logger_file.flush()

            is_best = test_acc > bst_acc
            bst_acc = max(bst_acc, test_acc)
            #print(description.format(test_acc, test_loss, bst_acc))
                
            self.save_checkpoint(self.model.state_dict(), is_best)
            
        print(description.format(test_acc, test_loss, bst_acc))

    def afterTrain(self, current_task):
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        self.numclass += self.task_size
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        torch.save(self.global_model, filename)
        for name, param in self.model.named_parameters():
            if 'linear_a_q_{}'.format(current_task) in name:
                self.W_aq.append(param)
            if 'linear_a_v_{}'.format(current_task) in name:
                self.W_av.append(param)
            if 'linear_b_q_{}'.format(current_task) in name:
                self.W_bq.append(param)
            if 'linear_b_v_{}'.format(current_task) in name:
                self.W_bv.append(param)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()