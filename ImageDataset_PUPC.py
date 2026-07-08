import torch
import random
from torch.utils.data import Dataset


class ImgDataset_train(Dataset):
    def __init__(self, A_ori_img, A_gt, B_ori_img, B_gt, len_num=160000):
        self.A_ori_img = torch.cat((A_ori_img, A_ori_img[1:], A_ori_img[2:], A_ori_img[3:]), dim=0)
        self.A_gt = torch.cat((A_gt, A_gt[1:], A_gt[2:], A_gt[3:]), dim=0)

        self.B_ori_img = B_ori_img
        self.B_gt = B_gt

        self.len_num = len_num

    def __len__(self):
        return self.len_num

    def __getitem__(self, index):
        num_A = 19674
        num_B = 19660

        idx_A = index % (num_A*4-6)

        if index % 4 == 0:
            ori_img_A = self.A_ori_img[idx_A]
            gt_A = self.A_gt[idx_A]

            idx_B = random.randint(0, num_B - 1)
            while torch.argmax(gt_A) != torch.argmax(self.B_gt[idx_B]):
                idx_B = random.randint(0, num_B - 1)

            ori_img_B = self.B_ori_img[idx_B]
            gt_B = self.B_gt[idx_B]

            return [ori_img_A, gt_A, ori_img_B, gt_B]

        else:
            ori_img_A = self.A_ori_img[idx_A]
            gt_A = self.A_gt[idx_A]

            idx_B = random.randint(0, num_B - 1)
            while torch.argmax(gt_A) == torch.argmax(self.B_gt[idx_B]):
                idx_B = random.randint(0, num_B - 1)

            ori_img_B = self.B_ori_img[idx_B]
            gt_B = self.B_gt[idx_B]

            return [ori_img_A, gt_A, ori_img_B, gt_B]


class ImgDataset_test(Dataset):
    def __init__(self, ori_img, gt):
        self.ori_img, self.gt = ori_img, gt

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, index):
        try:
            ori_img = self.ori_img[index]
            gt = self.gt[index]

        except:
            print(index)

        return ori_img, gt

