#! /bin/bash
#From scratch_avg
for lambda1 in 1 1 1 1 1 1 1 1 1 1 1;do
    /root/workspace/Lecture/gct634-2020/env/bin/python -m mnist_calculate_IS --nsamples 60000 --cuda  --netG ./log/total/scratch/l1_1.0_l2_1.7_l3_0.7/netG_epoch_99.pth --netD ./log/total/scratch/l1_1.0_l2_1.7_l3_0.7/netD_epoch_99.pth
done
# 1.0 2.0 10.0
