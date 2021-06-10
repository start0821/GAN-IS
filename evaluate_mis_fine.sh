#! /bin/bash
for lambda1 in 1.0 2.0 3.0 10.0;do
    for lambda2 in 0.1 0.3;do
        echo /root/workspace/Lecture/gct634-2020/env/bin/python -m mnist_calculate_fid --nsamples 60000 --cuda --netG ./log/mIS/finetune/l1_"$lambda1"_l2_"$lambda2"/netG_epoch_99.pth --netD ./log/mIS/finetune/l1_"$lambda1"_l2_"$lambda2"/netD_epoch_99.pth
        /root/workspace/Lecture/gct634-2020/env/bin/python -m mnist_calculate_fid --nsamples 60000 --cuda --netG ./log/mIS/finetune/l1_"$lambda1"_l2_"$lambda2"/netG_epoch_99.pth --netD ./log/mIS/finetune/l1_"$lambda1"_l2_"$lambda2"/netD_epoch_99.pth
        echo /root/workspace/Lecture/gct634-2020/env/bin/python -m mnist_calculate_IS --nsamples 60000 --cuda --netG ./log/mIS/finetune/l1_"$lambda1"_l2_"$lambda2"/netG_epoch_99.pth --netD ./log/mIS/finetune/l1_"$lambda1"_l2_"$lambda2"/netD_epoch_99.pth
        /root/workspace/Lecture/gct634-2020/env/bin/python -m mnist_calculate_IS --nsamples 60000 --cuda --netG ./log/mIS/finetune/l1_"$lambda1"_l2_"$lambda2"/netG_epoch_99.pth --netD ./log/mIS/finetune/l1_"$lambda1"_l2_"$lambda2"/netD_epoch_99.pth
        echo /root/workspace/Lecture/gct634-2020/env/bin/python -m mnist_calculate_mIS --nsamples 60000 --cuda --netG ./log/mIS/finetune/l1_"$lambda1"_l2_"$lambda2"/netG_epoch_99.pth --netD ./log/mIS/finetune/l1_"$lambda1"_l2_"$lambda2"/netD_epoch_99.pth
        /root/workspace/Lecture/gct634-2020/env/bin/python -m mnist_calculate_mIS --nsamples 60000 --cuda --netG ./log/mIS/finetune/l1_"$lambda1"_l2_"$lambda2"/netG_epoch_99.pth --netD ./log/mIS/finetune/l1_"$lambda1"_l2_"$lambda2"/netD_epoch_99.pth    
    done
done
# 1.0 2.0 10.0
