dataset=cifar10; size=40; valid=100; time=48; folder=debug; arch=resnet #shake #
kimg=32768; ratio=7; con=0.95; wd=5e-4; batch=64; lr=0.03; i=0; aug="d.d.d"; seed=3; mom=0.9; bal=0;delt=0; wu=1; set_wu=1; 

#wu=1; batch=64; lr=0.03; wd=5e-4; time=48; bal=0; delt=0; bootfactor=0; folder=fixmatch #fross #fixmatch #hp_tests;
#batch=32; wclr=0; lr=0.05; wd=8e-4; time=60; bal=4; delt=0.2; bootfactor=0; folder=boss #fross #fixmatch #hp_tests;
wd=4e-4;lr=0.1; batch=64;
folder=cycle #lr #pla #fixmatch #add-ons # schedule #ablation #supervised #boss #factor #valclass #newLRsched #
#bal=4; delt=0.25
bootfactor=16; valacc=50; cycle=0
kimg=8192 #kimg=16384; time=60

echo " "  >> history
#jobname=FR-LuxWu0 #R-LclrxWclr0 # 
#jobname=FR-WDxWd0 #FR-BSxBatch0 #FR-LRxLr0 #
#jobname=FM-64KxKimg0 #Bp-KxKimg0 #FR-KxKimg0BFxBoot0
jobname=PLA-CYxCycle0KxKimg0 #VAxValacc0KxKimg0 #MUD-LRxLr0KxKimg0 #Bp-KxKimg0 #FR-KxKimg0BFxBoot0
#jobname=FR-KxKimg0AxClraug  #R-KxKimg0BSxBschedule0
#jobname=PLA-WDxWd0KxKimg0 #PLA-LRxLr0KxKimg0 #PLA-CYxCycle0KxKimg0 #PLA-LRxLr2KxKimg0 #
#jobname=PLA-SxSeed0KxKimg0

# Corresponds 512   384   256  192   128   64 Epochs
#for kimg in 32768 24576 16384 12288 8192 4096; do
for kimg in 16384 12288 8192; do
#for kimg in 16384 12288; do
#for kimg in 32768 16384 12288 8192; do
#for kimg in 32768 24576 ; do

#for valacc in 85 80; do
for cycle in 8 4 0; do
#for bootfactor in 8 12; do
#for valid in 60 50; do

#for seed  in 0 1 2 3 4 5; do
#for dataset in cifar10 cifar10p; do
#for wu in 1 4; do

#for batch  in 64 128; do
#for lr  in 0.1 0.2 0.3; do
#for wd  in 3e-4 4e-4 6e-4; do
#for aug in "rac.m.rac" "d.m.rac" "aac.m.aac" "d.m.aac"; do
#for aug in "d.d.d" "d.x.d" "d.d.aac" "d.d.rac" "d.x.aac" "d.x.rac"; do
#for i  in 1 2 3; do
if [ $kimg -eq 4096 ];
then
time=16
#time=60 #60 #48
elif [ $kimg -eq 8192 ];
then
time=24
#time=120 #96
elif [ $kimg -eq 12288 ];
then
time=36
#time=168
elif [ $kimg -eq 16384 ];
then
time=48
#time=96 #117 #168
else
time=96
fi
#time=26
    filename="${dataset}.${seed}@${size}-${valid}Iter${kimg}U${ratio}C${con}D${delt}WD${wd}BS${batch}LR${lr}Aug${aug}Bal${bal}"
    echo $filename
    echo $filename >> history
    sed -e "s/xJob0/${jobname}/g" -e "s/xData0/${dataset}/g" -e "s/xSeed0/${seed}/g" -e "s/xSize0/${size}/g" -e "s/xValid0/${valid}/g" -e "s/xTime/$time/g" -e "s/xKimg0/$kimg/g" -e "s/xUratio0/$ratio/g" -e "s/xConf0/$con/g" -e "s/xArch0/$arch/g" -e "s/xWd0/$wd/g" -e "s/xWu0/$wu/g" -e "s/xBatch0/$batch/g" -e "s/xLr0/$lr/g" -e "s/xDelt0/$delt/g" -e "s/xMom0/$mom/g" -e "s/xAug0/$aug/g" -e "s/xBal0/$bal/g" -e "s/xValacc0/$valacc/g" -e "s/xCycle0/$cycle/g" -e "s/xBoot0/$bootfactor/g" -e "s/xFolder0/${folder}/g" 4x.pbs > Q/$filename
#    exit
    qsub  Q/$filename
    sleep 1
done
done
exit
    sed -e "s/xData0/${dataset}/g" -e "s/xSeed0/${seed}/g" -e "s/xSize0/${size}/g" -e "s/xValid0/${valid}/g" -e "s/xTime/$time/g" -e "s/xKimg0/$kimg/g"\
 -e "s/xUratio0/$ratio/g" -e "s/xConf0/$con/g" -e "s/xArch0/$arch/g" -e "s/xWd0/$wd/g" -e "s/xWu0/$wu/g" -e "s/xBatch0/$batch/g" -e "s/xLr0/$lr/g" -e "\
s/xDelt0/$delt/g" -e "s/xMom0/$mom/g" -e "s/xRep0/2/g" -e "s/xClr/$wclr/g"  -e "s/xRep1/3/g" -e "s/xAug0/$aug/g" -e "s/xBal0/$bal/g" -e "s/xFolder0/$fo\
lder/g" 2x.pbs > Q/$filename
    exit
    qsub  Q/$filename
    sleep 1
done
#done
qstat
exit

if [ $wclr -eq 0 ];
then
#  wd=5e-4;lr=0.03; batch=64;  aug="d.d.d"; wu=1; temper=1; clrratio=1; time=38; folder=fixmatch
  wd=8e-4;lr=0.05; batch=32;  aug="d.d.d"; wu=1; temper=1; clrratio=1; time=48; bal=4; delt=0.2; ratio=9; folder=boss
elif [ $wclr -eq 1 ];
then
#  wd=8e-4;lr=0.06; batch=32;  aug="d.d.rac"; wu=2; temper=2; clrratio=4; time=48; folder=fross
#  wd=8e-4;lr=0.06; batch=32;  aug="d.d.d"; wu=2; temper=2; time=48; kimg=6350; folder=hp_tests
i=1
fi
exit
# Fully supervised baseline without mixup (not shown in paper since Mixup is better)
python fully_supervised/fs_baseline.py --train_dir experiments/fs --dataset=cifar10-1 --wd=0.02 --smoothing=0.001
python fully_supervised/fs_baseline.py --train_dir experiments/fs --dataset=cifar100-1 --wd=0.02 --smoothing=0.001
python fully_supervised/fs_baseline.py --train_dir experiments/fs --dataset=svhn-1 --wd=0.002 --smoothing=0.01
python fully_supervised/fs_baseline.py --train_dir experiments/fs --dataset=svhn_noextra-1 --wd=0.002 --smoothing=0.01

if [ $kimg -eq 3175 ];
then
time=16
#time=30 #60 #48
elif [ $kimg -eq 6300 ];
then
time=24
#time=48 #120 #96
elif [ $kimg -eq 9425 ];
then
time=32
#time=72 #168
elif [ $kimg -eq 12550 ];
then
time=40
#time=96 #117 #168
fi
