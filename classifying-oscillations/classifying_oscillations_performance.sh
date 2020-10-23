#! /bin/bash

MILESTONES="35"
OUTPUT=$1
EPOCHS=50
i=0

printf "%s\n" "arch,noise_intens,n_freqs,n_params,acc." > $OUTPUT

N=20
for NOISE in 1.0 2.0 4.0 8.0;
do
    for ARCH in fully_connected_tiny_shallow fully_connected_small_shallow \
                fully_connected_medium_shallow fully_connected_large_shallow \
                fully_connected_tiny_2hidden fully_connected_small_2hidden \
                fully_connected_medium_2hidden fully_connected_large_2hidden \
                resnet20_tiny_const resnet20_small_const resnet20_medium_const resnet20_large_const \
                resnet20_tiny_incr resnet20_small_incr resnet20_medium_incr resnet20_large_incr \
                resnet8_small_incr resnet20_small_incr resnet32_small_incr resnet44_small_incr \
                resnet56_small_incr resnet110_small_incr resnet272_small_incr;
        do
        ((i=i+1))
        echo "Setting $i"
        # Run train.py script and get best accuracy from the output
        ACC=$(
        python train.py --arch $ARCH --noise_intens $NOISE --epochs $EPOCHS --n_freq $N --milestones $MILESTONES --dropout 0|
         grep "Best accuracy = " |
          sed 's/Best accuracy =//')
        # Run train.py script with zero epochs just to get the number of parameters
        NPARAMS=$(
        python train.py --arch $ARCH --noise_intens $NOISE --epochs 0 --n_freq $N --milestones $MILESTONES --dropout 0|
         grep "num of parameters = " |
          sed 's/num of parameters =//')
        echo "Best acc. = $ACC"
        echo "n params = $NPARAMS"
        echo "**************************************"
        printf "%s,%.2f,%d,%d,%2.3f\n" $ARCH $NOISE $N $NPARAMS $ACC >> $OUTPUT
    done;
done