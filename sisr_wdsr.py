import os
import matplotlib.pyplot as plt

from data import DIV2K
# from model.edsr import edsr
from train import WdsrTrainer
import tensorflow as tf
from model.wdsr_weight_norm import wdsr_a, wdsr_b


# Number of residual blocks
depth = [1, 3, 5, 8]

# Super-resolution factor
scale = 4

# Downgrade operator
downgrade = 'bicubic'

div2k_train = DIV2K(scale=scale, subset='train', downgrade=downgrade)
div2k_valid = DIV2K(scale=scale, subset='valid', downgrade=downgrade)

train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
valid_ds = div2k_valid.dataset(
    batch_size=1, random_transform=False, repeat_count=1)


wdsra_psnr = {}

for i in depth:
    '''Store Model Weights. Will have |depth| different folders = 4'''
    weights_dir = f'weights/wdsr-a-{i}-x{scale}'
    weights_file = os.path.join(weights_dir, 'weights.h5')
    os.makedirs(weights_dir, exist_ok=True)

    '''instantiate training mechanism'''
    trainer = WdsrTrainer(model=wdsr_a(scale=scale, num_res_blocks=i),
                          checkpoint_dir=f'.ckpt/wdsr-a-{i}-x{scale}')
    '''
    Train. 10k total steps. Evaluate, print PSNR and Val Loss (MAE) every 1000 steps
    Checkpoint best model.
    '''
    trainer.train(train_ds,
                  valid_ds.take(10),
                  steps=10000,
                  evaluate_every=1000,
                  save_best_only=True)
    '''Fetch Model From above'''
    trainer.restore()

    '''Store weights in weight directory'''
    trainer.model.save_weights(weights_file)

    '''Fetch total params - Can delete later'''
    model = wdsr_a(scale=scale, num_res_blocks=i)
    model.load_weights(weights_file)
    print(model.count_params())

    '''Get PSNR on full vali set'''
    psnrv = trainer.evaluate(valid_ds)
    print('Res_blocks: ', i, end='')
    print(f'PSNR = {psnrv.numpy():3f}')
    wdsra_psnr[i] = psnrv.numpy()
