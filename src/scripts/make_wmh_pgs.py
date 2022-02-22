def make_config(seed, rate, mode, gpu, temp):
    return f"""
#trianing and test splits
val_mode: 'val_heldout'
train_sup_rate: {rate} #rate of labeled data    cvpr rates: 20, 25, 30    (available rates: 5, 10, 20, 25, 30)
#data loader and pre-processing
t1: True
wmh_threshold: 0.5
mixup_threshold: None
intensity_rescale: True
#model
model: "PGS" #PGS,PGSMT, UNET
semi_mode: "shared"   # shared, MT
gradient_stop_strategy: "stop_feature" #stop_feature, stop_output
information_passing_strategy: "teacher" #student, teacher
oneHot: False
num_perturbators: 4
#optimizer
optimizer: "SGD" #SGD OR ADAM
#supervised training setting
supervised_training:
  sup_loss: 'CE' # CE, DSC, FOCAL, CE_DSC, CE_FOCAL, DSC_FOCAL, CE_FOCAL_DSC
  lr: 0.01
  lr_gamma: 0.5
  scheduler_step_size: 20
#unsupervised training
unsupervised_training:
  consistency_training_method: 'layerwise_normal' # 'layerwise_normal' 'layerwise_no_detach'
  loss_method: 'output-wise' # output-wise , down-sample
  consistency_loss: 'MSE' # CE, KL, MSE, DSC
  T: {temp}
  lr: 0.01
  lr_gamma: 0.5
  scheduler_step_size: 30
  consist_w_unsup:
    rampup: 'linear_rampup' #ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    final_w: 10
    rampup_ends: 30
#hyper parameters
experiment_mode: "{mode}"  # [ 'semi_alternate' , 'partially_sup' , 'semi_alternate_mix_F_G']
n_epochs: 100
batch_size: 16
parallel: False
cuda: "{gpu}"
seed: {seed}   #seeds:  41, 42, 43
information: "layerwise2"
"""


if __name__ == '__main__':
    for seed in [41, 42, 43]:
        for rate in [5, 10, 20, 25, 30, 100]:
            for mode in ["fully_sup", "partially_sup", "semi_alternate"]:
                for temp in [0.5, 0.6, 0.7, 0.8]:
                    # for gpu in [0, 1]:
                    # manually select device with CUDA_VISIBLE_DEVICES
                    gpu = 0
                    with open(f'../src/anthony_configs/wmh-{mode}-{rate}-{seed}-{temp}.yaml', 'w') as out:
                        out.write(make_config(seed, rate, mode, gpu, temp))
