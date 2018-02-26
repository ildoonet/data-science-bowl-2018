# Data Science Bawl 2018

## Architectures

### Segmentation

- network_basic
  - FCN. 4 convolution + 1 convolution, multiple feature concatenation.
  - HyperParameters
    - lr=0.01
    - batchsize=32
  - LB : 0.195 @ Validation Loss=0.0908, decay_steps=200, decay_rate=0.33 (basic_lr=0.010_epoch=20_bs=32_180225T1155)
  - LB : 0.221 @ Validation loss=0.0613, decay_steps=300, decay_rate=0.33 (basic_lr=0.010_epoch=30_bs=32_180225T1148) 
