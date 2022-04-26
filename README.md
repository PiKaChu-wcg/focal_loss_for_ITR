# My Experiment Result

Base on VSE and VSE++, and I add the InfoNCE and Focal loss to make the model focus on the hard negative sample.

## Encoder 
The image encoder is mobilenetV3-large,
the text encoder is GRU (with default 256 word embedding size).

## loss 

**five types loss**:

1. the triplet loss (with semi-hard)
2. the infoNCE
3. the Focal triplet loss (using softmax to get focal factor)
4. the Focal triplet loss (using mlp to get focal factor)
5. the FocalInfoNCE 

## experiment detail

| K          | V        |
| ---------- | -------- |
| vocab_path | ./vocab/ |
| dim_image  | 1280     |
| dim        | 512      |
| dim_word   | 256      |
| margin     | 0.2      |
| epochs     | 30       |
| batch_size | 128      |
| lrate      | 0.001    |

## result

| loss    | finetune | batch size | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@med | t2i R@1 | t2i R@5 | t2i R@10 | t2i R@med |
| ------- | -------- | ---------- | ------- | ------- | -------- | --------- | ------- | ------- | -------- | --------- |
| triplet | False    | 256        | 22.98   | 50.99   | 62.33    | 5         | 16.77   | 40.55   | 53.21    | 9         |
| InfoNCE | False    | 256        | 27.51   | 55.82   | 66.17    | 4         | 19.51   | 44.52   | 57       | 7         |
| triplet | True     | 64         | 35.01   | 63.61   | 73.82    | 3         | 25.78   | 59.59   | 66.29    | 5         |
| InfoNCE | True     | 64         | 40.63   | 68.84   | 78.3     | 2         | 31.07   | 54.51   | 70.73    | 3         |
| triplet | True     | 128        | 46.55   | 77.91   | 86.04    | 2         | 36.77   | 67      | 77.97    | 2.5       |
| InfoNCE | True     | 128        | 53.16   | 79.59   | 87.18    | 1         | 40.87   | 70.02   | 79.53    | 2         |
| triplet | True     | 256        | 58.59   | 83.76   | 91.71    | 1         | 45.93   | 76.76   | 86.2     | 2         |
| InfoNCE | True     | 256        | 63.29   | 85.83   | 92.13    | 1         | 50.40   | 79.02   | 87.33    | 2         |
