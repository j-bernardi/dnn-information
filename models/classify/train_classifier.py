"""" map a segmentation map to the four referral decisions and the ten additional diagnoses (see Supplementary Fig. 16)."""

# We also used a small amount (0.05) of label-smoothing regularization
# and added some (1 × 10−5) weight decay.
# takes as input a 300 × 350 × 43 subsampling of the original 448 × 512 × 128 segmentation map
# the output is a 14-component vector.
# The inputs are one-hot encoded
# and augmented by random three-dimensional affine and elastic transformations [14]
# The loss was the sum of the softmax cross entropy loss for the first four components (multi-class referral decision)
# the sigmoid cross entropy losses for the remaining ten components (additional diagnoses labels)
# Adam optimiser[40]
# for 160,000 iterations of batch size 8
# spread across 8 GPUs with 1 sample per GPU with dataset 3 in Supplementary Table 3.
"""
The initial learning rate was 0.02
and set to 0.02/2 after 10% of the total iterations,
0.02/4 after 20%,
0.02/8 after 50%,
0.02/64 after 70%,
0.02/256 after 90% and
finally 0.02/512 for the final 5% of training.
"""
