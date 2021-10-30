import * as tf from "@tensorflow/tfjs";
export declare function create_padding_mask(seq: tf.Tensor): tf.Tensor;
export declare function create_look_ahead_mask(seq_len: number): tf.Tensor;
