import * as tf from "@tensorflow/tfjs";
import { MultiHeadAttention } from "./Attention";
export declare function pointWiseFeedForwardNetwork(d_model: number, dff: number): tf.Sequential;
export declare class Encoder extends tf.layers.Layer {
    multiheadAttention: MultiHeadAttention;
    ffn: tf.Sequential;
    layernorm1: tf.layers.Layer;
    layernorm2: tf.layers.Layer;
    dropout1: tf.layers.Layer;
    dropout2: tf.layers.Layer;
    constructor(d_model: number, num_heads: number, dff: number, rate?: number);
    call(inputs: [tf.Tensor, tf.Tensor | undefined], { training }: {
        training: boolean;
    }): tf.Tensor<tf.Rank>;
}
