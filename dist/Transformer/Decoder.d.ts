import * as tf from "@tensorflow/tfjs";
import { MultiHeadAttention } from "./Attention";
export declare class DecoderLayer extends tf.layers.Layer {
    mha1: MultiHeadAttention;
    mha2: MultiHeadAttention;
    ffn: tf.Sequential;
    layernorm1: tf.layers.Layer;
    layernorm2: tf.layers.Layer;
    layernorm3: tf.layers.Layer;
    dropout1: tf.layers.Layer;
    dropout2: tf.layers.Layer;
    dropout3: tf.layers.Layer;
    constructor(d_model: number, num_heads: number, dff: number, rate?: number);
    call(inputs: tf.Tensor[], kwargs: any): tf.Tensor[];
}
