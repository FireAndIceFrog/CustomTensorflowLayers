import * as tf from "@tensorflow/tfjs";
export declare function scaled_attention(query: tf.Tensor, key: tf.Tensor, values: tf.Tensor, mask: tf.Tensor | undefined): tf.Tensor<tf.Rank>[];
export declare class ScaledAttentionLayer extends tf.layers.Layer {
    constructor();
    computeOutputShape(inputShape: tf.Shape): number[];
    call(inputs: tf.Tensor[]): tf.Tensor<tf.Rank>;
}
