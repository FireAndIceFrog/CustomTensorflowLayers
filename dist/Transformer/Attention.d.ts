import * as tf from "@tensorflow/tfjs";
export declare function scaled_attention(query: tf.Tensor, key: tf.Tensor, values: tf.Tensor, mask: tf.Tensor | undefined): tf.Tensor<tf.Rank>[];
export declare class MultiHeadAttention extends tf.layers.Layer {
    num_heads: number;
    d_model: number;
    depth: number;
    wq: tf.layers.Layer;
    wk: tf.layers.Layer;
    wv: tf.layers.Layer;
    dense: tf.layers.Layer;
    scaled_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor | undefined): tf.Tensor<tf.Rank>[];
    split_heads(x: tf.Tensor, batch_size: number): tf.Tensor<tf.Rank>;
    call([value, key, query, mask]: tf.Tensor[]): tf.Tensor[];
    constructor({ d_model, num_heads }: {
        d_model: number;
        num_heads: number;
    });
}
