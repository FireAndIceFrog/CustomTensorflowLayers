import * as tf from "@tensorflow/tfjs";
export declare function scaled_attention(query: tf.Tensor | tf.SymbolicTensor, key: tf.Tensor | tf.SymbolicTensor, values: tf.Tensor | tf.SymbolicTensor, mask: tf.Tensor | tf.SymbolicTensor | undefined): (tf.Tensor<tf.Rank> | tf.SymbolicTensor)[];
export declare class MultiHeadAttention extends tf.layers.Layer {
    num_heads: number;
    d_model: number;
    depth: number;
    wq: tf.layers.Layer;
    wk: tf.layers.Layer;
    wv: tf.layers.Layer;
    dense: tf.layers.Layer;
    scaled_attention(q: tf.Tensor | tf.SymbolicTensor, k: tf.Tensor | tf.SymbolicTensor, v: tf.Tensor | tf.SymbolicTensor, mask: tf.Tensor | tf.SymbolicTensor | undefined): (tf.Tensor<tf.Rank> | tf.SymbolicTensor)[];
    split_heads(x: tf.Tensor | tf.SymbolicTensor, batch_size: number): tf.Tensor<tf.Rank> | tf.SymbolicTensor;
    call(input: tf.Tensor[] | tf.Tensor | tf.SymbolicTensor[] | tf.SymbolicTensor): tf.Tensor[];
    constructor({ d_model, num_heads }: {
        d_model: number;
        num_heads: number;
    });
}
