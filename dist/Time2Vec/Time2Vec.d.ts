import * as tf from "@tensorflow/tfjs";
declare type activationType = 'sin' | "cos";
export declare class Time2Vec extends tf.layers.Layer {
    output_dim: number;
    p_activation: activationType;
    LatentVectorA: tf.LayerVariable;
    LatentVectorB: tf.LayerVariable;
    biasWeightA: tf.LayerVariable;
    biasWeightB: tf.LayerVariable;
    constructor(output_dim: number, periodic_activation?: activationType);
    build(input_shape: tf.Shape): void;
    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor | tf.Tensor[];
}
export {};
