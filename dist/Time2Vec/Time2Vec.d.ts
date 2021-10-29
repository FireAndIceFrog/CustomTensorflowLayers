import * as tf from "@tensorflow/tfjs";
declare type activationType = 'sin' | "cos";
export declare class Time2Vec extends tf.layers.Layer {
    k: number;
    p_activation: activationType;
    wb: tf.LayerVariable;
    bb: tf.LayerVariable;
    wa: tf.LayerVariable;
    ba: tf.LayerVariable;
    constructor(kernel_size: number, periodic_activation?: activationType);
    build(inputShape: tf.Shape): void;
    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor | tf.Tensor[];
    computeOutputShape(input_shape: tf.Shape): tf.Shape;
}
export {};
