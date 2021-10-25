import { LayerVariable, Shape } from "@tensorflow/tfjs-layers";
import { Tensor } from "@tensorflow/tfjs-core";
import { Layer } from "@tensorflow/tfjs-layers/dist/exports_layers";
declare type activationType = 'sin' | "cos";
declare type Kwargs = {
    [key: string]: any;
};
export declare class Time2Vec extends Layer {
    k: number;
    p_activation: activationType;
    wb: LayerVariable;
    bb: LayerVariable;
    wa: LayerVariable;
    ba: LayerVariable;
    constructor(kernel_size: number, periodic_activation?: activationType);
    build(inputShape: Shape): void;
    apply(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    compute_output_shape(input_shape: Shape): number;
}
export {};
