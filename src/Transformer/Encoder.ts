import * as tf from "@tensorflow/tfjs"
import { MultiHeadAttention } from "./Attention";

export function pointWiseFeedForwardNetwork(d_model: number, dff: number) {
    return tf.sequential({
        layers: [
            tf.layers.dense({
                units: dff,
                activation: 'relu',
                kernelInitializer: 'glorotNormal',
                name: 'first-dense',
                inputShape: [d_model]
            }),
            tf.layers.dense({
                units: d_model,
                kernelInitializer: 'glorotNormal',
                name: 'second-dense'
            })
        ]
    });
}

export class Encoder extends tf.layers.Layer {
    multiheadAttention: MultiHeadAttention;
    ffn: tf.Sequential;
    layernorm1: tf.layers.Layer;
    layernorm2: tf.layers.Layer;
    dropout1: tf.layers.Layer;
    dropout2: tf.layers.Layer;

    constructor(d_model: number, num_heads: number, dff: number, rate=0.1) {
        super({
            name: 'encoder'
        });
        this.multiheadAttention = new MultiHeadAttention({d_model, num_heads});
        this.ffn = pointWiseFeedForwardNetwork(d_model, dff);

        this.layernorm1 = tf.layers.layerNormalization({
            epsilon: 1e-6,
            name: 'layernorm1'
        });
        this.layernorm2 = tf.layers.layerNormalization({
            epsilon: 1e-6,
            name: 'layernorm2'
        });

        this.dropout1 = tf.layers.dropout({
            rate: rate,
            name: 'dropout1'
        });
        this.dropout2 = tf.layers.dropout({
            rate: rate,
        });
    }

    call(inputs: [tf.Tensor, tf.Tensor | undefined], {training}: {training: boolean}) {
        let [x, mask] = inputs;

        let [attentionOutput] = this.multiheadAttention.call([x, x, x, mask]); //(batch_size, input_seq_len, d_model)
        attentionOutput = this.dropout1.apply(attentionOutput, {training}) as tf.Tensor;
        const out1 = this.layernorm1.apply(x.add(attentionOutput)) as tf.Tensor;

        let ffnOutput = this.ffn.apply(out1) as tf.Tensor; //(batch_size, input_seq_len, d_model)
        ffnOutput = this.dropout2.apply(ffnOutput, {training}) as tf.Tensor;
        const out2 = this.layernorm2.apply(out1.add(ffnOutput)) as tf.Tensor;

        return out2;
    }

}