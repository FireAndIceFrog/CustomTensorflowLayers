import * as tf from "@tensorflow/tfjs"
import { MultiHeadAttention } from "./Attention";
import  { pointWiseFeedForwardNetwork } from "./Encoder"

export class DecoderLayer extends tf.layers.Layer {
    mha1: MultiHeadAttention;
    mha2: MultiHeadAttention;

    ffn: tf.Sequential;

    layernorm1: tf.layers.Layer;
    layernorm2: tf.layers.Layer;
    layernorm3: tf.layers.Layer;

    dropout1: tf.layers.Layer;
    dropout2: tf.layers.Layer;
    dropout3: tf.layers.Layer;

    constructor(d_model: number, num_heads: number, dff: number, rate=0.1) {
        super({
            name: 'DecoderLayer',
        });

        this.mha1 = new MultiHeadAttention({d_model, num_heads});
        this.mha2 = new MultiHeadAttention({d_model, num_heads});

        this.ffn = pointWiseFeedForwardNetwork(d_model, dff);

        this.layernorm1 = tf.layers.layerNormalization({epsilon: 1e-6});
        this.layernorm2 = tf.layers.layerNormalization({epsilon: 1e-6});
        this.layernorm3 = tf.layers.layerNormalization({epsilon: 1e-6});

        this.dropout1 = tf.layers.dropout({rate: rate});
        this.dropout2 = tf.layers.dropout({rate: rate});
        this.dropout3 = tf.layers.dropout({rate: rate});
    }

    apply(inputs: tf.Tensor<tf.Rank>[], kwargs: any) {
        const result = this.call(inputs, kwargs)
        return result;
    }

    call(inputs: tf.Tensor[], kwargs: any): tf.Tensor[] {
        const [x, encoder_outputs, look_ahead_mask, padding_mask] = inputs;
        let [attn1, attn_weights_block1] = this.mha1.call([x, x, x, look_ahead_mask])
        attn1 = this.dropout1.apply(attn1, kwargs) as tf.Tensor;

        const out1 = this.layernorm1.apply(attn1.add(x), kwargs) as tf.Tensor;

        let [attn2, attn_weights_block2] = this.mha2.call([encoder_outputs, encoder_outputs, out1, padding_mask])
        attn2 = this.dropout2.apply(attn2, kwargs) as tf.Tensor;
        const out2 = this.layernorm2.apply(attn2.add(out1), kwargs) as tf.Tensor;

        let ffn_output = this.ffn.apply(out2);
        ffn_output = this.dropout3.apply(ffn_output, kwargs) as tf.Tensor;
        const out3 = this.layernorm3.apply(ffn_output.add(out2), kwargs) as tf.Tensor;

        return [out3, attn_weights_block1, attn_weights_block2];
    }
}