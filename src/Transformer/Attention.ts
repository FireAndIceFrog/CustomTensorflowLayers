import * as tf from "@tensorflow/tfjs"

export function scaled_attention(query: tf.Tensor, key: tf.Tensor, values: tf.Tensor, mask: tf.Tensor | undefined) {
    const matmul_qk = tf.matMul(query, key, false, true);
    const dk = tf.cast(key.shape[-1], "float32");
    // (Q * K) / sqrt(dk) is the scaled dot product of query and key
    const scaled_attention_logits = matmul_qk.div(tf.sqrt(dk));

    if(mask) {
        scaled_attention_logits.add(mask.mul(-1e9));
    }

    const attention_weights = tf.softmax(scaled_attention_logits, -1);

    const output = tf.matMul(attention_weights, values)
    return [output, attention_weights]
}

export class ScaledAttentionLayer extends tf.layers.Layer {
    constructor() {
        super({});
    }

    computeOutputShape(inputShape: tf.Shape) {
        return [inputShape[0], inputShape[2]]
    }

    call(inputs: tf.Tensor[]) {
        const [query, key, values, mask] = inputs;
        const [output] = scaled_attention(query, key, values, mask);
        return output;
    }
}