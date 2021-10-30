import * as tf from "@tensorflow/tfjs"

export function scaled_attention(query: tf.Tensor, key: tf.Tensor, values: tf.Tensor, mask: tf.Tensor | undefined) {
    const matmul_qk = tf.matMul(query, key, false, true);
    const dk = tf.cast(key.shape[key.shape.length-1], "float32");
    // (Q * K) / sqrt(dk) is the scaled dot product of query and key
    const scaled_attention_logits = matmul_qk.div(tf.sqrt(dk));

    if(mask) {
        scaled_attention_logits.add(mask.mul(-1e9));
    }

    const attention_weights = tf.softmax(scaled_attention_logits, -1);

    const output = tf.matMul(attention_weights, values)
    return [output, attention_weights]
}

export class MultiHeadAttention extends tf.layers.Layer {
    num_heads: number;
    d_model: number;
    depth: number;
    wq: tf.layers.Layer;
    wk: tf.layers.Layer;
    wv: tf.layers.Layer;
    dense: tf.layers.Layer;

    

    scaled_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor | undefined) {
        return scaled_attention(q, k, v, mask)
    }

    split_heads(x: tf.Tensor, batch_size: number){
        x = tf.reshape(x, [batch_size, -1, this.num_heads, this.depth])
        return tf.transpose(x, [0, 2, 1, 3])
    }

    call([value, key, query, mask]: tf.Tensor[]): tf.Tensor[] {
        const batch_size = query.shape[0];
        let predQuery = this.wq.apply(query) as tf.Tensor;
        let predKey = this.wk.apply(key) as tf.Tensor;
        let predValues = this.wv.apply(value) as tf.Tensor;

        predQuery = this.split_heads(predQuery, batch_size)     // (batch_size, num_heads, seq_len_q, depth)
        predKey = this.split_heads(predKey, batch_size)         // (batch_size, num_heads, seq_len_k, depth)
        predValues = this.split_heads(predValues, batch_size)   // (batch_size, num_heads, seq_len_v, depth)

        // scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        // attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        const [scaled_attention, attention_weights] = this.scaled_attention(predQuery, predKey, predValues, mask)
        const transposed_scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) // (batch_size, seq_len_q, num_heads, depth)
        const concat_attention = tf.reshape(transposed_scaled_attention, [batch_size, -1, this.d_model])  // (batch_size, seq_len_q, d_model)

        const output = this.dense.apply(concat_attention) as tf.Tensor// (batch_size, seq_len_q, d_model)
        return [output, attention_weights]
    }
    
    constructor({d_model, num_heads}: {d_model: number, num_heads: number}) {
        super({
            trainable: true,
            name: 'MultiHeadAttention'
        })
        this.num_heads = num_heads;
        this.d_model = d_model;

        if(d_model % this.num_heads !== 0) {
            throw new Error("D_model must be divisible by num_heads");
        }

        this.depth = Math.floor(d_model / this.num_heads);
        this.wq = tf.layers.dense({units: d_model})
        this.wk = tf.layers.dense({units: d_model})
        this.wv = tf.layers.dense({units: d_model})
        this.dense = tf.layers.dense({units: d_model})
    }
}