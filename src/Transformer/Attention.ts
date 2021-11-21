import * as tf from "@tensorflow/tfjs"

export function scaled_attention(query: tf.Tensor | tf.SymbolicTensor, key: tf.Tensor | tf.SymbolicTensor, values: tf.Tensor | tf.SymbolicTensor, mask: tf.Tensor | tf.SymbolicTensor | undefined) {
    const matmul_qk = tf.layers.dot({ axes: -1}).apply(query, key) as tf.Tensor; // type hack
    
    if(matmul_qk instanceof tf.SymbolicTensor ) {
        let finalTensor;
        let finalTensorDot;
        if(Array.isArray(matmul_qk) && matmul_qk[0] instanceof tf.SymbolicTensor) {
            finalTensor = new tf.SymbolicTensor("float32", matmul_qk.shape, null, matmul_qk, {});
        } else {
            finalTensor = new tf.SymbolicTensor("float32", matmul_qk.shape, null, [matmul_qk], {});
        }
        finalTensorDot = tf.layers.dot({ axes: -1}).apply(finalTensor, values) as tf.Tensor;
        return [finalTensor, finalTensorDot];
    }
    const dk = tf.cast(key.shape[key.shape.length-1], "float32");
    // (Q * K) / sqrt(dk) is the scaled dot product of query and key
    const scaled_attention_logits = matmul_qk.div(tf.sqrt(dk));

    tf.math

    if(mask) {
        scaled_attention_logits.add((mask as tf.Tensor).mul(-1e9));
    }

    const attention_weights = tf.softmax(scaled_attention_logits, -1);

    const output = tf.matMul(attention_weights, values as tf.Tensor);
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

    scaled_attention(q: tf.Tensor | tf.SymbolicTensor, k: tf.Tensor | tf.SymbolicTensor, v: tf.Tensor | tf.SymbolicTensor, mask: tf.Tensor | tf.SymbolicTensor | undefined) {
        return scaled_attention(q, k, v, mask)
    }

    split_heads(x: tf.Tensor | tf.SymbolicTensor){
        const reshaped = tf.layers.reshape({targetShape: [ -1, this.num_heads, this.depth]}).apply(x)
        return tf.layers.permute({dims: [0, 2, 1, 3]}).apply(reshaped) as tf.Tensor | tf.SymbolicTensor
    }

    call(input: tf.Tensor[] | tf.Tensor | tf.SymbolicTensor[] | tf.SymbolicTensor): tf.Tensor[] {
        const [value, key, query, mask] = input as tf.Tensor[] | tf.SymbolicTensor[];
        const batch_size = query.shape[0];
        let predQuery = this.wq.apply(query) as tf.Tensor | tf.SymbolicTensor;
        let predKey = this.wk.apply(key) as tf.Tensor | tf.SymbolicTensor;
        let predValues = this.wv.apply(value) as tf.Tensor | tf.SymbolicTensor;

        predQuery = this.split_heads(predQuery)     // (batch_size, num_heads, seq_len_q, depth)
        predKey = this.split_heads(predKey)         // (batch_size, num_heads, seq_len_k, depth)
        predValues = this.split_heads(predValues)   // (batch_size, num_heads, seq_len_v, depth)

        // scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        // attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        const [scaled_attention, attention_weights] = this.scaled_attention(predQuery, predKey, predValues, mask)
        const transposed_scaled_attention = tf.layers.permute({dims:  [0, 2, 1, 3]}).apply((scaled_attention)) // (batch_size, seq_len_q, num_heads, depth)
        const concat_attention = tf.layers.reshape({targetShape: [batch_size, -1, this.d_model]}).apply(transposed_scaled_attention)  // (batch_size, seq_len_q, d_model)

        const output = this.dense.apply(concat_attention) as tf.Tensor// (batch_size, seq_len_q, d_model)
        return [output, attention_weights as tf.Tensor]
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