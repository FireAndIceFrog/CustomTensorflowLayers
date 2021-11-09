'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

var tf = require('@tensorflow/tfjs');

function _interopNamespace(e) {
    if (e && e.__esModule) return e;
    var n = Object.create(null);
    if (e) {
        Object.keys(e).forEach(function (k) {
            if (k !== 'default') {
                var d = Object.getOwnPropertyDescriptor(e, k);
                Object.defineProperty(n, k, d.get ? d : {
                    enumerable: true,
                    get: function () { return e[k]; }
                });
            }
        });
    }
    n["default"] = e;
    return Object.freeze(n);
}

var tf__namespace = /*#__PURE__*/_interopNamespace(tf);

class Time2Vec extends tf__namespace.layers.Layer {
    constructor(kernel_size, periodic_activation = 'sin') {
        super({
            trainable: true,
            name: 'Time2VecLayer_' + periodic_activation.toUpperCase()
        });
        this.k = kernel_size;
        this.p_activation = periodic_activation;
    }
    build(inputShape) {
        this.built = true;
        this.wb = this.addWeight("wb", [inputShape[0], 1], "float32", tf__namespace.initializers.glorotUniform({}), undefined, true);
        this.bb = this.addWeight("bb", [inputShape[0], 1], "float32", tf__namespace.initializers.glorotUniform({}), undefined, true);
        this.wa = this.addWeight("wa", [inputShape[1], this.k], "float32", tf__namespace.initializers.glorotUniform({}), undefined, true);
        this.ba = this.addWeight("ba", [inputShape[0], this.k], "float32", tf__namespace.initializers.glorotUniform({}), undefined, true);
        super.build(inputShape);
    }
    call(inputs) {
        if (Array.isArray(inputs)) {
            inputs = inputs[0];
        }
        if (inputs instanceof tf__namespace.SymbolicTensor) {
            return inputs;
        }
        const bb = this.bb.read().slice([0], [inputs.shape[0]]);
        const wb = this.wb.read().slice([0], [inputs.shape[0]]);
        const ba = this.ba.read().slice([0], [inputs.shape[0]]);
        const wa = this.wa.read().slice([0], [inputs.shape[1]]);
        const bias = bb.add(wb.mul(inputs));
        let posFunction;
        if (this.p_activation === 'sin') {
            posFunction = tf__namespace.sin;
        }
        else if (this.p_activation === 'cos') {
            posFunction = tf__namespace.cos;
        }
        else {
            throw new TypeError('Neither sine or cosine periodic activation be selected.');
        }
        const wgts = posFunction(ba.add(inputs.dot(wa)));
        const concatLayer = bias.concat(wgts, -1);
        this.output;
        return concatLayer;
    }
    computeOutputShape(input_shape) {
        return [input_shape[0], input_shape[1] * 2];
    }
}

var index$1 = /*#__PURE__*/Object.freeze({
    __proto__: null,
    Time2Vec: Time2Vec
});

function create_padding_mask(seq) {
    const newSeq = tf__namespace.cast(tf__namespace.notEqual(seq, 0), 'float32');
    //should be [batch_size, 1, 1, seq_len]
    const addedDims = newSeq.expandDims(1).expandDims(1);
    return addedDims;
}
function create_look_ahead_mask(seq_len) {
    let mask = tf__namespace.ones([1], 'float32');
    mask = mask.sub(tf__namespace.linalg.bandPart(tf__namespace.ones([seq_len, seq_len]), -1, 0));
    return mask;
}

function scaled_attention(query, key, values, mask) {
    const matmul_qk = tf__namespace.matMul(query, key, false, true);
    const dk = tf__namespace.cast(key.shape[key.shape.length - 1], "float32");
    // (Q * K) / sqrt(dk) is the scaled dot product of query and key
    const scaled_attention_logits = matmul_qk.div(tf__namespace.sqrt(dk));
    if (mask) {
        scaled_attention_logits.add(mask.mul(-1e9));
    }
    const attention_weights = tf__namespace.softmax(scaled_attention_logits, -1);
    const output = tf__namespace.matMul(attention_weights, values);
    return [output, attention_weights];
}
class MultiHeadAttention extends tf__namespace.layers.Layer {
    constructor({ d_model, num_heads }) {
        super({
            trainable: true,
            name: 'MultiHeadAttention'
        });
        this.num_heads = num_heads;
        this.d_model = d_model;
        if (d_model % this.num_heads !== 0) {
            throw new Error("D_model must be divisible by num_heads");
        }
        this.depth = Math.floor(d_model / this.num_heads);
        this.wq = tf__namespace.layers.dense({ units: d_model });
        this.wk = tf__namespace.layers.dense({ units: d_model });
        this.wv = tf__namespace.layers.dense({ units: d_model });
        this.dense = tf__namespace.layers.dense({ units: d_model });
    }
    scaled_attention(q, k, v, mask) {
        return scaled_attention(q, k, v, mask);
    }
    split_heads(x, batch_size) {
        x = tf__namespace.reshape(x, [batch_size, -1, this.num_heads, this.depth]);
        return tf__namespace.transpose(x, [0, 2, 1, 3]);
    }
    call([value, key, query, mask]) {
        const batch_size = query.shape[0];
        let predQuery = this.wq.apply(query);
        let predKey = this.wk.apply(key);
        let predValues = this.wv.apply(value);
        predQuery = this.split_heads(predQuery, batch_size); // (batch_size, num_heads, seq_len_q, depth)
        predKey = this.split_heads(predKey, batch_size); // (batch_size, num_heads, seq_len_k, depth)
        predValues = this.split_heads(predValues, batch_size); // (batch_size, num_heads, seq_len_v, depth)
        // scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        // attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        const [scaled_attention, attention_weights] = this.scaled_attention(predQuery, predKey, predValues, mask);
        const transposed_scaled_attention = tf__namespace.transpose(scaled_attention, [0, 2, 1, 3]); // (batch_size, seq_len_q, num_heads, depth)
        const concat_attention = tf__namespace.reshape(transposed_scaled_attention, [batch_size, -1, this.d_model]); // (batch_size, seq_len_q, d_model)
        const output = this.dense.apply(concat_attention); // (batch_size, seq_len_q, d_model)
        return [output, attention_weights];
    }
}

function pointWiseFeedForwardNetwork(d_model, dff) {
    return tf__namespace.sequential({
        layers: [
            tf__namespace.layers.dense({
                units: dff,
                activation: 'relu',
                kernelInitializer: 'glorotNormal',
                name: 'first-dense',
                inputShape: [d_model]
            }),
            tf__namespace.layers.dense({
                units: d_model,
                kernelInitializer: 'glorotNormal',
                name: 'second-dense'
            })
        ]
    });
}
class Encoder extends tf__namespace.layers.Layer {
    constructor(d_model, num_heads, dff, rate = 0.1) {
        super({
            name: 'encoder'
        });
        this.multiheadAttention = new MultiHeadAttention({ d_model, num_heads });
        this.ffn = pointWiseFeedForwardNetwork(d_model, dff);
        this.layernorm1 = tf__namespace.layers.layerNormalization({
            epsilon: 1e-6,
            name: 'layernorm1'
        });
        this.layernorm2 = tf__namespace.layers.layerNormalization({
            epsilon: 1e-6,
            name: 'layernorm2'
        });
        this.dropout1 = tf__namespace.layers.dropout({
            rate: rate,
            name: 'dropout1'
        });
        this.dropout2 = tf__namespace.layers.dropout({
            rate: rate,
        });
    }
    call(inputs, { training }) {
        if (inputs instanceof tf__namespace.SymbolicTensor) {
            return inputs;
        }
        let [x, mask] = inputs;
        let [attentionOutput] = this.multiheadAttention.call([x, x, x, mask]); //(batch_size, input_seq_len, d_model)
        attentionOutput = this.dropout1.apply(attentionOutput, { training });
        const out1 = this.layernorm1.apply(x.add(attentionOutput));
        let ffnOutput = this.ffn.apply(out1); //(batch_size, input_seq_len, d_model)
        ffnOutput = this.dropout2.apply(ffnOutput, { training });
        const out2 = this.layernorm2.apply(out1.add(ffnOutput));
        return out2;
    }
}

class DecoderLayer extends tf__namespace.layers.Layer {
    constructor(d_model, num_heads, dff, rate = 0.1) {
        super({
            name: 'DecoderLayer',
        });
        this.mha1 = new MultiHeadAttention({ d_model, num_heads });
        this.mha2 = new MultiHeadAttention({ d_model, num_heads });
        this.ffn = pointWiseFeedForwardNetwork(d_model, dff);
        this.layernorm1 = tf__namespace.layers.layerNormalization({ epsilon: 1e-6 });
        this.layernorm2 = tf__namespace.layers.layerNormalization({ epsilon: 1e-6 });
        this.layernorm3 = tf__namespace.layers.layerNormalization({ epsilon: 1e-6 });
        this.dropout1 = tf__namespace.layers.dropout({ rate: rate });
        this.dropout2 = tf__namespace.layers.dropout({ rate: rate });
        this.dropout3 = tf__namespace.layers.dropout({ rate: rate });
    }
    call(inputs, kwargs) {
        if (inputs instanceof tf__namespace.SymbolicTensor) {
            return inputs;
        }
        const [x, encoder_outputs, look_ahead_mask, padding_mask] = inputs;
        let [attn1, attn_weights_block1] = this.mha1.call([x, x, x, look_ahead_mask]);
        attn1 = this.dropout1.apply(attn1, kwargs);
        const out1 = this.layernorm1.apply(attn1.add(x), kwargs);
        let [attn2, attn_weights_block2] = this.mha2.call([encoder_outputs, encoder_outputs, out1, padding_mask]);
        attn2 = this.dropout2.apply(attn2, kwargs);
        const out2 = this.layernorm2.apply(attn2.add(out1), kwargs);
        let ffn_output = this.ffn.apply(out2);
        ffn_output = this.dropout3.apply(ffn_output, kwargs);
        const out3 = this.layernorm3.apply(ffn_output.add(out2), kwargs);
        return [out3, attn_weights_block1, attn_weights_block2];
    }
}

var index = /*#__PURE__*/Object.freeze({
    __proto__: null,
    create_look_ahead_mask: create_look_ahead_mask,
    create_padding_mask: create_padding_mask,
    MultiHeadAttention: MultiHeadAttention,
    scaled_attention: scaled_attention,
    Encoder: Encoder,
    pointWiseFeedForwardNetwork: pointWiseFeedForwardNetwork,
    DecoderLayer: DecoderLayer
});

exports.Time2Vec = index$1;
exports.Transformer = index;
