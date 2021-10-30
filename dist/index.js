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

/*! *****************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
/* global Reflect, Promise */

var extendStatics = function(d, b) {
    extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
    return extendStatics(d, b);
};

function __extends(d, b) {
    if (typeof b !== "function" && b !== null)
        throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
    extendStatics(d, b);
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}

var Time2Vec = /** @class */ (function (_super) {
    __extends(Time2Vec, _super);
    function Time2Vec(kernel_size, periodic_activation) {
        if (periodic_activation === void 0) { periodic_activation = 'sin'; }
        var _this = _super.call(this, {
            trainable: true,
            name: 'Time2VecLayer_' + periodic_activation.toUpperCase()
        }) || this;
        _this.k = kernel_size;
        _this.p_activation = periodic_activation;
        return _this;
    }
    Time2Vec.prototype.build = function (inputShape) {
        this.built = true;
        this.wb = this.addWeight("wb", [inputShape[0], 1], "float32", tf__namespace.initializers.glorotUniform({}), undefined, true);
        this.bb = this.addWeight("bb", [inputShape[0], 1], "float32", tf__namespace.initializers.glorotUniform({}), undefined, true);
        this.wa = this.addWeight("wa", [inputShape[1], this.k], "float32", tf__namespace.initializers.glorotUniform({}), undefined, true);
        this.ba = this.addWeight("ba", [inputShape[0], this.k], "float32", tf__namespace.initializers.glorotUniform({}), undefined, true);
        _super.prototype.build.call(this, inputShape);
    };
    Time2Vec.prototype.call = function (inputs) {
        if (Array.isArray(inputs)) {
            inputs = inputs[0];
        }
        var bb = this.bb.read().slice([0], [inputs.shape[0]]);
        var wb = this.wb.read().slice([0], [inputs.shape[0]]);
        var ba = this.ba.read().slice([0], [inputs.shape[0]]);
        var wa = this.wa.read().slice([0], [inputs.shape[1]]);
        var bias = bb.add(wb.mul(inputs));
        var posFunction;
        if (this.p_activation === 'sin') {
            posFunction = tf__namespace.sin;
        }
        else if (this.p_activation === 'cos') {
            posFunction = tf__namespace.cos;
        }
        else {
            throw new TypeError('Neither sine or cosine periodic activation be selected.');
        }
        var wgts = posFunction(ba.add(inputs.dot(wa)));
        var concatLayer = bias.concat(wgts, -1);
        this.output;
        return concatLayer;
    };
    Time2Vec.prototype.computeOutputShape = function (input_shape) {
        return [input_shape[0], input_shape[1] * 2];
    };
    return Time2Vec;
}(tf__namespace.layers.Layer));

var index$1 = /*#__PURE__*/Object.freeze({
    __proto__: null,
    Time2Vec: Time2Vec
});

function create_padding_mask(seq) {
    var newSeq = tf__namespace.cast(tf__namespace.notEqual(seq, 0), 'float32');
    //should be [batch_size, 1, 1, seq_len]
    var addedDims = newSeq.expandDims(1).expandDims(1);
    return addedDims;
}
function create_look_ahead_mask(seq_len) {
    var mask = tf__namespace.ones([1], 'float32');
    mask = mask.sub(tf__namespace.linalg.bandPart(tf__namespace.ones([seq_len, seq_len]), -1, 0));
    return mask;
}

var Transformer = /** @class */ (function (_super) {
    __extends(Transformer, _super);
    function Transformer() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    return Transformer;
}(tf__namespace.layers.Layer));

function scaled_attention(query, key, values, mask) {
    var matmul_qk = tf__namespace.matMul(query, key, false, true);
    var dk = tf__namespace.cast(key.shape[key.shape.length - 1], "float32");
    // (Q * K) / sqrt(dk) is the scaled dot product of query and key
    var scaled_attention_logits = matmul_qk.div(tf__namespace.sqrt(dk));
    if (mask) {
        scaled_attention_logits.add(mask.mul(-1e9));
    }
    var attention_weights = tf__namespace.softmax(scaled_attention_logits, -1);
    var output = tf__namespace.matMul(attention_weights, values);
    return [output, attention_weights];
}
var MultiHeadAttention = /** @class */ (function (_super) {
    __extends(MultiHeadAttention, _super);
    function MultiHeadAttention(_a) {
        var d_model = _a.d_model, num_heads = _a.num_heads;
        var _this = _super.call(this, {
            trainable: true,
            name: 'MultiHeadAttention'
        }) || this;
        _this.num_heads = num_heads;
        _this.d_model = d_model;
        if (d_model % _this.num_heads !== 0) {
            throw new Error("D_model must be divisible by num_heads");
        }
        _this.depth = Math.floor(d_model / _this.num_heads);
        _this.wq = tf__namespace.layers.dense({ units: d_model });
        _this.wk = tf__namespace.layers.dense({ units: d_model });
        _this.wv = tf__namespace.layers.dense({ units: d_model });
        _this.dense = tf__namespace.layers.dense({ units: d_model });
        return _this;
    }
    MultiHeadAttention.prototype.scaled_attention = function (q, k, v, mask) {
        return scaled_attention(q, k, v, mask);
    };
    MultiHeadAttention.prototype.split_heads = function (x, batch_size) {
        x = tf__namespace.reshape(x, [batch_size, -1, this.num_heads, this.depth]);
        return tf__namespace.transpose(x, [0, 2, 1, 3]);
    };
    MultiHeadAttention.prototype.call = function (_a) {
        var value = _a[0], key = _a[1], query = _a[2], mask = _a[3];
        var batch_size = query.shape[0];
        var predQuery = this.wq.apply(query);
        var predKey = this.wk.apply(key);
        var predValues = this.wv.apply(value);
        predQuery = this.split_heads(predQuery, batch_size); // (batch_size, num_heads, seq_len_q, depth)
        predKey = this.split_heads(predKey, batch_size); // (batch_size, num_heads, seq_len_k, depth)
        predValues = this.split_heads(predValues, batch_size); // (batch_size, num_heads, seq_len_v, depth)
        // scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        // attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        var _b = this.scaled_attention(predQuery, predKey, predValues, mask), scaled_attention = _b[0], attention_weights = _b[1];
        var transposed_scaled_attention = tf__namespace.transpose(scaled_attention, [0, 2, 1, 3]); // (batch_size, seq_len_q, num_heads, depth)
        var concat_attention = tf__namespace.reshape(transposed_scaled_attention, [batch_size, -1, this.d_model]); // (batch_size, seq_len_q, d_model)
        var output = this.dense.apply(concat_attention); // (batch_size, seq_len_q, d_model)
        return [output, attention_weights];
    };
    return MultiHeadAttention;
}(tf__namespace.layers.Layer));

var index = /*#__PURE__*/Object.freeze({
    __proto__: null,
    create_look_ahead_mask: create_look_ahead_mask,
    create_padding_mask: create_padding_mask,
    Transformer: Transformer,
    MultiHeadAttention: MultiHeadAttention,
    scaled_attention: scaled_attention
});

exports.Time2Vec = index$1;
exports.Transformer = index;
