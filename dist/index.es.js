import * as tf from '@tensorflow/tfjs';

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
        this.wb = this.addWeight("wb", [inputShape[0], 1], "float32", tf.initializers.glorotUniform({}), undefined, true);
        this.bb = this.addWeight("bb", [inputShape[0], 1], "float32", tf.initializers.glorotUniform({}), undefined, true);
        this.wa = this.addWeight("wa", [inputShape[1], this.k], "float32", tf.initializers.glorotUniform({}), undefined, true);
        this.ba = this.addWeight("ba", [inputShape[0], this.k], "float32", tf.initializers.glorotUniform({}), undefined, true);
        _super.prototype.build.call(this, inputShape);
    };
    Time2Vec.prototype.call = function (inputs) {
        if (Array.isArray(inputs)) {
            inputs = inputs[0];
        }
        inputs.shape[0];
        this.outputShape[0];
        var bb = this.bb.read().slice([0], [inputs.shape[0]]);
        var wb = this.wb.read().slice([0], [inputs.shape[0]]);
        var ba = this.ba.read().slice([0], [inputs.shape[0]]);
        var wa = this.wa.read().slice([0], [inputs.shape[1]]);
        var bias = bb.add(wb.mul(inputs));
        var posFunction;
        if (this.p_activation === 'sin') {
            posFunction = tf.sin;
        }
        else if (this.p_activation === 'cos') {
            posFunction = tf.cos;
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
}(tf.layers.Layer));

var index = /*#__PURE__*/Object.freeze({
    __proto__: null,
    Time2Vec: Time2Vec
});

export { index as Time2Vec };
