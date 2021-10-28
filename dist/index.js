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
    function Time2Vec(output_dim, periodic_activation) {
        if (periodic_activation === void 0) { periodic_activation = 'sin'; }
        var _this = _super.call(this, {
            trainable: true,
            name: 'Time2VecLayer_' + periodic_activation.toUpperCase()
        }) || this;
        _this.output_dim = output_dim;
        _this.p_activation = periodic_activation;
        return _this;
    }
    Time2Vec.prototype.build = function (input_shape) {
        this.built = true;
        this.LatentVectorA = this.addWeight("wb", [input_shape[input_shape.length - 1], this.output_dim], "float32", tf__namespace.initializers.glorotUniform({}), undefined, true);
        this.LatentVectorB = this.addWeight("bb", [input_shape[1], this.output_dim], "float32", tf__namespace.initializers.glorotUniform({}), undefined, true);
        this.biasWeightA = this.addWeight("wa", [input_shape[1], 1], "float32", tf__namespace.initializers.glorotUniform({}), undefined, true);
        this.biasWeightB = this.addWeight("ba", [input_shape[1], 1], "float32", tf__namespace.initializers.glorotUniform({}), undefined, true);
        _super.prototype.build.call(this, input_shape);
    };
    Time2Vec.prototype.call = function (inputs) {
        if (Array.isArray(inputs)) {
            inputs = inputs[0];
        }
        var bias = this.biasWeightB.read().add(this.biasWeightA.read().mul(inputs));
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
        var wgts = posFunction(this.LatentVectorB.read().add(inputs.dot(this.LatentVectorA.read())));
        var concatLayer = bias.concat(wgts, -1);
        return concatLayer;
    };
    return Time2Vec;
}(tf__namespace.layers.Layer));

var index = /*#__PURE__*/Object.freeze({
    __proto__: null,
    Time2Vec: Time2Vec
});

exports.Time2Vec = index;
