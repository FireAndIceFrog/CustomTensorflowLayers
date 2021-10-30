var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
import * as tf from "@tensorflow/tfjs";
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
        this.wb = this.addWeight("wb", [1, 1], "float32", tf.initializers.glorotUniform({}), undefined, true);
        this.bb = this.addWeight("bb", [1, 1], "float32", tf.initializers.glorotUniform({}), undefined, true);
        this.wa = this.addWeight("wa", [1, this.k], "float32", tf.initializers.glorotUniform({}), undefined, true);
        this.ba = this.addWeight("ba", [1, this.k], "float32", tf.initializers.glorotUniform({}), undefined, true);
        _super.prototype.build.call(this, inputShape);
    };
    Time2Vec.prototype.apply = function (inputs) {
        var bias = tf.add(this.bb.read(), tf.mul(this.wb.read(), inputs));
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
        var wgts = posFunction(tf.add(this.ba.read(), tf.dot(this.wa.read(), inputs)));
        var concatLayer = tf.layers.concatenate({ axis: -1 });
        return concatLayer.apply([bias, wgts]);
    };
    Time2Vec.prototype.compute_output_shape = function (input_shape) {
        return (input_shape[0], input_shape[1], this.k + 1);
    };
    return Time2Vec;
}(tf.layers.Layer));
export { Time2Vec };
