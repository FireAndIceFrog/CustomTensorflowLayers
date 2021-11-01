/// <reference path ="../../dist/index.d.ts"> 
//@ts-ignore
import { Transformer } from "../../dist/index.es.js";
import * as tf from "@tensorflow/tfjs";
import * as assert from "assert"

it("Feed Forward has correct shape", ()=> {
    const sample_ffn = Transformer.pointWiseFeedForwardNetwork(512, 2048);
    const result = sample_ffn.call(tf.randomUniform([64, 50, 512]), {}) as tf.Tensor[]
    assert.deepEqual(result[0].shape, [64, 50, 512])
})

it("Encoder has the correct shape", ()=> {
    const sample_ffn = new Transformer.Encoder(512, 8, 2048, 0.2);
    const inputs = [tf.randomUniform([64, 43, 512]), undefined] as tf.Tensor[]
    const result = sample_ffn.call(inputs, {}) as tf.Tensor;

    assert.deepEqual(result.shape, [64, 43, 512])
})
