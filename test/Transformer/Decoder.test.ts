/// <reference path ="../../dist/index.d.ts"> 
//@ts-ignore
import { Transformer } from "../../dist/index.es.js";
import * as tf from "@tensorflow/tfjs";
import * as assert from "assert"

it("Decoder has the correct shape", ()=> {
    const sample_encoder_layer_output = tf.randomNormal([64, 43, 512])

    const sample_ffn = new Transformer.DecoderLayer(512, 8, 2048);
    const [sample_decoder_layer_output] = sample_ffn.call(
        [tf.randomUniform([64, 50, 512]), sample_encoder_layer_output,false, undefined, undefined]
    )

    assert.deepEqual(sample_decoder_layer_output.shape, [64, 50, 512])
})
