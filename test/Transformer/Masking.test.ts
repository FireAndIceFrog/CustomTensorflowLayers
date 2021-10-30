//@ts-ignore
import { Transformer } from "../../dist/index.es.js";
import * as tf from "@tensorflow/tfjs";
import * as assert from "assert"

it("Can create a padding mask", ()=> {
    const inputs = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
    const x = tf.tensor(inputs)
    const result = Transformer.create_padding_mask(x)
    const shape = result.shape
    const printedValues = result.dataSync()
    const maskShouldMatchResults =  inputs.flat().flatMap(x => x !== 0 ? 1 : 0).every((x, i) => x === printedValues[i])


    assert.equal(shape[0], 3)
    assert.equal(shape[1], 1)
    assert.equal(shape[2], 1)
    assert.equal(shape[3], 5)
    assert.equal(shape.length, 4)
    assert.equal(maskShouldMatchResults, true)
})

it("Can create a lookahead mask", ()=>{
    const x = tf.randomUniform([3, 5])
    const result = Transformer.create_look_ahead_mask(x.shape[1])
    const shape = result.shape
})