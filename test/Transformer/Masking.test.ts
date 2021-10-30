//@ts-ignore
import { create_look_ahead_mask, create_padding_mask } from "../dist/index.es.js";
import * as tf from "@tensorflow/tfjs"

it("Can create a padding mask", ()=> {
    const x = tf.tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    const result = create_padding_mask(x)
    const shape = result.shape
    const printedValues = result.dataSync()
})