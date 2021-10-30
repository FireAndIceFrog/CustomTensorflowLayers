/// <reference path ="../../dist/index.d.ts"> 
// @ts-ignore
import { Transformer } from "../../dist/index.es.js";
import * as tf from "@tensorflow/tfjs";
import * as assert from "assert"

it("Can run attention", ()=>{
    const temp_k = tf.tensor([[10, 0, 0],
        [0, 10, 0],
        [0, 0, 10],
        [0, 0, 10]])

    const temp_v = tf.tensor([[1, 0],
        [10, 0],
        [100, 5],
        [1000, 6]])

    // Query signifies the tensor we want to use
    // Softmax should return a mask which represents the probability of the query
    const temp_q = tf.tensor([[0,0,10]])

    const [temp_out, temp_attn] = Transformer.scaled_attention(temp_q, temp_k, temp_v)

    const temp_out_data = temp_out.dataSync()
    const temp_attn_data = temp_attn.dataSync().map((x: number)=>x < 0.001 ? 0 : x)

    const temp_out_test_array = Array.prototype.map.bind(temp_out_data)((x: number)=>x)
    const temp_attn_test_array = Array.prototype.map.bind(temp_attn_data)((x: number)=>x)
    
    assert.deepEqual(temp_out_test_array, [550, 5.5])
    assert.deepEqual(temp_attn_test_array, [0, 0, 0.5, 0.5])
})

it("Does have the correct shapes for MultiHead Attn.", ()=>{
    const temp_mha = new Transformer.MultiHeadAttention({d_model:512, num_heads:8})
    const y = tf.randomUniform([1,60,512]) // Batch, Seq, D_model
    const [temp_out, temp_attn] = temp_mha.call([y, y, y])
    const tem_out_shape = Array.prototype.map.bind(temp_out.shape)((x: number)=>x)
    const temp_attn_shape = Array.prototype.map.bind(temp_attn.shape)((x: number)=>x)

    assert.deepEqual(tem_out_shape, [1, 60, 512])
    assert.deepEqual(temp_attn_shape, [1, 8, 60, 60])
})