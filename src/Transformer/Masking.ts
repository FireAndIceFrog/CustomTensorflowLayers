import * as tf from "@tensorflow/tfjs"

export function create_padding_mask(seq: tf.Tensor): tf.Tensor {
    const newSeq =  tf.cast(tf.notEqual(seq, 0), 'float32')
    //should be [batch_size, 1, 1, seq_len]
    const addedDims = newSeq.expandDims(1).expandDims(1)
    return addedDims
}

export function create_look_ahead_mask(seq_len: number): tf.Tensor {
    let mask = tf.ones([1], 'float32')
    
    mask = mask.sub(tf.linalg.bandPart(
            tf.ones([seq_len, seq_len]) as tf.Tensor,
            -1, 
            0
        ))

    return mask
}

