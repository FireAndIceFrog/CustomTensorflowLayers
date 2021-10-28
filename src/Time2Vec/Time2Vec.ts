import * as tf from "@tensorflow/tfjs"

type activationType = 'sin' | "cos"
type Kwargs = {[key: string]: any}

export class Time2Vec extends tf.layers.Layer {
    k: number;
    p_activation: activationType;
    wb: tf.LayerVariable;
    bb: tf.LayerVariable;
    wa: tf.LayerVariable;
    ba: tf.LayerVariable;

    constructor(kernel_size: number, periodic_activation: activationType ='sin') {
        super({
            trainable: true,
            name: 'Time2VecLayer_'+periodic_activation.toUpperCase()
        })
        
        this.k = kernel_size
        this.p_activation = periodic_activation
    }

    build(inputShape: tf.Shape) {
        this.built = true;
        this.wb = this.addWeight(
            "wb",
            [inputShape[0],1] as tf.Shape,
            "float32",
            tf.initializers.glorotUniform({}),
            undefined,
            true
        )
        
        this.bb = this.addWeight(
            "bb",
            [inputShape[0],1] as tf.Shape,
            "float32",
            tf.initializers.glorotUniform({}),
            undefined,
            true
        )
        
        this.wa = this.addWeight(
            "wa",
            [inputShape[1], this.k],
            "float32",
            tf.initializers.glorotUniform({}),
            undefined,
            true
        )
        
        this.ba = this.addWeight(
            "ba",
            [inputShape[0], this.k],
            "float32",
            tf.initializers.glorotUniform({}),
            undefined,
            true
        )
        
        super.build(inputShape)
    }

    call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor | tf.Tensor[]{ 
        if(Array.isArray(inputs)){
            inputs = inputs[0]
        }
        const bias: tf.Tensor = this.bb.read().add(this.wb.read().mul(inputs as tf.Tensor))

        let posFunction: typeof tf.sin | typeof tf.cos;
        if ( this.p_activation === 'sin' ) {
            posFunction = tf.sin
        } else if ( this.p_activation === 'cos' ){
            posFunction = tf.cos
        } else {
            throw new TypeError('Neither sine or cosine periodic activation be selected.')
        }

        const wgts: tf.Tensor = posFunction(this.ba.read().add(inputs.dot(this.wa.read())) as tf.Tensor)

        const concatLayer = bias.concat(wgts, -1)
        return concatLayer
    }

    compute_output_shape(input_shape: tf.Shape){
        return (input_shape[0], input_shape[1]*2)
    }
}