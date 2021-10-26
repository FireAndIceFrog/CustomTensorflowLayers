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
            [1,1] as tf.Shape,
            "float32",
            tf.initializers.glorotUniform({}),
            undefined,
            true
        )
        
        this.bb = this.addWeight(
            "bb",
            [1,1] as tf.Shape,
            "float32",
            tf.initializers.glorotUniform({}),
            undefined,
            true
        )
        
        this.wa = this.addWeight(
            "wa",
            [1, this.k],
            "float32",
            tf.initializers.glorotUniform({}),
            undefined,
            true
        )
        
        this.ba = this.addWeight(
            "ba",
            [1, this.k],
            "float32",
            tf.initializers.glorotUniform({}),
            undefined,
            true
        )
        
        super.build(inputShape)
    }

    apply(inputs: tf.Tensor | tf.Tensor[] | tf.SymbolicTensor | tf.SymbolicTensor[]): tf.Tensor | tf.Tensor[] | tf.SymbolicTensor | tf.SymbolicTensor[] { 
        const bias: tf.Tensor = tf.add(this.bb.read(), tf.mul(this.wb.read(), inputs as tf.Tensor))

        let posFunction: typeof tf.sin | typeof tf.cos;
        if ( this.p_activation === 'sin' ) {
            posFunction = tf.sin
        } else if ( this.p_activation === 'cos' ){
            posFunction = tf.cos
        } else {
            throw new TypeError('Neither sine or cosine periodic activation be selected.')
        }

        const wgts: tf.Tensor = posFunction(tf.add(this.ba.read(), tf.dot(this.wa.read(), inputs as tf.Tensor)) as tf.Tensor)

        const concatLayer = tf.layers.concatenate({axis: -1});
        return concatLayer.apply([bias, wgts]) as tf.Tensor;
    }

    compute_output_shape(input_shape: tf.Shape){
        return (input_shape[0], input_shape[1], this.k + 1)
    }
}