import { LayerVariable, Shape, initializers, SymbolicTensor } from "@tensorflow/tfjs-layers";
import { Tensor, sin, cos, dot, add as scalarAdd, mul} from "@tensorflow/tfjs-core"
import { concatenate, Layer } from "@tensorflow/tfjs-layers/dist/exports_layers"

type activationType = 'sin' | "cos"
type Kwargs = {[key: string]: any}

export class Time2Vec extends Layer {
    k: number;
    p_activation: activationType;
    wb: LayerVariable
    bb: LayerVariable
    wa: LayerVariable
    ba: LayerVariable

    constructor(kernel_size: number, periodic_activation: activationType ='sin') {
        super({
            trainable: true,
            name: 'Time2VecLayer_'+periodic_activation.toUpperCase()
        })
        
        this.k = kernel_size
        this.p_activation = periodic_activation
    }

    build(inputShape: Shape) {
        this.built = true;
        this.wb = this.addWeight(
            "wb",
            [1,1] as Shape,
            "float32",
            initializers.glorotUniform({}),
            undefined,
            true
        )
        
        this.bb = this.addWeight(
            "bb",
            [1,1] as Shape,
            "float32",
            initializers.glorotUniform({}),
            undefined,
            true
        )
        
        this.wa = this.addWeight(
            "wa",
            [1, this.k],
            "float32",
            initializers.glorotUniform({}),
            undefined,
            true
        )
        
        this.ba = this.addWeight(
            "ba",
            [1, this.k],
            "float32",
            initializers.glorotUniform({}),
            undefined,
            true
        )
        
        super.build(inputShape)
    }

    apply(inputs: Tensor | Tensor[] , kwargs: Kwargs): Tensor | Tensor[] { 
        const bias: Tensor = scalarAdd(this.bb.read(), mul(this.wb.read(), inputs as Tensor))

        let posFunction: typeof sin | typeof cos;
        if ( this.p_activation === 'sin' ) {
            posFunction = sin
        } else if ( this.p_activation === 'cos' ){
            posFunction = cos
        } else {
            throw new TypeError('Neither sine or cosine periodic activation be selected.')
        }

        const wgts: Tensor = posFunction(scalarAdd(this.ba.read(), dot(this.wa.read(), inputs as Tensor)) as Tensor)

        const concatLayer = concatenate({axis: -1});
        return concatLayer.apply([bias, wgts]) as Tensor;
    }

    compute_output_shape(input_shape: Shape){
        return (input_shape[0], input_shape[1], this.k + 1)
    }
}