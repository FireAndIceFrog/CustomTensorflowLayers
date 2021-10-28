import * as assert from "assert"
import * as tf from "@tensorflow/tfjs"
//@ts-ignore
import { Time2Vec } from "../dist/index.es.js";
const days = 365;
const kernelSize = 5;

function createTestData(): [tf.Tensor2D, tf.Tensor] {
    const data = []
    const classes = []

    for(let i = 0; i < days-kernelSize; ++i) {
        const kernelData = []
        for(let j = 0; j < kernelSize; ++j) {
            kernelData.push(j+i)
        }
        kernelData.reverse()
        data.push(kernelData)
        classes.push(i%7===0 ? [1] : [2])
    }

    const inputs = tf.tensor2d(data)
    const outputs = tf.oneHot(tf.tensor2d(classes, [days-kernelSize,1], "int32"), 2)

    return [inputs, outputs]
}

function createModel(useTime2vec: boolean) {
    const dataInput = tf.input({batchShape: [32, kernelSize]});
    let layers: tf.SymbolicTensor

    if(useTime2vec){
        layers = new Time2Vec.Time2Vec(kernelSize, "sin").apply(dataInput) as tf.SymbolicTensor
    }
    
    layers = tf.layers.dense({
        units: 50,
    }).apply(layers) as tf.SymbolicTensor

    layers = tf.layers.reshape({
        targetShape: [10, kernelSize]
    }).apply(layers) as tf.SymbolicTensor
    
    layers = tf.layers.lstm({
        units: 64,
        returnSequences: true
    }).apply(layers) as tf.SymbolicTensor

    layers = tf.layers.lstm({
        units: 64,
        returnSequences: true
    }).apply(layers) as tf.SymbolicTensor

    layers = tf.layers.dense({
        units: 50,
    }).apply(layers) as tf.SymbolicTensor

    layers = tf.layers.reshape({
        targetShape: [1,500]
    }).apply(layers) as tf.SymbolicTensor
    
    layers = tf.layers.dense({
        units: 2,
        activation: 'softmax'
    }).apply(layers) as tf.SymbolicTensor

    const model = tf.model({
        inputs: [dataInput],
        outputs: layers
    });

    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: 'adam'
    });
    return model
}

it('Can train', async () => {
    const [inputs, outputs] = createTestData();
    const model = createModel(true);

    const train = tf.slice(inputs, 0, Math.floor(days/2))
    const trainClasses = tf.slice(outputs, 0, Math.floor(days/2))

    const history = await model.fit(train, trainClasses, {epochs: 2})
    assert(history.history.loss.length===2);

});
