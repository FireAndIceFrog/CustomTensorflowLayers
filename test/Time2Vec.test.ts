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
        layers = tf.layers.dense({
            units: 50,
        }).apply(layers) as tf.SymbolicTensor
    } else {
        layers = tf.layers.dense({
            units: 50,
        }).apply(dataInput) as tf.SymbolicTensor
    }
    
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

    // layers = tf.layers.dense({
    //     units: 50,
    // }).apply(layers) as tf.SymbolicTensor

    layers = tf.layers.reshape({
        targetShape: [1,640]
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

const [inputs, outputs] = createTestData();
const trainTestSplit = 0.75

const train = tf.slice(inputs, 0, Math.floor(days*trainTestSplit))
const trainClasses = tf.slice(outputs, 0, Math.floor(days*trainTestSplit))

const test = tf.slice(inputs, [Math.floor(days*trainTestSplit)],[-1])
const testClasses = tf.slice(outputs, [Math.floor(days*trainTestSplit)],[-1])

const Time2VecModel = createModel(true);
const time2VecPromise = Time2VecModel.fit(train, trainClasses, {epochs: 40, batchSize: 32})

const normalModel = createModel(false);
const normalModelPromise = normalModel.fit(train, trainClasses, {epochs: 40, batchSize: 32})

it('Can train', async () => {
    const history = await time2VecPromise
    assert.equal(history.history.loss.length, 2);
});

it('Compares better than normal Model on test data', async()=>{
    const history = await time2VecPromise
    const normalHistory = await normalModelPromise

    const time2VecTest = Time2VecModel.predict(test) as tf.Tensor
    const normalTest = normalModel.predict(test) as tf.Tensor
    
    const time2VecLoss = tf.losses.softmaxCrossEntropy(testClasses, time2VecTest)
    const normalLoss = tf.losses.softmaxCrossEntropy(testClasses, normalTest)

    const normalMean = normalLoss.mean().arraySync()

    const time2vecMean = time2VecLoss.mean().arraySync()

    assert.ok(normalMean > time2vecMean)
})