import { MnistData } from './data.js';

async function showExamples(data) {
  const examplesContainer = document.getElementById('examples');
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    examplesContainer.appendChild(canvas);

    imageTensor.dispose();
  }
}

function getModel() {
  const model = tf.sequential();
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  model.add(tf.layers.flatten());

  const NUM_OUTPUT_CLASSES = 10;
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax',
    })
  );

  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function train(model, data) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = document.getElementById('training-graph-container'); // Graph container inside the training div
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks,
  });
}

const classNames = [
  'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'
];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}

async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = document.getElementById('accuracy-container');
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = document.getElementById('confusion-container');
  tfvis.render.confusionMatrix(container, { values: confusionMatrix, tickLabels: classNames });

  labels.dispose();
}

function initializeCanvas(model) {
  const canvas = document.getElementById('drawing-canvas');
  const ctx = canvas.getContext('2d');
  const predictionOutput = document.getElementById('prediction-output');
  const clearButton = document.getElementById('clearButton');
  const IMAGE_SIZE = 28;

  // Set background to black
  ctx.fillStyle = 'black'; // Fill the canvas with black color
  ctx.fillRect(0, 0, canvas.width, canvas.height); // Fill the entire canvas

  // Drawing logic
  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;

  canvas.addEventListener('mousedown', (event) => {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    lastX = event.clientX - rect.left;
    lastY = event.clientY - rect.top;
  });

  canvas.addEventListener('mouseup', () => (isDrawing = false));
  canvas.addEventListener('mousemove', draw);

  function draw(event) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.stroke();

    lastX = x;
    lastY = y;
  }

  // Clear canvas logic
  clearButton.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictionOutput.textContent = '';
  });

  // Predict logic
  canvas.addEventListener('mouseup', async () => {
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  
    const inputTensor = tf.tidy(() => {
      const tempCanvas = document.createElement('canvas');
      const tempCtx = tempCanvas.getContext('2d');
      tempCanvas.width = IMAGE_SIZE;
      tempCanvas.height = IMAGE_SIZE;
  
      // Resize the canvas content to 28x28
      tempCtx.drawImage(canvas, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  
      // Get the resized image data and normalize it
      const resizedImageData = tempCtx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
      const tensor = tf.browser.fromPixels(resizedImageData, 1)
        .toFloat()
        .div(255.0) // Normalize pixel values
        .reshape([1, IMAGE_SIZE, IMAGE_SIZE, 1]);
  
      console.log('Input Tensor:', tensor.arraySync()); // Debugging
      return tensor;
    });
  
    // Make prediction using the trained model
    const prediction = model.predict(inputTensor);
    prediction.print(); // Debugging predictions
  
    const predictedIndex = prediction.argMax(-1).dataSync()[0];
    predictionOutput.textContent = `Prediction: ${predictedIndex} (${classNames[predictedIndex]})`;
  
    inputTensor.dispose();
  });
}

// Upload and predict logic
function handleImageUpload(model) {
  const uploadInput = document.getElementById('uploadInput');
  const uploadPredictionOutput = document.getElementById('upload-prediction-output');

  uploadInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Create a file reader and load the image
    const reader = new FileReader();
    reader.onload = async () => {
      const imgElement = document.createElement('img');
      imgElement.src = reader.result;
      imgElement.onload = async () => {
        // Prepare the image
        const tensor = tf.browser.fromPixels(imgElement).toFloat().resizeNearestNeighbor([28, 28]).mean(2).expandDims(2).expandDims(0).div(255.0);
        console.log('Input Tensor:', tensor.arraySync());

        // Make a prediction
        const prediction = model.predict(tensor);
        const predictedIndex = prediction.argMax(-1).dataSync()[0];

        uploadPredictionOutput.textContent = `Prediction: ${predictedIndex} (${classNames[predictedIndex]})`;

        tensor.dispose();
      };
    };

    reader.readAsDataURL(file);
  });
}

async function run() {
  const data = new MnistData();
  await data.load();
  await showExamples(data);

  const model = getModel();
  tfvis.show.modelSummary(document.getElementById('model-summary-container'), model);

  document.getElementById('trainButton').addEventListener('click', async () => {
    await train(model, data);
    await showAccuracy(model, data);
    await showConfusion(model, data);
    initializeCanvas(model); // Initialize canvas after training
    handleImageUpload(model); // Initialize image upload handler
  });

  document.getElementById('loadButton').addEventListener('click', async () => {
    try {
      const loadedModel = await tf.loadLayersModel('/model/my-model.json');
      await showAccuracy(loadedModel, data);
      await showConfusion(loadedModel, data);
      initializeCanvas(loadedModel); // Initialize canvas after loading
      handleImageUpload(loadedModel); // Initialize image upload handler
    } catch (error) {
      console.error('Error loading model:', error);
    }
  });
}

document.addEventListener('DOMContentLoaded', run);
