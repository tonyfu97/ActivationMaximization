////////////////////////////// SPECIFY UNIT ///////////////////////////////////

// Load rf data.
let rf_data = {
    "alexnet": {
        "layer_indices": [0, 3, 6, 8, 10],
        "rf_sizes": [11, 51, 99, 131, 163],
        "xn": [15, 63, 127, 159, 191],
        "nums_units": [64, 192, 384, 256, 256]
    },
    "vgg16": {
        "layer_indices": [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28],
        "rf_sizes": [3, 5, 10, 14, 24, 32, 40, 60, 76, 92, 132, 164, 196],
        "xn": [5, 7, 14, 18, 28, 36, 52, 72, 88, 104, 176, 208, 240],
        "nums_units": [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    },
    "resnet18": {
        "layer_indices": [0, 4, 7, 10, 13, 16, 19, 21, 24, 27, 30, 33, 35, 38, 41, 44, 47, 49, 52, 55],
        "rf_sizes": [7, 19, 27, 35, 43, 51, 67, 43, 83, 99, 115, 147, 99, 179, 211, 243, 307, 211, 371, 435],
        "xn": [9, 25, 33, 41, 49, 65, 81, 49, 97, 113, 129, 193, 129, 225, 257, 321, 385, 257, 449, 513],
        "nums_units": [64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512]
    }
}

var model_name_menu = document.getElementById("model");
var layer_menu = document.getElementById("layer");
var unit_id_input = document.getElementById("unit-index");

// Get model, layer, and unit_id (the default values) from index.html.
let model_name = model_name_menu.value;
let layer = layer_menu.value;
let conv_i = parseInt(layer.match(/\d+/)[0]) - 1;
let unit_id = parseInt(unit_id_input.value);
let num_layers = rf_data[model_name].layer_indices.length;
let num_units = rf_data[model_name].nums_units[conv_i];
unit_id_input.max = num_units - 1;

// populate the layer dropdown menu according to the model.
const populateLayerMenu = () => {
    // Clear all options first.
    layer_menu.innerHTML = '';

    // Populate with new layer options.
    for(let i = 1; i < num_layers+1; i++) {
        // Deeper layers are too big. They are too slow to render and should
        // not be included in the dropdown menu.
        let thisLayerName = `conv${i}`;
        var newOption = document.createElement("option");
        newOption.textContent = thisLayerName;
        newOption.value = thisLayerName;
        layer_menu.appendChild(newOption);
    }
}
populateLayerMenu();

// model dropdown menu logic:
model_name_menu.addEventListener('change', async (event) => {
    model_name = model_name_menu.value;
    num_layers = rf_data[model_name].layer_indices.length;
    populateLayerMenu();
    layer = layer_menu.value;
    conv_i = parseInt(layer.match(/\d+/)[0]) - 1
    num_units = rf_data[model_name].nums_units[conv_i];
    unit_id_input.max = num_units - 1;

    sess = new onnx.InferenceSession();
    loadingModelPromise = await sess.loadModel(`./onnx_files/${model_name}_${layer}.onnx`);

    load_img();
    updateCanvasSize(model_name, conv_i);
});

// layer dropdown menu logic:
layer_menu.addEventListener('change', async (event) => {
    layer = layer_menu.value;
    conv_i = parseInt(layer.match(/\d+/)[0]) - 1
    num_units = rf_data[model_name].nums_units[conv_i];
    unit_id_input.max = num_units - 1;

    sess = new onnx.InferenceSession();
    loadingModelPromise = await sess.loadModel(`./onnx_files/${model_name}_${layer}.onnx`);

    load_img();
    updateCanvasSize(model_name, conv_i);
});

// unit input form logic:
unit_id_input.addEventListener('change', (event) => {
    unit_id = parseInt(unit_id_input.value);
    load_img();
    updateCanvasSize(model_name, conv_i);
});

/////////////////////////// LOAD IMAGE FROM AWS S3 ////////////////////////////

/* Configuring AWS S3 access
Reference Website: Viewing Photos in an Amazon S3 Bucket from a Browser
https://docs.aws.amazon.com/sdk-for-javascript/v2/developer-guide/s3-example-photos-view.html
*/
var albumBucketName = 'rfmapping';
AWS.config.region = 'us-west-2'; // Region
AWS.config.credentials = new AWS.CognitoIdentityCredentials({
    IdentityPoolId: 'us-west-2:495c99f0-9773-4aef-9e01-29b3c4127e50',
});

// create a new service object
var s3 = new AWS.S3({
    apiVersion: '2006-03-01',
    params: {Bucket: albumBucketName}
});

// utility function to create HTML.
function getHtml(template) {
    return template.join('\n');
};

// show the photos that exist in an album.
function load_img() {
    var img_url_top_patch = `https://s3.us-west-2.amazonaws.com/rfmapping/SGD/top_patch_initialized/${model_name}/${layer}/${unit_id}.png`;
    document.getElementById('gradientAscentImage').src = img_url_top_patch;
};
load_img();

/////////////////////////////// MANAGE CANVAS /////////////////////////////////

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const colorPicker = document.getElementById('color');
const brushSizeInput = document.getElementById('brush-size');
const unmuteButton = document.getElementById('unmute');

let isDrawing = false;
let brushColor = colorPicker.value;
let brushSize = brushSizeInput.value;

// Canvas sizing
let canvas_size = rf_data[model_name].xn[conv_i];

// Canvas dimension logic: (scaled to the neuron's RF size)
const updateCanvasSize = (model_name, conv_i) => {
    canvas_size = rf_data[model_name].xn[conv_i];

    canvas.height = canvas_size;
    canvas.width = canvas_size;
}
updateCanvasSize(model_name, conv_i);

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);

colorPicker.addEventListener('change', (e) => {
    brushColor = e.target.value;
});

brushSizeInput.addEventListener('change', (e) => {
    brushSize = e.target.value;
    document.getElementById("brush-size-display").innerHTML = brushSize;
});

unmuteButton.addEventListener('click', () => {
    unmuteButton.classList.toggle('active');
    updateUnmuteButton();
    });
    
    function updateUnmuteButton() {
    if (unmuteButton.classList.contains('active')) {
        unmuteButton.style.backgroundColor = 'green';
        unmuteButton.textContent = 'Mute';
    } else {
        unmuteButton.style.backgroundColor = '';
        unmuteButton.textContent = 'Unmute';
    }
    spike.muted = !spike.muted;
}

const resetButton = document.getElementById('reset');
resetButton.addEventListener('click', resetCanvas);

function startDrawing(e) {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}

function draw(e) {
    if (!isDrawing) return;
    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.strokeStyle = brushColor;
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.stroke();
}

function resetCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    updatePredictions()
}

function stopDrawing() {
    isDrawing = false;
    ctx.closePath();
    updatePredictions()
}

///////////////////////////// ONNX MODEL //////////////////////////////////////

// Load model.
let sess = new onnx.InferenceSession();
let loadingModelPromise = sess.loadModel(`./onnx_files/${model_name}_${layer}.onnx`);
let response = 0;


// Get prediction. This function is called whenever the mouse is moved.
async function updatePredictions() {
    const imgData = ctx.getImageData(0, 0, canvas_size, canvas_size).data;
    // imgData is 1D array with length 1 * 4 * xn * xn.

    // Reshape 1D array into [1, 3, xn, xn] (but still flattens it).
    let rgbArray = new Float32Array(3 * canvas_size * canvas_size);
    let idx = 0
    for (var rgb_i = 0; rgb_i < 3; rgb_i++) { // RGB only (ignoring the 4th channel)
        for (var i = 0; i < canvas_size; i++) { // Height
            for (var j = 0; j < canvas_size; j++) { // Width
                let offset = (i * canvas_size * 4) + (j * 4) + rgb_i
                // Change color range from [0, 255] to [-1, 1).
                rgbArray[idx++] = (imgData[offset] - 128) / 128;
            }
        }
    }
    const input = new onnx.Tensor(rgbArray, "float32", [1, 3, canvas_size, canvas_size]);
    await loadingModelPromise;
    const outputMap = await sess.run([input]);
    const outputTensor = outputMap.values().next().value;
    const responses = outputTensor.data;

    let element = document.getElementById('response');
    let output_size = Math.sqrt(responses.length / num_units)
    let unit_id_flatten = ((output_size ** 2) * unit_id) - 1 + (output_size * Math.floor(output_size/2)) + Math.ceil(output_size/2);
    response = responses[unit_id_flatten];
    element.innerHTML = `Response = ${Math.round(response * 100) / 100}`;
};

// Continuously playing spike sound even when mouse is not moving:
var spike = document.getElementById("audio");
let MAX_VOLUME = 5;
canvas.addEventListener('mouseover',
    event => {
    setInterval(() => {
        if (!spike.muted && response > 0) {
        spike.volume = Math.min(response/MAX_VOLUME, 1);
        spike.play();
        }
    }, 100);
});


