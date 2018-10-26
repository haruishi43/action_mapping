/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  COCO Objects
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Variables:

var coco_label_names = ['background',
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'mirror', 'dining table', 'window', 'desk','toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

var coco_label_colors = {
    1: [1, 0.7, 0],
    2: [0, 0, 0],
    3: [0, 0, 0],
    4: [0, 0, 0],
    5: [0, 0, 0],
    6: [0, 0, 0],
    7: [0, 0, 0],
    8: [0, 0, 0],
    9: [0, 0, 0],
    10: [0, 0, 0],
    11: [0, 0, 0],
    13: [0, 0, 0],
    14: [0, 0, 0],
    15: [0, 0, 0],
    16: [0, 0, 0],
    17: [0, 0, 0],
    18: [0, 0, 0],
    19: [0, 0, 0],
    20: [0, 0, 0],
    21: [0, 0, 0],
    22: [0, 0, 0],
    23: [0, 0, 0],
    24: [0, 0, 0],
    25: [0, 0, 0],
    27: [0.5, 0, 0.5],
    28: [0.3, 0.6, 1],
    31: [0.8, 0, 0.1],
    32: [0, 0.9, 1],
    33: [0.2, 0.2, 1],
    34: [0, 0, 0],
    35: [0, 0, 0],
    36: [0.1, 0.4, 0],
    37: [0, 0, 0],
    38: [0, 0, 0],
    39: [0, 0, 0],
    40: [0, 0, 0],
    41: [0, 0, 0],
    42: [0, 0, 0],
    43: [0, 0, 0],
    44: [0.9, 0.7, 1],
    46: [0, 0.2, 0.6],
    47: [1, 0.4, 0.5],
    48: [0, 0.1, 0.5],
    49: [0.2, 1, 0.2],
    50: [0.4, 0.7, 0.7],
    51: [0, 0, 0.3],
    52: [0, 0.5, 0.1],
    53: [0.1, 0.7, 0.3],
    54: [0.6, 0.5, 0.4],
    53: [0.3, 0.2, 0.1],
    54: [0.1, 0.2, 0.3],
    55: [0.4, 0.5, 0.6],
    56: [0.9, 0.8, 0.7],
    57: [0.6, 0.7, 1],
    58: [0.1, 0.1, 0.3],
    59: [0, 1, 0.5],
    60: [0.5, 0.3, 0.8],
    61: [0.6, 0.3, 0.1],
    62: [0.1, 0.6, 0.8],
    63: [1, 0.2, 0.6],
    64: [1, 0, 0.6],
    65: [0.9, 0.1, 0.9],
    67: [0.8, 0.3, 0.8],
    70: [0.4, 0.3, 0.9],
    72: [0, 0.3, 0.3],
    73: [0, 0.7, 1],
    74: [0, 0.5, 0.5],
    75: [1, 0.3, 0.2],
    76: [0.4, 0.4, 1],
    77: [0.1, 0.4, 0.3],
    78: [0, 0, 0.5],
    79: [0.7, 0.7, 0.3],
    80: [0.4, 0.3, 0],
    81: [0.8, 0.5, 0.3],
    82: [0.6, 0.9, 0.3],
    83: [0.5, 0.6, 0.2],
    84: [0.3, 0.6, 0.6],
    85: [0.9, 0.5, 0.1],
    86: [0.3, 0.5, 0.5],
    87: [1, 1, 0.3],
    88: [0.8, 1, 1],
    89: [0.3, 0.5, 0.3],
    90: [0.7, 0.8, 0.2]
};

/// Functions:

function updateCenters(centers) {
    if (typeof centers === "undefined") {
        console.log("something is wrong");
    } else {
        var jsonCenters = JSON.parse(centers);

        for (var key in jsonCenters) {
            var center = jsonCenters[key];
            
            if (!(key in centerGroup)) {
                // add text
                createText(key, center);
            }

        }

    }
}

function updateBboxes(bboxes) {
    if (typeof bboxes === "undefined") {
        console.log("something is wrong");
    } else {
        var jsonBboxes = JSON.parse(bboxes);
        console.log(jsonBboxes);

        for (var key in jsonBboxes) {

            var i = Number(key);
            var bbox = jsonBboxes[key];
            
        }

    }
}

function createText(text, pos) {

    var id_instance = cocoLabelTextToObjectID(text);
    var id = id_instance[0];
    var instance = id_instance[1];
    var name = coco_label_names[id];
    instance = instance.toString();
    name = name + " " + instance;
    var color = coco_label_colors[id.toString()];
    color = color.map(function(i) { return i*255 });

    var canvas1 = document.createElement('canvas');
    canvas1.height = 128;
    canvas1.width = 256;
    var context1 = canvas1.getContext('2d');
    context1.font = "Bold 40px Arial";
    context1.fillStyle = "rgba("+color[0]+","+color[1]+","+color[2]+",0.9)";
    context1.fillText(name, 0, 64);
    
    // canvas contents will be used for a texture
    var texture1 = new THREE.Texture(canvas1) 
    texture1.needsUpdate = true;
    
    var material1 = new THREE.MeshBasicMaterial( {map: texture1, side:THREE.DoubleSide } );
    material1.transparent = true;
    var mesh1 = new THREE.Mesh(
        new THREE.PlaneGeometry(8, 8),
        material1
    );
    mesh1.position.set(pos[0], pos[1], pos[2]);
    mesh1.rotation.x = Math.PI / 2; 
    renderGroup.add( mesh1 );

    centerGroup[text] = mesh1;
}

