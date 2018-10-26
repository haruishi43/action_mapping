/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Utils
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth, window.innerHeight );
}

// Convert a single color to hex
var rgbToHex = function ( rgb ) {
    var hex = Number( rgb ).toString(16);
    if (hex.length < 2) {
        hex = "0" + hex;
    }
    return hex;
};

// Convert RGB to hex 
var fullColorHex = function(r, g, b) {
    var red = rgbToHex(r);
    var green = rgbToHex(g);
    var blue = rgbToHex(b);
    return red+green+blue;
};

var cocoLabelTextToObjectID = function( text ) {
    splits = text.split("_");
    id = Number(splits[0]);
    if (splits.length > 1) {
        instanceNumber = Number(splits[1]);
    }
    if (isNaN(id)) {
        return undefined;
    } else {
        if (isNaN(instanceNumber)) {
            return [id];
        } else {
            return [id, instanceNumber];
        }
    }
};