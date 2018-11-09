
var THREASHOLD = 1.0;

function detectDrinkAction(nosePosition, cupPosition) {
    var distance = nosePosition.distanceTo(cupPosition);
    // console.log(distance);
    if (distance < THREASHOLD) {
        createDrinkText(cupPosition);
        return true;
    } else {

        return false;
    }
}

function createDrinkText(pos) {

    var actionName = "drink"
    var canvas1 = document.createElement('canvas');
    canvas1.height = 128;
    canvas1.width = 256;
    var context1 = canvas1.getContext('2d');
    context1.font = "Bold 40px Arial";
    context1.fillStyle = "rgba(255,255,255,1)";
    context1.fillText(actionName, 0, 64);
    context1.textAlign = "center";
    
    // canvas contents will be used for a texture
    var texture1 = new THREE.Texture(canvas1) 
    texture1.needsUpdate = true;
    
    var material1 = new THREE.MeshBasicMaterial( {map: texture1, side:THREE.DoubleSide } );
    material1.transparent = true;
    var mesh1 = new THREE.Mesh(
        new THREE.PlaneGeometry(8, 8),
        material1
    );
    mesh1.position.set(pos.x, pos.y, pos.z);
    mesh1.rotation.x = Math.PI / 2; 
    renderGroup.add( mesh1 );
    
    drinkAction = mesh1;
}

function removeDrinkText() {
    if (!(drinkAction===undefined)) {
        renderGroup.remove(drinkAction);
        drinkAction.geometry.dispose();
        drinkAction.material.dispose();
        
    }
    drinkAction = undefined;
}