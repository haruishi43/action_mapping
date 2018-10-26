/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Pose
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Variables:

var jointColors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85]
];

var limbColors = [
    [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
    [0, 85, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0.],
    [255, 0, 85], [170, 255, 0], [85, 255, 0], [170, 0, 255.], [0, 0, 255],
    [0, 0, 255], [255, 0, 255], [170, 0, 255], [255, 0, 170]
];

var limbConnections = [ // connect which joint to which (by index)
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], 
    [12, 13], [1, 2], [2, 3], [3, 4], [1, 5], 
    [5, 6], [6, 7], [1, 0], [0, 14], [0, 15], 
    [14, 16], [15, 17]
];

/// Functions:

function initHuman() {
    var joints = [];
    var limbs = [];

    // create geometry for joints
    for (var i=0; i<jointColors.length; i++) {
        
        // initialize joint points
        
        var jointColor = jointColors[i];
        var jointColorHex = fullColorHex(jointColor[0], jointColor[1], jointColor[2]);
        jointColorHex = new THREE.Color(parseInt(jointColorHex, 16));

        // var jointGeometry = new THREE.Geometry();
        // var jointMaterial = new THREE.PointsMaterial( { size: 1, sizeAttenuation: false });
        // jointMaterial.color = jointColorHex;
        // jointGeometry.colors = jointColorHex;
        // jointGeometry.computeBoundingBox();
        // var joint = new THREE.Points( jointGeometry, jointMaterial );
        // joint.scale.set( 10,10,10 );
        
        var geometry = new THREE.SphereGeometry( 0.3, 8, 8 );
        var material = new THREE.MeshBasicMaterial( {color: jointColorHex} );
        var joint = new THREE.Mesh(geometry, material);

        
        renderGroup.add(joint);

        joints.push(joint);

    }
    
    for (var j=0; j<limbConnections.length; j++) {
        var connectionPair = limbConnections[j];
        var n1 = connectionPair[0];
        var n2 = connectionPair[1];

        var v1 = joints[n1].position;
        var v2 = joints[n2].position;

        var lineColor = limbColors[j];
        var lineColorHex = fullColorHex(lineColor[0], lineColor[1], lineColor[2]);
        lineColorHex = new THREE.Color(parseInt(lineColorHex, 16));

        var lineGeometry = new THREE.Geometry();
        lineGeometry.vertices.push(v1);
        lineGeometry.vertices.push(v2);
        lineGeometry.dynamic = true;
        lineGeometry.verticesNeedUpdate = true;

        var lineMaterial = new THREE.LineBasicMaterial( { color: lineColorHex, opacity: 0.5, linewidth: 3 } );
        var line = new THREE.Line(lineGeometry, lineMaterial);

        renderGroup.add(line);

        limbs.push(line);
    }

    jointGroup.push(joints);
    limbGroup.push(limbs);
}

function updateJoints(index, pose) {
    // call this function when new pose needs to be added
    
    for (var i=0; i<jointColors.length; i++) {
        var id = i.toString();
        if (typeof pose[id] === "undefined") {
            jointGroup[index][i].visible = false;
        }
        else {
            var x = pose[id][0];
            var y = pose[id][1];
            var z = pose[id][2];

            jointGroup[index][i].visible = true;
            jointGroup[index][i].position.set(x, y, z);
        }
    }
    
    // update line
    var ids = Object.keys(pose);
    for (var j=0; j<limbConnections.length; j++) {
        var n1 = limbConnections[j][0].toString();
        var n2 = limbConnections[j][1].toString();

        limbGroup[index][j].geometry.verticesNeedUpdate = true;
        limbGroup[index][j].computeLineDistances();
        if (ids.includes(n1) && ids.includes(n2)) {
            
            limbGroup[index][j].visible = true;
        } else {
            limbGroup[index][j].visible = false;
        }
    }
}

function updatePoses(poses) {
    if (typeof poses === "undefined") {
        console.log("something is wrong");
    } else {
        var jsonPoses = JSON.parse(poses);
        var currentPoseCount = Object.keys(jsonPoses).length;
        var posesShouldDisappear = jointGroup.length - currentPoseCount;

        for (var key in jsonPoses) {

            var i = Number(key);
            var pose = jsonPoses[key];

            if (i+1 > jointGroup.length) {
                initHuman();
                console.log("new pose")
            }

            // if (i === 0) {
            // 	// only a single person for now
            // 	updateJoints(i, pose);
            // }
            updateJoints(i, pose);
        }
        
        // make other pose disappear
        for (var i=currentPoseCount; i<jointGroup.length; i++) {
            for (var j=0; j<jointColors.length; j++) {
                jointGroup[i][j].visible = false;
            }
        }

        for (var i=currentPoseCount; i<limbGroup.length; i++) {
            for (var j=0; j<limbConnections.length; j++) {
                limbGroup[i][j].visible = false;
            }
        }

    }
}