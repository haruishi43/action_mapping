function initThreeD ( url ) {

    // Creating a scene
    scene = new THREE.Scene();
    
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 200 );
    camera.position.z = 30;
    camera.position.y = 30;
    scene.add( camera );

    gui = new dat.GUI();
    
    // Lights
    lights = [];
    lights[ 0 ] = new THREE.PointLight( 0xffffff, 1, 0 );
    lights[ 1 ] = new THREE.PointLight( 0xffffff, 1, 0 );
    lights[ 2 ] = new THREE.PointLight( 0xffffff, 1, 0 );
    lights[ 0 ].position.set( 0, 200, 0 );
    lights[ 1 ].position.set( 100, 200, 100 );
    lights[ 2 ].position.set( - 100, - 200, - 100 );
    scene.add( lights[ 0 ] );
    scene.add( lights[ 1 ] );
    scene.add( lights[ 2 ] );

    // renderer
    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    renderer.setClearColor( 0x000000, 1 );

    // DOMs
    container = document.createElement( 'div' );
    document.body.appendChild( container );
    container.appendChild( renderer.domElement );
    
    // stats
    stats = new Stats();
    container.appendChild( stats.dom );

    // helper
    var helper = new THREE.GridHelper( 160, 10 );
    // helper.rotation.x = Math.PI / 4;
    helper.position.y = -15.0;
	scene.add( helper );

    // orbit
    orbit = new THREE.OrbitControls( camera, container );
    orbit.enableZoom = true;

    // PLY file
    var loader = new THREE.PLYLoader();
    loader.load( url, function ( geometry ) {
        geometry.computeVertexNormals();
        var material = new THREE.MeshStandardMaterial( { color: 0xffffff, flatShading: true } );
        
        var roomMesh = new THREE.Mesh( geometry, material );
        // roomMesh.position.y =  0.0;
        // roomMesh.position.z =  0.0;
        roomMesh.rotation.x = - Math.PI / 2;
        roomMesh.scale.multiplyScalar( 0.01 );
        roomMesh.castShadow = true;
        roomMesh.receiveShadow = true;
        roomMesh.name = "new_room_edited.ply";
        scene.add( roomMesh );
    } );
    
    // Event listeners
    window.addEventListener( 'resize', onWindowResize, false );

}

function setupDatGui () {
    var folder = gui.addFolder( "General Options" );

    folder.add( state, "renderPose" );
    folder.__controllers[ 0 ].name( "Render Pose" );

    folder.add( state, "animateBones" );
    folder.__controllers[ 0 ].name( "Animate Bones" );

    var bones = mesh.skeleton.bones;

    // for ( var i = 0; i < bones.length; i ++ ) {
    //     var bone = bones[ i ];
    //     folder = gui.addFolder( "Bone " + i );
    //     folder.add( bone.position, 'x', - 10 + bone.position.x, 10 + bone.position.x );
    //     folder.add( bone.position, 'y', - 10 + bone.position.y, 10 + bone.position.y );
    //     folder.add( bone.position, 'z', - 10 + bone.position.z, 10 + bone.position.z );

    //     folder.add( bone.rotation, 'x', - Math.PI * 0.5, Math.PI * 0.5 );
    //     folder.add( bone.rotation, 'y', - Math.PI * 0.5, Math.PI * 0.5 );
    //     folder.add( bone.rotation, 'z', - Math.PI * 0.5, Math.PI * 0.5 );

    //     folder.add( bone.scale, 'x', 0, 2 );
    //     folder.add( bone.scale, 'y', 0, 2 );
    //     folder.add( bone.scale, 'z', 0, 2 );

    //     folder.__controllers[ 0 ].name( "position.x" );
    //     folder.__controllers[ 1 ].name( "position.y" );
    //     folder.__controllers[ 2 ].name( "position.z" );

    //     folder.__controllers[ 3 ].name( "rotation.x" );
    //     folder.__controllers[ 4 ].name( "rotation.y" );
    //     folder.__controllers[ 5 ].name( "rotation.z" );

    //     folder.__controllers[ 6 ].name( "scale.x" );
    //     folder.__controllers[ 7 ].name( "scale.y" );
    //     folder.__controllers[ 8 ].name( "scale.z" );
    // }
}

function createGeometry ( sizing ) {
    var geometry = new THREE.CylinderGeometry(
        5,                       // radiusTop
        5,                       // radiusBottom
        sizing.height,           // height
        8,                       // radiusSegments
        sizing.segmentCount * 3, // heightSegments
        true                     // openEnded
    );

    for ( var i = 0; i < geometry.vertices.length; i ++ ) {
        var vertex = geometry.vertices[ i ];
        var y = ( vertex.y + sizing.halfHeight );
        var skinIndex = Math.floor( y / sizing.segmentHeight );
        var skinWeight = ( y % sizing.segmentHeight ) / sizing.segmentHeight;
        geometry.skinIndices.push( new THREE.Vector4( skinIndex, skinIndex + 1, 0, 0 ) );
        geometry.skinWeights.push( new THREE.Vector4( 1 - skinWeight, skinWeight, 0, 0 ) );
    }
    return geometry;
}

function createBones ( sizing ) {
    bones = [];
    var prevBone = new THREE.Bone();
    bones.push( prevBone );
    prevBone.position.y = - sizing.halfHeight;

    for ( var i = 0; i < sizing.segmentCount; i ++ ) {
        var bone = new THREE.Bone();
        bone.position.y = sizing.segmentHeight;
        bones.push( bone );
        prevBone.add( bone );
        prevBone = bone;
    }
    return bones;
}

function createMesh ( geometry, bones ) {

    var material = new THREE.MeshPhongMaterial( {
        skinning : true,
        color: 0x156289,
        emissive: 0x072534,
        side: THREE.DoubleSide,
        flatShading: true
    } );

    var mesh = new THREE.SkinnedMesh( geometry,	material );
    var skeleton = new THREE.Skeleton( bones );
    mesh.add( bones[ 0 ] );
    mesh.bind( skeleton );
    skeletonHelper = new THREE.SkeletonHelper( mesh );
    skeletonHelper.material.linewidth = 2;
    scene.add( skeletonHelper );

    return mesh;
}

function initBones () {

    var segmentHeight = 8;
    var segmentCount = 4;
    var height = segmentHeight * segmentCount;
    var halfHeight = height * 0.5;

    var sizing = {
        segmentHeight : segmentHeight,
        segmentCount : segmentCount,
        height : height,
        halfHeight : halfHeight
    };

    var geometry = createGeometry( sizing );
    var bones = createBones( sizing );
    mesh = createMesh( geometry, bones );

    mesh.scale.multiplyScalar( 1 );
    scene.add( mesh );

}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth, window.innerHeight );
}

