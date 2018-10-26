/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Initializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Initialize renderers
function initThreeD ( url ) {

    // Creating a scene
    scene = new THREE.Scene();
    renderGroup = new THREE.Group();
    
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
    var gridSize = 60;
    var gridDivision = 60;
    var helper = new THREE.GridHelper( gridSize, gridDivision );
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
        roomMesh.rotation.x = 0; //-Math.PI / 2;
        roomMesh.scale.multiplyScalar( 0.01 );
        roomMesh.castShadow = true;
        roomMesh.receiveShadow = true;
        roomMesh.name = "new_room_edited.ply";
        room = roomMesh;
        renderGroup.add( room );
    } );

    renderGroup.rotation.x = -Math.PI / 2;
    scene.add( renderGroup );
    
    // Event listeners
    window.addEventListener( 'resize', onWindowResize, false );

}

function setupDatGui () {
    var folder = gui.addFolder( "General Options" );

    folder.add( state, "renderPose" );
    folder.__controllers[ 0 ].name( "Render Pose" );

    folder.add( state, "renderRoom" );
    folder.__controllers[ 1 ].name( "Render Room" );

}