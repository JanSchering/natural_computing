let CPM = require("../../build/artistoo-cjs.js")

let simSettings = {
    // Cells on the grid
    NRCELLS : [10, 2],					    // Number of cells to seed for all
    // non-background cellkinds.

    // Runtime etc
    BURNIN : 500,
    RUNTIME : 3000,
    RUNTIME_BROWSER : 10000,
    
    // Visualization
    //CANVASCOLOR: "eaecef",
    CANVASCOLOR : "BAB9B9",
    CELLCOLOR : ["515151", "000000"],
    ACTCOLOR : [true, true, true],			// Should pixel activity values be displayed?
    SHOWBORDERS : [true, true],				// Should cellborders be displayed?
    zoom : 4,							    // zoom in on canvas with this factor.
    
    // Output images
    SAVEIMG : true,					        // Should a png image of the grid be saved
    // during the simulation?
    IMGFRAMERATE : 1,					    // If so, do this every <IMGFRAMERATE> MCS.
    SAVEPATH : "output/img/Project",		// ... And save the image in this folder.
    EXPNAME : "t",					        // Used for the filename of output images.
    
    // Output stats etc
    STATSOUT : { browser: true, node: true },   // Should stats be computed?
    LOGRATE : 10							    // Output stats every <LOGRATE> MCS.
}

let spcConfig = {
    // Grid settings
    ndim : 2,
    field_size : [200, 200],
    
    // CPM parameters and configuration
    conf : {
        // Basic CPM parameters
        torus : [true, true],					// Should the grid have linked borders?
        seed : 1,							    // Seed for random number generation.
        // T : 15,								// CPM temperature
        T : 20,								    // CPM temperature     
        
        // Parameters used in the self-study exercise 3.3
        // Adhesion parameters:
        J : [[NaN, 0, 0], [100, 10, 1000], [20, 1000, 0]],		

        // VolumeConstraint parameters
        LAMBDA_V : [0, 40, 50], // VolumeConstraint importance per cellkind
        V : [0, 120, 200],	    // Target volume of each cellkind	
    
        // Perimeter Constraints
        LAMBDA_P: [0, 30, 2],
        P: [0, 60, 180],	
        // To keep in mind for obstacles: lambda v = 50, v = 100, lambda p = 20, p = 40
    
        // ActivityConstraint parameters					
        LAMBDA_ACT : [0, 0, 200],				// ActivityConstraint importance per cellkind
        MAX_ACT : [0, 0, 80],					// Activity memory duration per cellkind		
        ACT_MEAN : "geometric"				    // Is neighborhood activity computed as a "geometric" or "arithmetic" mean?
    },
    simsettings: simSettings
    
}

/**
 * Function to iniatilize the simulation grid
 */
function initializeGrid(){
	
    // add the GridManipulator if not already there and if you need it
	if( !this.helpClasses["gm"] ){ this.addGridManipulator() }
    
    /**
     * Function that initializes the obstacles forming the grid of the simulation
     * 
     * @param {number} width The width of the field
     * @param {number} height The height of the field
     * @param {number} numRows The number of rows forming the obstacle grid to be generated
     * @param {number} numCols The number of columns forming the obstacle grid to be generated
     */
    function initObstaclesAsGrid(width, height, numRows, numCols) {
        // set the spacing between obstacles to have a regular spacing between them
        const rowSpacing = height / (numRows + 1);
        const colSpacing = width / (numCols + 1);
        var cellCount = 0;
        // initialize the obstacles forming the grid in the field space
        [...Array(numRows).keys()].map(rowIdx => {
            [...Array(numCols).keys()].map(colIdx => {
                this.gm.seedCellAt(1, [Math.floor(colSpacing*(colIdx+1)), Math.floor(rowSpacing*(rowIdx+1))])
                cellCount++;
            })
        })
    }

    /**
     * Function that initializes the obstacles forming the two walls of the simulation
     * 
     * @param {number} leftWallX x-location of the left wall to be generated
     * @param {number} rightWallX x-location of the right wall to be generated
     * @param {number} height The height of the field
     */
    function initObstaclesAsWalls(leftWallX, rightWallX, height) {
        // set the step size that defines the vertical spacing between obstacles
        const stepSize = 10;
        var y = 0;
        // initialize the obstacles forming the walls in the field space
        while (y < height) {
            this.gm.seedCellAt(1, [leftWallX, y]);
            this.gm.seedCellAt(1, [rightWallX, y]);
            y += stepSize
        }
    }

    /**
     * Function that generates random coordinates within the field space
     * 
     * @param {number} width The width of the field
     * @param {number} height The height of the field
     * @returns The pair (x, y) of random coordinates
     */
    function getRandomCoordinates(width, height) {
        const xCoord = Math.floor(Math.random() * width)
        const yCoord = Math.floor(Math.random() * height)
        return [xCoord, yCoord]
    }

    const width = this.C.extents[0]
    const height = this.C.extents[1]
	const numRows = 5;
	const numCols = 5;
	const leftWallX = 70;
	const rightWallX = 100;

    /* Comment and uncomment the following lines depending on the type of simulation
    to be run (whether obstacle as grid or obatacles as walls). The following will 
    only iniatize the obstacles
    */
	initObstaclesAsGrid.bind(this)(width, height, numRows, numCols);
	// initObstaclesAsWalls.bind(this)(leftWallX, rightWallX, height);
    
}

let custommethods = {
    initializeGrid : initializeGrid
}

let sim = new CPM.Simulation( spcConfig, custommethods );
// let sim = new CPM.Simulation( apcConfig, custommethods );

[...Array(100).keys()].map(idx => {
    sim.step()
})

if( !sim.helpClasses["gm"] ){ sim.addGridManipulator() }

/**
 * Function that generates random coordinates within the field space
 * 
 * @param {number} width The width of the field
 * @param {number} height The height of the field
 * @returns The pair (x, y) of random coordinates
 */
function getRandomCoordinates(width, height) {
    const xCoord = Math.floor(Math.random() * width)
    const yCoord = Math.floor(Math.random() * height)
    return [xCoord, yCoord]
}

/**
 * Function that initializes the active cells for an obstacle as grid simulation
 * 
 * @param {number} width The width of the field
 * @param {number} height The height of the field
 * @param {Simulation} sim The simulation being run
 */
function initCellsForGrid(width, height, sim) {
    // set the desired number of active cells
    const numActiveCells = 150;

    // initialize the active cells at random locations in the field space
    [...Array(numActiveCells).keys()].map(idx => {
        const pos = getRandomCoordinates(width, height)
        sim.gm.seedCellAt(2, pos)
    })
}

/**
 * Function that initializes the active cells for an obstacle as walls simulation
 * 
 * @param {number} height The height of the field
 * @param {Simulation} sim The simulation being run
 */
function initCellsForWalls(height, sim) {
    // set the desired number of active cells
    const numCells = 3;
    const leftWallX = 70;
    const rightWallX = 100;
    
    // initialize the active cells at random locations in between both walls
    [...Array(numCells).keys()].map(idx => {
        const innerWidth = rightWallX - leftWallX;
        const pos = getRandomCoordinates(innerWidth, height)
        sim.gm.seedCellAt(2, [leftWallX + pos[0], pos[1]])
    })
}

const width = sim.C.extents[0]
const height = sim.C.extents[1];

/* Comment and uncomment the following lines depending on the type of simulation
to be run (whether obstacle as grid or obatacles as walls). The following will 
only iniatize the active cells
*/
initCellsForGrid(width, height, sim);
// initCellsForWalls(height, sim)


sim.run()
