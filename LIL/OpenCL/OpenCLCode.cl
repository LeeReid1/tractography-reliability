__kernel void MultiplyPointwise(__global float * v1, __global float * v2,__global float * result)
{
    // Vector element index
    int i = get_global_id(0);
    result[i] = v1[i] * v2[i];
}
__kernel void Test(__global float * v1, __global float * v2,__global float * result)
{
    // Vector element index
    int i = get_global_id(0);
    //result[i] = v1[i] * v2[i] / 18.0 * (v1[i] * v1[i] + 7.0) / 90.0;
	result[i] =sqrt( v1[i] * v2[i] / 18.0 * (v1[i] * v1[i] + 7.0));
}


//-------------MATRIX MULTIPLICATION

//Multiplies a single 4colx3row matrix by a (series of) 3D coords
//The matrix should be ordered row-at-a-time
__kernel void MultiplyMatrix4x3ByVector3(__global float * mat,__global float * x,__global float * y,__global float * z,__global float * result)
{
	int i = get_global_id(0);
	//result[i * 3]		= x[i] * mat_0_0 + y[i] * mat_0_1 + z[i] * mat_0_2 + mat_0_3;
	//result[i * 3 + 1]	= x[i] * mat_1_0 + y[i] * mat_1_1 + z[i] * mat_1_2 + mat_1_3;
	//result[i * 3 + 2]	= x[i] * mat_2_0 + y[i] * mat_2_1 + z[i] * mat_2_2 + mat_2_3;	
	
	//result[i * 3]		= x[i] * mat[0] + y[i] * mat[1] + z[i] * mat[2] + mat[3];
	//result[i * 3 + 1]	= x[i] * mat[4] + y[i] * mat[5] + z[i] * mat[6] + mat[7];
	//result[i * 3 + 2]	= x[i] * mat[8] + y[i] * mat[9] + z[i] * mat[10] + mat[11];	

	private float _x = x[i];
	private float _y = y[i];
	private float _z = z[i];
		result[i * 3]		= _x * mat[0] + _y * mat[1] + _z * mat[2] + mat[3];
	result[i * 3 + 1]	= _x * mat[4] + _y * mat[5] + _z * mat[6] + mat[7];
	result[i * 3 + 2]	= _x * mat[8] + _y * mat[9] + _z * mat[10] + mat[11];	
}


/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 * modified from http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl
 */
 
// OpenCL Kernel
__kernel void matrixMul_f(__global float* A, __global float* B, int nColsARowsB,int nRowsA, int nColsB, __global float* C)
{
  
   // 2D Thread ID
   // Old CUDA code
   //int tx = blockIdx.x * TILE_SIZE + threadIdx.x;
   //int ty = blockIdx.y * TILE_SIZE + threadIdx.y;
   int tx = get_global_id(0); //column number for B
   int ty = get_global_id(1);//row number for A
 
   // value stores the element that is 
   // computed by the thread
   float value = 0;
   for (int k = 0; k < nColsARowsB; ++k)
   {
      float elementA = A[ty * nColsARowsB + k];//A: row ty, column k
      float elementB = B[k * nColsB + tx];//B: row k, column tx
      value += elementA * elementB;
	  //value = elementA;
   }
 
   // Write the matrix to device memory each 
   // thread writes one element
   C[ty * nColsB + tx] =value;//A[ty * nColsARowsB];//value;
}

/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 * modified from http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl
 */
 
// OpenCL Kernel
__kernel void matrixMul_int64(__global long* A, __global long* B, int nColsARowsB,int nRowsA, int nColsB, __global long* C)
{
  
   // 2D Thread ID
   // Old CUDA code
   //int tx = blockIdx.x * TILE_SIZE + threadIdx.x;
   //int ty = blockIdx.y * TILE_SIZE + threadIdx.y;
   int tx = get_global_id(0); //column number for B
   int ty = get_global_id(1);//row number for A
 
   // value stores the element that is 
   // computed by the thread
   long value = 0;
   for (int k = 0; k < nColsARowsB; ++k)
   {
      long elementA = A[ty * nColsARowsB + k];//A: row ty, column k
      long elementB = B[k * nColsB + tx];//B: row k, column tx
      value += elementA * elementB;
	  //value = elementA;
   }
 
   // Write the matrix to device memory each 
   // thread writes one element
   C[ty * nColsB + tx] =value;//A[ty * nColsARowsB];//value;
}




// OpenCL Kernel
//Calculates the distance between each point. matrix A should be a 3 column matrix

__kernel void distMatrix_f_old(__global float* A, int nRows, int startFrom, __global float* C)
{

	//Equation is sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2) 
	//Takes in one matrix only, organised column-at-a-time (x,x,x,x...y,y,y,y,y,...z,z,z,z,z...). Should be 3 columns, each coords on a new row
	//pointwise subtracts row tx by ty
	//squares the subtractions 
	//sqrts the sum of the squares
	//Returns an array of distances, ordered r0vsr0, r0vsr1, r0vsr2, etc


	// 2D Thread ID
	int row1 = get_global_id(0); //row 1
	int row2 = get_global_id(1)+startFrom;//row 2
 
	int posX1 = row1;
	int posX2 = row2;
   
    //X
	float element = A[posX1];
	float elementB = A[posX2];
	float diff = element-elementB;
	float value = diff*diff;

	//Y
	element = A[posX1+nRows];//A: row ty, column 1
	elementB = A[posX2+nRows];
	diff = element-elementB;
	value += diff*diff;

	//Z
	element = A[posX1+2*nRows];//A: row ty, column 2
	elementB = A[posX2+2*nRows];
	diff = element-elementB;
	  
	value = sqrt(value+ diff*diff);

 
	// Write the matrix to device memory - each thread writes one element
	C[row1 * nRows + row2] =value;
}

__kernel void distMatrix_f(__global float* A, int nRows, int startFrom, int nColsInFinalResult, __global float* C)
{

	//Equation is sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2) 
	//Takes in one matrix only, organised column-at-a-time (x,x,x,x...y,y,y,y,y,...z,z,z,z,z...). Should be 3 columns, each coords on a new row
	//pointwise subtracts row tx by ty
	//squares the subtractions 
	//sqrts the sum of the squares
	//Returns an array of distances, ordered r0vsr0, r0vsr1, r0vsr2, etc


	// 2D Thread ID
	int id0 = get_global_id(0);
	int row1 = id0+startFrom; //row 1
	int row2 = get_global_id(1)+startFrom;//row 2
 
	int posX1 = row1;
	int posX2 = row2;
   
    //X
	float element = A[posX1];
	float elementB = A[posX2];
	float diff = element-elementB;
	float value = diff*diff;

	//Y
	element = A[posX1+nRows];//A: row ty, column 1
	elementB = A[posX2+nRows];
	diff = element-elementB;
	value += diff*diff;

	//Z
	element = A[posX1+2*nRows];//A: row ty, column 2
	elementB = A[posX2+2*nRows];
	diff = element-elementB;
	  
	value = sqrt(value+ diff*diff);

 
	// Write the matrix to device memory - each thread writes one element
	C[id0 * nColsInFinalResult + row2] =row1;//value;
}

__kernel void SelfDifference_int(__global int* A, int vectorLength,int row2WasOffestBy, __global int* C)
{

	//Equation is sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2) 
	//Takes in one vector only
	//pointwise subtracts row x by row y
	//saves to [x][y] in a matrix
	//Returns matrix as an array, ordered row at a time


	// 2D Thread ID
	int col = get_global_id(0); //column of result (first row to read from)
	int row = get_global_id(1);//row of result (second row to read from)
   
	int diff = A[row] - A[col];
	
	// Write the matrix to device memory - each thread writes one element
	
	C[(row -row2WasOffestBy)* vectorLength + col]=diff;
	//C[row1* vectorLength + (row2-row2WasOffestBy)]=row1;
}


__kernel void PointwiseDifference_int(__global int* Left, __global int* Top, int vectorLength,int rowWasOffsetBy, __global int* C)
{

	//Equation is sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2) 
	//Takes in one vector only
	//pointwise subtracts row x by row y
	//saves to [x][y] in a matrix
	//Returns matrix as an array, ordered row at a time


	// 2D Thread ID
	int row = get_global_id(0); //row of result (first row to read from)
	int col = get_global_id(1);//column of result (second row to read from)
   
	int diff = Left[row] - Top[col];
	
	// Write the matrix to device memory - each thread writes one element
	
	C[(row-rowWasOffsetBy)* vectorLength + col]=diff;
}
