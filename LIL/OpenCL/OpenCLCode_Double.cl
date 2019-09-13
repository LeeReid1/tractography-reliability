#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error //Double precision floating point not supported by OpenCL implementation.
#endif

__kernel void MultiplyPointwiseDouble(__global float * v1, __global double * v2,__global double * result)
{
    // Vector element index
    int i = get_global_id(0);
    result[i] = v1[i] * v2[i];
}

__kernel void MultiplyPointwise_d(__global double * v1, __global double * v2,__global double * result)
{
    // Vector element index
    int i = get_global_id(0);
    result[i] = v1[i] * v2[i];
}
__kernel void MultiplyPointwiseDouble_d(__global double * v1, __global double * v2,__global double * result)
{
    // Vector element index
    int i = get_global_id(0);
    result[i] = v1[i] * v2[i];
}



/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 * modified from http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmopencl
 */
 
// OpenCL Kernel
__kernel void matrixMul_d(__global double* A, __global double* B, int nColsARowsB,int nRowsA, int nColsB, __global double* C)
{
  
   // 2D Thread ID
   // Old CUDA code
   //int tx = blockIdx.x * TILE_SIZE + threadIdx.x;
   //int ty = blockIdx.y * TILE_SIZE + threadIdx.y;
   int tx = get_global_id(0); //column number for B
   int ty = get_global_id(1);//row number for A
 
   // value stores the element that is 
   // computed by the thread
   double value = 0;
   for (int k = 0; k < nColsARowsB; ++k)
   {
      double elementA = A[ty * nColsARowsB + k];//A: row ty, column k
      double elementB = B[k * nColsB + tx];//B: row k, column tx
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
 */
 
// OpenCL Kernel
//For when A is columnwise.
__kernel void matrixMul_d_AColumnwise(__global double* A, __global double* B, int nColsARowsB,int nRowsA, int nColsB, __global double* C)
{
  
   // 2D Thread ID
   // Old CUDA code
   //int tx = blockIdx.x * TILE_SIZE + threadIdx.x;
   //int ty = blockIdx.y * TILE_SIZE + threadIdx.y;
   int tx = get_global_id(0); //column number for B
   int ty = get_global_id(1);//row number for A
 
   // value stores the element that is 
   // computed by the thread
   double value = 0;
   for (int k = 0; k < nColsARowsB; ++k)
   {
   
      double elementA = A[nRowsA * k + ty];//A: row ty, column k
      double elementB = B[k * nColsB + tx];//B: row k, column tx
      value += elementA * elementB;
	  //value = elementA;
   }
 
   // Write the matrix to device memory each 
   // thread writes one element
   C[ty * nColsB + tx] = value;//A[ty];//wA;// A[ty * wA];//value;
}

//adds all values in the input
//Assumes a workgroup size of 1024
kernel void ReduceSum_d(global double* g_idata, int n, global double* result)
{
	__local double sdata[1024]; 

    const int workgroupSize = 1024;//get_local_size(0);//no items in this workgroup
    
	


    //int itemsPerWorkgroup = nItemsPerWorkgroup;//n / workgroupSize;

    int tid = get_local_id(0);//Work-item (Thread) ID
    
    // int globalStart =  get_group_id(0)*(blockSize*2) + tid;
    int gridSize = workgroupSize*2*get_num_groups(0);
    
    
	
    

    
    //Semi-unrolled first loop - read from global memory, and add every second, saving to local memory
    //We continue doing this until we have incorporated every item into the local array
    //Pretend there are 16 numbers and two work groups each with a size of FOUR. This loop adds numbers in like so:
    //a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 d0 d1 d2 d3
    //Workgroup 0
    //localArray={a0 + b0, a1 + b1, a2 + b2, a3 + b3}
    //Workgroup 1
    //localArray={c0 + d0, c1 + d1, c2 + d2, c3 + d3}
    
    //If there were 16 numbers, and two work groups each with a size of TWO, this loop would execute twice:
    //Workgroup 0
    //-Loop 0:
    //localArray={a0 + a2, a1 + a3}
    //-Loop 1:
    //--thread0: localArray[0] += (c0 + c2)
    //--thread1: localArray[1] +=  (c1 + c3)
    //Workgroup 1
    //-Loop 0:
    //localArray={b0 + b2, b1 + b3}
    //-Loop 1:
    //--thread0: localArray[0] += (d0 + d2)
    //--thread1: localArray[1] +=  (d1 + d3)

	
	if(true)
    {    
	
	int i = get_group_id(0)*(workgroupSize*2) + tid;

	if(i + workgroupSize < n)
	{
		sdata[tid] = 0;//intialise at zero
		while (i < n)
		{
			//result[get_group_id(0)] =i;
			sdata[tid] += g_idata[i] + g_idata[i+workgroupSize];  
			//sdata[tid] =i+workgroupSize;  
			i += gridSize;  
		}
	} 
	else
	{
		sdata[tid] = g_idata[i];  
	}

    
    
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads

    //Unrolled 2nd loop - read from local memory, and add every second, saving to local memory
	
    //if (workgroupSize >= 1024) //NVidia cards on bragg have max worksize of 1024, which is why we start here
    { 
        if (tid < 512) 
        { 
            sdata[tid] += sdata[tid + 512]; 
        } 
        barrier(CLK_LOCAL_MEM_FENCE); //Sync threads
    }
	

    //Unrolled 3rd loop 
    //if (workgroupSize >= 512) 
    { 
        if (tid < 256) 
        { 
            sdata[tid] += sdata[tid + 256]; 
        } 
        barrier(CLK_LOCAL_MEM_FENCE); //Sync threads
    }

    //Unrolled 4th loop 
    //if (workgroupSize >= 256) 
    { 
        if (tid < 128) 
        { 
            sdata[tid] += sdata[tid + 128]; 
        } 
        barrier(CLK_LOCAL_MEM_FENCE); //Sync threads
    }


    //Unrolled 5th loop 
    //if (workgroupSize >= 128) 
    {
        if (tid < 64) 
        { 
            sdata[tid] += sdata[tid +   64]; 
        } 
        barrier(CLK_LOCAL_MEM_FENCE); //Sync threads
    }

    //Unrolled 6th? loop(s?)
    if (tid < 32)
    {
      //  if (workgroupSize >= 64) 
        {
            sdata[tid] += sdata[tid + 32];
        }
        //if (workgroupSize >= 32)
        {        
            sdata[tid] += sdata[tid + 16];
        }
        //if (workgroupSize >= 16)
        {
            sdata[tid] += sdata[tid +  8];
        }
        //if (workgroupSize >= 8)
        {
            sdata[tid] += sdata[tid +  4];
        }
        //if(workgroupSize >= 4)
        {
            sdata[tid] += sdata[tid +  2];
        }
        //if (workgroupSize >= 2)
        {
            sdata[tid] += sdata[tid +  1];
        }
    }
	

    if (tid == 0) 
    {
        result[get_group_id(0)] = sdata[0];
    }
    }
}




//adds all values in the input
//Assumes a workgroup size of 1024. Each workgroup processes 2048 values and puts the result into 'result'
kernel void ReduceSum_d_2048(global double* g_idata, global double* result, int tidyIfGroup, int tidyIfThreadLessThan)
{
	__local double sdata[1024]; 

    const int workgroupSize = 1024;//get_local_size(0);//no items in this workgroup
    
	


    //int itemsPerWorkgroup = nItemsPerWorkgroup;//n / workgroupSize;

    int tid = get_local_id(0);//Work-item (Thread) ID
    
    
    
  

    
    //Unrolled first loop - read from global memory, and add every second, saving to local memory
    //Pretend there are 16 numbers and two work groups each with a size of FOUR. (in reality, it's 2048 and 1024) .This adds numbers in like so:
    //a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 d0 d1 d2 d3
    //Workgroup 0
    //localArray={a0 + b0, a1 + b1, a2 + b2, a3 + b3}
    //Workgroup 1
    //localArray={c0 + d0, c1 + d1, c2 + d2, c3 + d3}
	{
		int groupId=get_group_id(0);
		int i = groupId*2048 + tid;
		double initRes=g_idata[i]+g_idata[i+1024];

		if(tidyIfGroup == groupId && tid < tidyIfThreadLessThan)
		{
			//There is a partial block after the end which has no assigned workgroup and 
			//so this workgroup must clean up

			int tidyIfThreadLessThan_2 = tidyIfThreadLessThan-1024;
			if(tid < tidyIfThreadLessThan_2)
			{
				//Include a third and fourth value
				initRes += g_idata[i+2048]+g_idata[i+2048+1024];
			}
			else
			{
				//Include a third value
				initRes += g_idata[i+2048];  	
			}
		}
		sdata[tid] = initRes;
	}
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads

    //Unrolled 2nd loop - read from local memory, and add every second, saving to local memory
	//1024 --> 512
    if (tid < 512) 
    { 
        sdata[tid] += sdata[tid + 512]; 
		
    } 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads

    //Unrolled 3rd loop 
	//512 --> 256
	if (tid < 256) 
        { 
            sdata[tid] += sdata[tid + 256]; 
        } 
        barrier(CLK_LOCAL_MEM_FENCE); //Sync threads
    

    //Unrolled 4th loop 
	//256 --> 128
    if (tid < 128) 
    { 
        sdata[tid] += sdata[tid + 128]; 
    } 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads


    //Unrolled 5th loop 
	//128 --> 64
    if (tid < 64) 
    { 
        sdata[tid] += sdata[tid + 64]; 
	} 
	barrier(CLK_LOCAL_MEM_FENCE); //Sync threads


    //Unrolled 6th? loop(s?)
	//64 numbers left to go
    if (tid < 32)
    {
		//64 --> 32
        sdata[tid] += sdata[tid + 32];//e.g. [0] = [0] + [32]; [31] = [31] + [63]
		//32 --> 16
        sdata[tid] += sdata[tid + 16];//e.g. [0] = [0] + [16]; [15] = [15] + [31]
		//16 --> 8
        sdata[tid] += sdata[tid +  8];//e.g. [0] = [0] + [8]
		//8 --> 4
        sdata[tid] += sdata[tid +  4];
		//4 --> 2
        sdata[tid] += sdata[tid +  2];     
		//2 --> 1
        sdata[tid] += sdata[tid +  1];
	}
	
	//save result
    if (tid == 0) 
    {
        result[get_group_id(0)] = sdata[0];
    }
	
}

//Assumes a workgroup size of 512. Each workgroup processes 1024 values and puts the result into 'result'
kernel void ReduceSum_d_1024(global double* g_idata, global double* result, int tidyIfGroup, int tidyIfThreadLessThan)
{
	__local double sdata[512]; 

    const int workgroupSize = 512;//get_local_size(0);//no items in this workgroup
    
	


    //int itemsPerWorkgroup = nItemsPerWorkgroup;//n / workgroupSize;

    int tid = get_local_id(0);//Work-item (Thread) ID
    
    
    
  

    
    //Unrolled first loop - read from global memory, and add every second, saving to local memory
    //Pretend there are 16 numbers and two work groups each with a size of FOUR. (in reality, it's 1024 and 512) .This adds numbers in like so:
    //a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 d0 d1 d2 d3
    //Workgroup 0
    //localArray={a0 + b0, a1 + b1, a2 + b2, a3 + b3}
    //Workgroup 1
    //localArray={c0 + d0, c1 + d1, c2 + d2, c3 + d3}
	{
		int groupId=get_group_id(0);
		int i = groupId*1024 + tid;
		double initRes=g_idata[i]+g_idata[i+512];

		if(tidyIfGroup == groupId && tid < tidyIfThreadLessThan)
		{
			//There is a partial block after the end which has no assigned workgroup and 
			//so this workgroup must clean up

			int tidyIfThreadLessThan_2 = tidyIfThreadLessThan-512;
			if(tid < tidyIfThreadLessThan_2)
			{
				//Include a third and fourth value
				initRes += g_idata[i+1024]+g_idata[i+1024+512];
			}
			else
			{
				//Include a third value
				initRes += g_idata[i+1024];  	
			}
		}
		sdata[tid] = initRes;
	}
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads

    //2nd loop skipped (code copied from the 2048 version where this is needed) 
    
    //Unrolled 3rd loop 
	//512 --> 256
	if (tid < 256) 
        { 
            sdata[tid] += sdata[tid + 256]; 
        } 
        barrier(CLK_LOCAL_MEM_FENCE); //Sync threads
    

    //Unrolled 4th loop 
	//256 --> 128
    if (tid < 128) 
    { 
        sdata[tid] += sdata[tid + 128]; 
    } 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads


    //Unrolled 5th loop 
	//128 --> 64
    if (tid < 64) 
    { 
        sdata[tid] += sdata[tid + 64]; 
	} 
	barrier(CLK_LOCAL_MEM_FENCE); //Sync threads


    //Unrolled 6th? loop(s?)
	//64 numbers left to go
    if (tid < 32)
    {
		//64 --> 32
        sdata[tid] += sdata[tid + 32];//e.g. [0] = [0] + [32]; [31] = [31] + [63]
		//32 --> 16
        sdata[tid] += sdata[tid + 16];//e.g. [0] = [0] + [16]; [15] = [15] + [31]
		//16 --> 8
        sdata[tid] += sdata[tid +  8];//e.g. [0] = [0] + [8]
		//8 --> 4
        sdata[tid] += sdata[tid +  4];
		//4 --> 2
        sdata[tid] += sdata[tid +  2];     
		//2 --> 1
        sdata[tid] += sdata[tid +  1];
	}
	
	//save result
    if (tid == 0) 
    {
        result[get_group_id(0)] = sdata[0];
    }
	
}



//adds all values in the input
//Assumes a workgroup size of 256. Each workgroup processes 512 values and puts the result into 'result'
kernel void ReduceSum_d_512(global double* g_idata, global double* result, int tidyIfGroup, int tidyIfThreadLessThan)
{
	__local double sdata[256]; 

    const int workgroupSize = 256;//get_local_size(0);//no items in this workgroup
    
	


    //int itemsPerWorkgroup = nItemsPerWorkgroup;//n / workgroupSize;

    int tid = get_local_id(0);//Work-item (Thread) ID
    
    
    
  

    
    //Unrolled first loop - read from global memory, and add every second, saving to local memory
    //Pretend there are 16 numbers and two work groups each with a size of FOUR. (in reality, it's 512 and 256) .This adds numbers in like so:
    //a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 d0 d1 d2 d3
    //Workgroup 0
    //localArray={a0 + b0, a1 + b1, a2 + b2, a3 + b3}
    //Workgroup 1
    //localArray={c0 + d0, c1 + d1, c2 + d2, c3 + d3}
	{
		int groupId=get_group_id(0);
		int i = groupId*512 + tid;
		double initRes=g_idata[i]+g_idata[i+256];

		if(tidyIfGroup == groupId && tid < tidyIfThreadLessThan)
		{
			//There is a partial block after the end which has no assigned workgroup and 
			//so this workgroup must clean up

			int tidyIfThreadLessThan_2 = tidyIfThreadLessThan-256;
			if(tid < tidyIfThreadLessThan_2)
			{
				//Include a third and fourth value
				initRes += g_idata[i+512]+g_idata[i+512+256];
			}
			else
			{
				//Include a third value
				initRes += g_idata[i+512];  	
			}
		}
		sdata[tid] = initRes;
	}
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads

    //2nd loop skipped (code copied from the 2048 version where this is needed) 
    //3rd loop skipped (code copied from the 2048 version where this is needed) 
    

    //Unrolled 4th loop 
	//256 --> 128
    if (tid < 128) 
    { 
        sdata[tid] += sdata[tid + 128]; 
    } 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads


    //Unrolled 5th loop 
	//128 --> 64
    if (tid < 64) 
    { 
        sdata[tid] += sdata[tid + 64]; 
	} 
	barrier(CLK_LOCAL_MEM_FENCE); //Sync threads


    //Unrolled 6th? loop(s?)
	//64 numbers left to go
    if (tid < 32)
    {
		//64 --> 32
        sdata[tid] += sdata[tid + 32];//e.g. [0] = [0] + [32]; [31] = [31] + [63]
		//32 --> 16
        sdata[tid] += sdata[tid + 16];//e.g. [0] = [0] + [16]; [15] = [15] + [31]
		//16 --> 8
        sdata[tid] += sdata[tid +  8];//e.g. [0] = [0] + [8]
		//8 --> 4
        sdata[tid] += sdata[tid +  4];
		//4 --> 2
        sdata[tid] += sdata[tid +  2];     
		//2 --> 1
        sdata[tid] += sdata[tid +  1];
	}
	
	//save result
    if (tid == 0) 
    {
        result[get_group_id(0)] = sdata[0];
    }
	
}

//adds all values in the input
//Assumes a workgroup size of 128. Each workgroup processes 256 values and puts the result into 'result'
kernel void ReduceSum_d_256(global double* g_idata, global double* result, int tidyIfGroup, int tidyIfThreadLessThan)
{
	__local double sdata[128]; 

    const int workgroupSize = 128;//get_local_size(0);//no items in this workgroup
    
	


    //int itemsPerWorkgroup = nItemsPerWorkgroup;//n / workgroupSize;

    int tid = get_local_id(0);//Work-item (Thread) ID
    
    
    
  

    
    //Unrolled first loop - read from global memory, and add every second, saving to local memory
    //Pretend there are 16 numbers and two work groups each with a size of FOUR. (in reality, it's 256 and 128) .This adds numbers in like so:
    //a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 d0 d1 d2 d3
    //Workgroup 0
    //localArray={a0 + b0, a1 + b1, a2 + b2, a3 + b3}
    //Workgroup 1
    //localArray={c0 + d0, c1 + d1, c2 + d2, c3 + d3}
	{
		int groupId=get_group_id(0);
		int i = groupId*256 + tid;
		double initRes=g_idata[i]+g_idata[i+128];

		if(tidyIfGroup == groupId && tid < tidyIfThreadLessThan)
		{
			//There is a partial block after the end which has no assigned workgroup and 
			//so this workgroup must clean up

			int tidyIfThreadLessThan_2 = tidyIfThreadLessThan-128;
			if(tid < tidyIfThreadLessThan_2)
			{
				//Include a third and fourth value
				initRes += g_idata[i+256]+g_idata[i+256+128];
			}
			else
			{
				//Include a third value
				initRes += g_idata[i+256];  	
			}
		}
		sdata[tid] = initRes;
	}
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads

    //2nd loop skipped (code copied from the 2048 version where this is needed) 
	//3rd loop skipped (code copied from the 2048 version where this is needed) 
	//4th loop skipped (code copied from the 2048 version where this is needed) 
    

    //Unrolled 5th loop 
	//128 --> 64
    if (tid < 64) 
    { 
        sdata[tid] += sdata[tid + 64]; 
	} 
	barrier(CLK_LOCAL_MEM_FENCE); //Sync threads


    //Unrolled 6th? loop(s?)
	//64 numbers left to go
    if (tid < 32)
    {
		//64 --> 32
        sdata[tid] += sdata[tid + 32];//e.g. [0] = [0] + [32]; [31] = [31] + [63]
		//32 --> 16
        sdata[tid] += sdata[tid + 16];//e.g. [0] = [0] + [16]; [15] = [15] + [31]
		//16 --> 8
        sdata[tid] += sdata[tid +  8];//e.g. [0] = [0] + [8]
		//8 --> 4
        sdata[tid] += sdata[tid +  4];
		//4 --> 2
        sdata[tid] += sdata[tid +  2];     
		//2 --> 1
        sdata[tid] += sdata[tid +  1];
	}
	
	//save result
    if (tid == 0) 
    {
        result[get_group_id(0)] = sdata[0];
    }
	
}

//adds all values in the input
//Assumes a workgroup size of 64. Each workgroup processes 128 values and puts the result into 'result'
kernel void ReduceSum_d_128(global double* g_idata, global double* result, int tidyIfGroup, int tidyIfThreadLessThan)
{
	__local double sdata[64]; 

    const int workgroupSize = 64;//get_local_size(0);//no items in this workgroup
    
	


    //int itemsPerWorkgroup = nItemsPerWorkgroup;//n / workgroupSize;

    int tid = get_local_id(0);//Work-item (Thread) ID
    
    
    
  

    
    //Unrolled first loop - read from global memory, and add every second, saving to local memory
    //Pretend there are 16 numbers and two work groups each with a size of FOUR. (in reality, it's 128 and 64) .This adds numbers in like so:
    //a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 d0 d1 d2 d3
    //Workgroup 0
    //localArray={a0 + b0, a1 + b1, a2 + b2, a3 + b3}
    //Workgroup 1
    //localArray={c0 + d0, c1 + d1, c2 + d2, c3 + d3}
	{
		int groupId=get_group_id(0);
		int i = groupId*128 + tid;
		double initRes=g_idata[i]+g_idata[i+64];

		if(tidyIfGroup == groupId && tid < tidyIfThreadLessThan)
		{
			//There is a partial block after the end which has no assigned workgroup and 
			//so this workgroup must clean up

			int tidyIfThreadLessThan_2 = tidyIfThreadLessThan-64;
			if(tid < tidyIfThreadLessThan_2)
			{
				//Include a third and fourth value
				initRes += g_idata[i+128]+g_idata[i+128+64];
			}
			else
			{
				//Include a third value
				initRes += g_idata[i+128];  	
			}
		}
		sdata[tid] = initRes;
	}
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads

    //2nd loop skipped (code copied from the 2048 version where this is needed) 
    //3rd loop skipped (code copied from the 2048 version where this is needed) 
	//4th loop skipped (code copied from the 2048 version where this is needed) 
	//5th loop skipped (code copied from the 2048 version where this is needed) 
	

    //Unrolled 6th? loop(s?)
	//64 numbers left to go
    if (tid < 32)
    {
		//64 --> 32
        sdata[tid] += sdata[tid + 32];//e.g. [0] = [0] + [32]; [31] = [31] + [63]
		//32 --> 16
        sdata[tid] += sdata[tid + 16];//e.g. [0] = [0] + [16]; [15] = [15] + [31]
		//16 --> 8
        sdata[tid] += sdata[tid +  8];//e.g. [0] = [0] + [8]
		//8 --> 4
        sdata[tid] += sdata[tid +  4];
		//4 --> 2
        sdata[tid] += sdata[tid +  2];     
		//2 --> 1
        sdata[tid] += sdata[tid +  1];
	}
	
	//save result
    if (tid == 0) 
    {
        result[get_group_id(0)] = sdata[0];
    }
	
}

//adds all values in the input
//Assumes a workgroup size of 32. Each workgroup processes 64 values and puts the result into 'result'
kernel void ReduceSum_d_64(global double* g_idata, global double* result, int tidyIfGroup, int tidyIfThreadLessThan)
{
	__local double sdata[32]; 

    const int workgroupSize = 32;//get_local_size(0);//no items in this workgroup
    
	


    //int itemsPerWorkgroup = nItemsPerWorkgroup;//n / workgroupSize;

    int tid = get_local_id(0);//Work-item (Thread) ID
    
    
    
  

    
    //Unrolled first loop - read from global memory, and add every second, saving to local memory
    //Pretend there are 16 numbers and two work groups each with a size of FOUR. (in reality, it's 64 and 32) .This adds numbers in like so:
    //a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 d0 d1 d2 d3
    //Workgroup 0
    //localArray={a0 + b0, a1 + b1, a2 + b2, a3 + b3}
    //Workgroup 1
    //localArray={c0 + d0, c1 + d1, c2 + d2, c3 + d3}
	{
		int groupId=get_group_id(0);
		int i = groupId*64 + tid;
		double initRes=g_idata[i]+g_idata[i+32];

		if(tidyIfGroup == groupId && tid < tidyIfThreadLessThan)
		{
			//There is a partial block after the end which has no assigned workgroup and 
			//so this workgroup must clean up

			int tidyIfThreadLessThan_2 = tidyIfThreadLessThan-32;
			if(tid < tidyIfThreadLessThan_2)
			{
				//Include a third and fourth value
				initRes += g_idata[i+64]+g_idata[i+64+32];
			}
			else
			{
				//Include a third value
				initRes += g_idata[i+64];  	
			}
		}
		sdata[tid] = initRes;
	}
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads

    //2nd loop skipped (code copied from the 2048 version where this is needed) 
    //3rd loop skipped (code copied from the 2048 version where this is needed) 
	//4th loop skipped (code copied from the 2048 version where this is needed) 
	//5th loop skipped (code copied from the 2048 version where this is needed) 

    //Unrolled 6th? loop(s?)
	//32 numbers left to go
    if (tid < 32)
    {
		//32 --> 16
        sdata[tid] += sdata[tid + 16];//e.g. [0] = [0] + [16]; [15] = [15] + [31]
		//16 --> 8
        sdata[tid] += sdata[tid +  8];//e.g. [0] = [0] + [8]
		//8 --> 4
        sdata[tid] += sdata[tid +  4];
		//4 --> 2
        sdata[tid] += sdata[tid +  2];     
		//2 --> 1
        sdata[tid] += sdata[tid +  1];
	}
	
	//save result
    if (tid == 0) 
    {
        result[get_group_id(0)] = sdata[0];
    }
	
}


//************************
//SELF ORGANISING MAP CODE
//************************

kernel void Closest_Min_d_32_1024(__global double* A, __global double* B, __local double* workingSpace, __global double* C, __private int rowOffset, int noRowsA,int noRowsB )// int noRowsBOfInterest)
	{
		//A and B should be supplied column-at-a-time
		//nCols must be 32
		//noRowsB must be >= 1024
		//Designed to only process 1024 rows from B per workgroup
		//Each workgroup processes one row in A and compares with 1024 rows in B
		// -- To specify which row in B the workgroup should start on, provide a rowOffset that is not zero
		//Each thread in that workgroup compares rowA with one row from B (ie nThreads = nRowsB)
		int nCols = 32;
		int myRowA = get_group_id(0);//row number for A
		int tid = get_local_id(0);//thread ID within this workgroup
		int myRowB = tid + rowOffset;//row number for B
		

		//Load the appropriate row from A into workingSpace
		if (tid < nCols)//nCols is the row length; tid for this bit only refers to which element of the row we are retrieving
		{
			workingSpace[tid] = A[tid * noRowsA + myRowA];
		}
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads



		double totalSqDist =0;

		{
			double left;
			double right;
			double diff;
			
		#pragma unroll
			for(int iRow = 0; iRow < 32; iRow++)
			{
				left = workingSpace[iRow];
				right = B[myRowB + iRow * noRowsB];
				diff = left - right;
				totalSqDist += diff * diff;
			}
		}

		//Save the result locally
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure that workingSpace is not still being read from
		workingSpace[tid] = totalSqDist;		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values



									  //Find the minimum
		

		//Condense from 1024 to best 512
		if (tid < 512)// && (tid + 512) > noRowsBOfInterest)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 512]);
		}
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values

		

		//Condense from 512 to best 256
		if (tid < 256)// && (tid + 256) > noRowsBOfInterest)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 256]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values
		
		//Condense from 256 to best 128
		if (tid < 128)// && (tid + 128) > noRowsBOfInterest)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 128]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values

		//Condense from 128 to best 64
		if (tid < 64)// && (tid + 64) > noRowsBOfInterest)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 64]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values

		//Condense from 64 to best 32
		if (tid < 32)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 32]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values

		//Condense from 32 to best 16
		if (tid < 16)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 16]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values

		//Condense from 16 to best 8
		if (tid < 8)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 8]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values

		//Condense from 8 to best 4
		if (tid < 4)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 4]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values


		//Condense from 8 to best 4
		if (tid < 2)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 2]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values


		//condense to best 1
		if (tid == 0)
		{
			//minVal = min(workingSpace[0], workingSpace[1]);
			//Save the result

			C[myRowA] =sqrt(min(workingSpace[0], workingSpace[1]));
			}

	//	if (tid == 1)
	//	{
	//			int iRowo = 0;
	//			double lefto = A[tid * noRowsA + myRowA];
	//			double righto = B[myRowB + iRowo * noRowsB];
	//			double diffo = lefto - righto;
	//			double totalSqDisto = diffo * diffo;
	//			C[myRowA] =righto;
	//	}

		
	}



	kernel void Closest_Min_d_32_2048(__global double* A, __global double* B, __local double* workingSpace, __global double* C, __private int rowOffset, int noRowsA,int noRowsB )// int noRowsBOfInterest)
	{
		//Calculates distances between each row in A and each row in B
		//A and B should be supplied column-at-a-time
		//nCols is dimension, and must be 32
		//noRowsB must be >= 2048
		//Designed to only process 2048 rows from B per workgroup
		//Each workgroup processes one row in A and compares with 2048 rows in B
		// -- To specify which row in B the workgroup should start on, provide a rowOffset that is not zero
		//Each thread in that workgroup compares rowA with two rows from B (ie nThreads = 1024 but 2048 rows are looked at)
		int nCols = 32;
		int myRowA = get_group_id(0);//row number for A
		int tid = get_local_id(0);//thread ID within this workgroup
		int myRowB = tid + rowOffset;//row number for B
		

		//Load the appropriate row from A into workingSpace
		if (tid < nCols)//nCols is the row length; tid for this bit only refers to which element of the row we are retrieving
		{
			workingSpace[tid] = A[tid * noRowsA + myRowA];
		}
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads



		double totalSqDist =0;

		{
			//Calc dist for the first row we look at
			double left;
			double right;
			double diff;
			int myRowB2=myRowB;
		#pragma unroll
			for(int iRow = 0; iRow < 32; iRow++)
			{
				left = workingSpace[iRow];
				right = B[myRowB2];
				diff = left - right;
				totalSqDist += diff * diff;
				myRowB2+=noRowsB;
			}
		}
		{
			//Cal dist for the second row we look at
double totalSqDist2 =0;
			double left;
			double right;
			double diff;
			
int myRowB2 = myRowB + 1024;
		#pragma unroll
			for(int iRow = 0; iRow < 32; iRow++)
			{
				left = workingSpace[iRow];
				right = B[myRowB2];
				diff = left - right;
				totalSqDist2 += diff * diff;
				myRowB2+=noRowsB;
			}

			//Keep only the best result
			if(totalSqDist > totalSqDist2)
			{
			totalSqDist=totalSqDist2;
			}
		}




/*
		{
			double left;
			double right;
			double diff;

			double totalSqDist2 = 0;
#pragma unroll
			for(int i = 0; i < 32; i++)
			{
				left = workingSpace[i+512];
				right = B[myRowB + 512 + i * noRowsB];
				diff = left - right;
				totalSqDist2 += diff * diff;
			}
			
			totalSqDist = min(totalSqDist,totalSqDist2);
		}
*/


		//Save the result locally
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure that workingSpace is not still being read from
		workingSpace[tid] = totalSqDist;		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values



									  //Find the minimum
		

		//Condense from 1024 to best 512
		if (tid < 512)// && (tid + 512) > noRowsBOfInterest)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 512]);
		}
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values

		

		//Condense from 512 to best 256
		if (tid < 256)// && (tid + 256) > noRowsBOfInterest)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 256]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values
		
		//Condense from 256 to best 128
		if (tid < 128)// && (tid + 128) > noRowsBOfInterest)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 128]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values

		//Condense from 128 to best 64
		if (tid < 64)// && (tid + 64) > noRowsBOfInterest)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 64]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values

		//Condense from 64 to best 32
		if (tid < 32)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 32]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values

		//Condense from 32 to best 16
		if (tid < 16)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 16]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values

		//Condense from 16 to best 8
		if (tid < 8)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 8]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values

		//Condense from 8 to best 4
		if (tid < 4)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 4]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values


		//Condense from 8 to best 4
		if (tid < 2)
		{
			workingSpace[tid] = min(workingSpace[tid], workingSpace[tid + 2]);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads to ensure all have written the values


		//condense to best 1
		if (tid == 0)
		{
			//minVal = min(workingSpace[0], workingSpace[1]);
			//Save the result

			C[myRowA] =sqrt(min(workingSpace[0], workingSpace[1]));
			}

	//	if (tid == 1)
	//	{
	//			int iRowo = 0;
	//			double lefto = A[tid * noRowsA + myRowA];
	//			double righto = B[myRowB + iRowo * noRowsB];
	//			double diffo = lefto - righto;
	//			double totalSqDisto = diffo * diffo;
	//			C[myRowA] =righto;
	//	}

		
	}

