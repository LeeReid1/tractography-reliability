inline double matrixMul_HighAccuracy_d_2_Sub(__global double* B, int nColsB, __local double* aRowData, int* nextB, int* k)
{
    //Multiply two entries
    //Multiply another two entries
    //Sum the results and save to the working space
    //We use halfWorkgroupSize because k is going up in increments of 2
    //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
    //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
    double res2=aRowData[(*k)] * B[(*nextB)]; 
    (*nextB)=(*nextB)+nColsB; 
    (*k)=(*k)+1;
    res2 +=aRowData[(*k)] * B[(*nextB)]; 
    (*nextB)=(*nextB)+nColsB; 
    (*k)=(*k)+1;
    
    return res2;
}
inline double matrixMul_HighAccuracy_d_4_Sub(__global double* B, int nColsB, __local double* aRowData, int* nextB, int* k)
{
    double res =matrixMul_HighAccuracy_d_2_Sub( B, nColsB, aRowData, nextB, k);
    return res + matrixMul_HighAccuracy_d_2_Sub( B, nColsB, aRowData, nextB, k);
}    
inline double matrixMul_HighAccuracy_d_8_Sub(__global double* B, int nColsB, __local double* aRowData, int* nextB, int* k)
{
    double res =matrixMul_HighAccuracy_d_4_Sub( B, nColsB, aRowData, nextB, k);
    return res + matrixMul_HighAccuracy_d_4_Sub( B, nColsB, aRowData, nextB, k);
}    
inline double matrixMul_HighAccuracy_d_16_Sub(__global double* B, int nColsB, __local double* aRowData, int* nextB, int* k)
{
    double res =matrixMul_HighAccuracy_d_8_Sub( B, nColsB, aRowData, nextB, k);
    return res + matrixMul_HighAccuracy_d_8_Sub( B, nColsB, aRowData, nextB, k);
}    
inline double matrixMul_HighAccuracy_d_32_Sub(__global double* B, int nColsB, __local double* aRowData, int* nextB, int* k)
{
    double res =matrixMul_HighAccuracy_d_16_Sub( B, nColsB, aRowData, nextB, k);
    return res + matrixMul_HighAccuracy_d_16_Sub( B, nColsB, aRowData, nextB, k);
}    
inline double matrixMul_HighAccuracy_d_64_Sub(__global double* B, int nColsB, __local double* aRowData, int* nextB, int* k)
{
    double res =matrixMul_HighAccuracy_d_32_Sub( B, nColsB, aRowData, nextB, k);
    return res + matrixMul_HighAccuracy_d_32_Sub( B, nColsB, aRowData, nextB, k);
}    
inline double matrixMul_HighAccuracy_d_128_Sub(__global double* B, int nColsB, __local double* aRowData, int* nextB, int* k)
{
    double res =matrixMul_HighAccuracy_d_64_Sub( B, nColsB, aRowData, nextB, k);
    return res + matrixMul_HighAccuracy_d_64_Sub( B, nColsB, aRowData, nextB, k);
}    
inline double matrixMul_HighAccuracy_d_256_Sub(__global double* B, int nColsB, __local double* aRowData, int* nextB, int* k)
{
    double res =matrixMul_HighAccuracy_d_128_Sub( B, nColsB, aRowData, nextB, k);
    return res + matrixMul_HighAccuracy_d_128_Sub( B, nColsB, aRowData, nextB, k);
}    
inline double matrixMul_HighAccuracy_d_512_Sub(__global double* B, int nColsB, __local double* aRowData, int* nextB, int* k)
{
    double res =matrixMul_HighAccuracy_d_256_Sub( B, nColsB, aRowData, nextB, k);
    return res + matrixMul_HighAccuracy_d_256_Sub( B, nColsB, aRowData, nextB, k);
}    
inline double matrixMul_HighAccuracy_d_1024_Sub(__global double* B, int nColsB, __local double* aRowData, int* nextB, int* k)
{
    double res =matrixMul_HighAccuracy_d_512_Sub( B, nColsB, aRowData, nextB, k);
    return res + matrixMul_HighAccuracy_d_512_Sub( B, nColsB, aRowData, nextB, k);
}    
inline double matrixMul_HighAccuracy_d_2048_Sub(__global double* B, int nColsB, __local double* aRowData, int* nextB, int* k)
{
    double res =matrixMul_HighAccuracy_d_1024_Sub( B, nColsB, aRowData, nextB, k);
    return res + matrixMul_HighAccuracy_d_1024_Sub( B, nColsB, aRowData, nextB, k);
}    
inline double matrixMul_HighAccuracy_d_4096_Sub(__global double* B, int nColsB, __local double* aRowData, int* nextB, int* k)
{
    double res =matrixMul_HighAccuracy_d_2048_Sub( B, nColsB, aRowData, nextB, k);
    return res + matrixMul_HighAccuracy_d_2048_Sub( B, nColsB, aRowData, nextB, k);
}    
/*
Suitable for when the left matrix has a very large numbers of columns - has a lower memory footprint than other methods by not loading the full left row
Assumes:
A workgroup size of 1024
nColsA is divisible by 4096
*/
__kernel void matrixMul_HighAccuracy_d_4096_XL(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __local double* toSum, __global double* C)
{
   //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//Work-item (Thread) ID
   
   
   int loops=nColsARowsB/4096;
   int remaining=loops;//remaining to sum
   int workingSpaceStart=tid;
   
   int nextB=bColStartingIndex+tid;
   int* nextBPtr=&nextB;
   int nextBStep=nColsB*4096;
   int nextWriteTo=workingSpaceStart;
   
   int startRead=rowA * nColsARowsB;
   
   
   for (int i_step = 0; i_step < loops; i_step++)
   {
		
		
//		#pragma unroll
//		for(int i_halfBlock=0;i_halfBlock<2;i_halfBlock++)
//		{
	   //Copy data locally so we can access it faster
	   //Each thread copies only a subset of items, splitting the workload across
	   //all threads in this workgroup
	   barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup so we know they have all finish accessing the local array
	   //Load 4096 items in
		int col=i_step*4096+ tid;
		for(int iLoad=tid; iLoad<4096; iLoad+= workgroupSize)
		{
		   aRowData[iLoad]=A[startRead  + col];//A: rowA, column readFromCol
		   col += workgroupSize;
		}
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup so we know they have all finished writigint to the local array
		//if(i_step==1)
		//{
		//	C[rowA * nColsB + bColStartingIndex + tid]=aRowData[4000+tid];//aRowData[tid];
		//	return;
		//}
		/*
   if(i_step > 0)
	{
		C[rowA * nColsB + bColStartingIndex + tid]=7;
	return;
	}*/
   
		//Run mult+add on the currently loaded 4096 items
		
		
		int k=0;
		//int* kptr = &k;
		//(*kptr)=(*kptr)+1;
		//(*nextBPtr)=(*nextBPtr)+nColsB; 
		//C[rowA * nColsB + bColStartingIndex + tid]=(*nextBPtr);
		
		double res=matrixMul_HighAccuracy_d_4096_Sub(B, nColsB, aRowData, nextBPtr, &k);
		
		
	
//		nextB+=nextBStep;
//		}
    
   
		
	   
       toSum[nextWriteTo] =res;
	   nextWriteTo+=workgroupSize;
   }
   
  //C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];
   //return;
	
 //if(true)
 {
   

   //We have done our pointwise multiplication
   //If we started with 'n' cols in A, we now have n/4096 items left to sum
  int doublWGSize=workgroupSize*2;
   while(remaining > 1)
   {

        //Add every second number
		int halfRem=remaining/2;
		
		int readLeft = workingSpaceStart;
		int writeTo =workingSpaceStart;
		
		
		
		for (int k = 0; k < halfRem; k++)
		{
            toSum[writeTo] =toSum[readLeft] +toSum[readLeft + workgroupSize];
			
			writeTo += workgroupSize;
			readLeft += doublWGSize;
		}
		
		if(remaining % 2 == 1)
		{
			//does not divide into even number
			//Add the last item to first element
            toSum[workingSpaceStart] += toSum[readLeft];
		}
	   

	   remaining=halfRem;
   }

 }
   // Write the matrix to device memory each 
   // thread writes one element
   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//A[ty * nColsARowsB];//value;
 
}
 


/*
Calculates a matrix multiplication. The left matrix number of cols and right matrix number of rows must be exactly 4096
*/
__kernel void matrixMul_HighAccuracy_d_4096_E(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __global double* C)
{  //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//workingSpaceStart = Work-item (Thread) ID 
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
   {
	   int startRead=rowA * nColsARowsB;
	   
	   for(int col = tid; col < nColsARowsB; col += workgroupSize)
	   {
		   aRowData[col]=A[startRead  + col];//A: rowA, column readFromCol
	   }
   }
 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup

   //High accurate summation (better accuracy when many many small items to sum because of floating point stuff)
  
   
   int nextB=bColStartingIndex+tid;
   
		//double result=matrixMul_HighAccuracy_d_4096_E_Sub( A,   B,  nColsARowsB,  nColsB, bColStartingIndex, aRowData,  workingSpaceStart,  nextB,0);
		double res0;
		double res_m;
		double res_n;
		double res_o;
		double res_p;
		double res_q;
		double res_r;
		double res_s;
		double res_t;
		double res_u;
		double res_v;
		double res_w;
   
       //Multiply two entries
       //Multiply another two entries
       //Sum the results and save to the working space
	   //We use halfWorkgroupSize because k is going up in increments of 2
	   //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
	   //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
	   int k=0;
		res_w=0;
		#pragma unroll
		for(int w=0;w<2;w++)
		{	   
			res_v=0;
			#pragma unroll
			for(int v=0;v<2;v++)
			{
				res_u=0;
				#pragma unroll
				for(int u=0;u<2;u++)
				{
					res_t=0;
					#pragma unroll
					for(int t=0;t<2;t++)
					{	   		
						res_s=0;
						#pragma unroll
						for(int s=0;s<2;s++)
						{	   		
							res_r=0;
							#pragma unroll
							for(int r=0;r<2;r++)
							{	   		
								res_q=0;
								#pragma unroll
								for(int q=0;q<2;q++)
								{	
									res_p=0;			
									#pragma unroll
									for(int p=0;p<2;p++)
									{	   
									res_o=0;
									#pragma unroll
									for(int o=0;o<2;o++)
									{
						
									   res_n=0;
									   #pragma unroll
									   for(int n=0;n<2;n++)
									   {
										   res_m=0;
											#pragma unroll
										   for(int m=0;m<2;m++)
										   {
											   res0 =aRowData[k] * B[nextB];
											   nextB+=nColsB; 
											   k++;
											   
											   res0 +=aRowData[k] * B[nextB];
											   nextB+=nColsB; 
											   k++;
											   
												res_m+=res0;
										   }
										   res_n+=res_m;
										}
										res_o+=res_n;
									}
									res_p+=res_o;
								   }
								   res_q+=res_p;
							   }
							   res_r+=res_q;
							}
							res_s+=res_r;
						}
						res_t+=res_s;
					}
					res_u+=res_t;
				}
				res_v+=res_u;
			}
			res_w+=res_v;
		}

        C[rowA * nColsB + bColStartingIndex + tid]= res_w;
	// C[rowA * nColsB + bColStartingIndex + tid]=matrixMul_HighAccuracy_d_4096_E_Sub( A,   B,  nColsARowsB,  nColsB, bColStartingIndex, aRowData,  workingSpaceStart,  nextB,0);
}


__kernel void matrixMul_HighAccuracy_d_1024(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __local double* toSum, __global double* C)
{
   //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//Work-item (Thread) ID
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
   {
	   int startRead=rowA * nColsARowsB;
	   
	   for(int col = tid; col < nColsARowsB; col += workgroupSize)
	   {
		   aRowData[col]=A[startRead  + col];//A: rowA, column readFromCol
	   }
   }
 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup

   //Naive summation (low accuracy when many many small items to sum because of floating point stuff)
   /*
   double value = 0;
   int nextB=bColStartingIndex+tid;
   for (int k = 0; k < nColsARowsB; k++)
   {
      value += aRowData[k] * B[nextB];// B[k * nColsB + tid];//B: row k, column tid
	  nextB+=nColsB;
   }
   C[rowA * nColsB + bColStartingIndex + tid]=value;
   return;
   */
   
 
   //High accurate summation (better accuracy when many many small items to sum because of floating point stuff)
   int remaining=nColsARowsB/1024;//remaining to sum
   int workingSpaceStart=tid;
   
   int nextB=bColStartingIndex+tid;
   int nextWriteTo=workingSpaceStart;
   for (int k = 0; k < nColsARowsB; )
   {
       //Multiply two entries
       //Multiply another two entries
       //Sum the results and save to the working space
	   //We use halfWorkgroupSize because k is going up in increments of 2
	   //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
	   //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
	   
	   double res0;
		double res_m;
		double res_n;
		double res_o;
		double res_p;
		double res_q;
		double res_r;
		double res_s;
		double res_t;
		double res_u;
		res_u=0;
		#pragma unroll
		for(int u=0;u<2;u++)
		{
		res_t=0;
		#pragma unroll
		for(int t=0;t<2;t++)
		{	   		
		res_s=0;
		#pragma unroll
		for(int s=0;s<2;s++)
		{	   		
		res_r=0;
		#pragma unroll
		for(int r=0;r<2;r++)
		{	   		
			res_q=0;
			#pragma unroll
			for(int q=0;q<2;q++)
			{	
				res_p=0;			
				#pragma unroll
				for(int p=0;p<2;p++)
				{	   
				res_o=0;
				#pragma unroll
				for(int o=0;o<2;o++)
				{
	
				   res_n=0;
				   #pragma unroll
				   for(int n=0;n<2;n++)
				   {
					   res_m=0;
						#pragma unroll
					   for(int m=0;m<2;m++)
					   {
						   res0 =aRowData[k] * B[nextB];
						   nextB+=nColsB; 
						   k++;
						   
						   res0 +=aRowData[k] * B[nextB];
						   nextB+=nColsB; 
						   k++;
						   
							res_m+=res0;
					   }
					   res_n+=res_m;
					}
					res_o+=res_n;
				}
				res_p+=res_o;
			   }
			   res_q+=res_p;
		   }
		   res_r+=res_q;
		}
		res_s+=res_r;
		}
		res_t+=res_s;
		}
		res_u+=res_t;
		}
	   //Add the sums
       toSum[nextWriteTo] =res_u;// + (aRowData[k+1] * B[nextB+ nColsB]);//B: row k, column tid.
	   nextWriteTo+=workgroupSize;
   }
	
 //if(true)
 {
   

   //We have done our pointwise multiplication
   //If we started with 'n' cols in A, we now have n/512 items left to sum
  int doublWGSize=workgroupSize*2;
   while(remaining > 1)
   {

        //Add every second number
		int halfRem=remaining/2;
		
		int readLeft = workingSpaceStart;
		int writeTo =workingSpaceStart;
		
		
		
		for (int k = 0; k < halfRem; k++)
		{
            toSum[writeTo] =toSum[readLeft] +toSum[readLeft + workgroupSize];
			
			writeTo += workgroupSize;
			readLeft += doublWGSize;
		}
		
		if(remaining % 2 == 1)
		{
			//does not divide into even number
			//Add the last item to first element
            toSum[workingSpaceStart] += toSum[readLeft];
	   }
	   

	   remaining=halfRem;
   }

 }
   // Write the matrix to device memory each 
   // thread writes one element
   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//A[ty * nColsARowsB];//value;
 
}
 
 
__kernel void matrixMul_HighAccuracy_d_512(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __local double* toSum, __global double* C)
{
   //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//Work-item (Thread) ID
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
   {
	   int startRead=rowA * nColsARowsB;
	   
	   for(int col = tid; col < nColsARowsB; col += workgroupSize)
	   {
		   aRowData[col]=A[startRead  + col];//A: rowA, column readFromCol
	   }
   }
 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup

   //Naive summation (low accuracy when many many small items to sum because of floating point stuff)
   /*
   double value = 0;
   int nextB=bColStartingIndex+tid;
   for (int k = 0; k < nColsARowsB; k++)
   {
      value += aRowData[k] * B[nextB];// B[k * nColsB + tid];//B: row k, column tid
	  nextB+=nColsB;
   }
   C[rowA * nColsB + bColStartingIndex + tid]=value;
   return;
   */
   
 
   //High accurate summation (better accuracy when many many small items to sum because of floating point stuff)
   int remaining=nColsARowsB/512;//remaining to sum
   int workingSpaceStart=tid;
   
   int nextB=bColStartingIndex+tid;
   int nextWriteTo=workingSpaceStart;
   for (int k = 0; k < nColsARowsB; )
   {
       //Multiply two entries
       //Multiply another two entries
       //Sum the results and save to the working space
	   //We use halfWorkgroupSize because k is going up in increments of 2
	   //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
	   //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
	   
	   double res0;
		double res_m;
		double res_n;
		double res_o;
		double res_p;
		double res_q;
		double res_r;
		double res_s;
		double res_t;
		res_t=0;
		#pragma unroll
		for(int t=0;t<2;t++)
		{	   		
		res_s=0;
		#pragma unroll
		for(int s=0;s<2;s++)
		{	   		
		res_r=0;
		#pragma unroll
		for(int r=0;r<2;r++)
		{	   		
			res_q=0;
			#pragma unroll
			for(int q=0;q<2;q++)
			{	
				res_p=0;			
				#pragma unroll
				for(int p=0;p<2;p++)
				{	   
				res_o=0;
				#pragma unroll
				for(int o=0;o<2;o++)
				{
	
				   res_n=0;
				   #pragma unroll
				   for(int n=0;n<2;n++)
				   {
					   res_m=0;
						#pragma unroll
					   for(int m=0;m<2;m++)
					   {
						   res0 =aRowData[k] * B[nextB];
						   nextB+=nColsB; 
						   k++;
						   
						   res0 +=aRowData[k] * B[nextB];
						   nextB+=nColsB; 
						   k++;
						   
							res_m+=res0;
					   }
					   res_n+=res_m;
					}
					res_o+=res_n;
				}
				res_p+=res_o;
			   }
			   res_q+=res_p;
		   }
		   res_r+=res_q;
		}
		res_s+=res_r;
		}
		res_t+=res_s;
		}
	   //Add the sums
       toSum[nextWriteTo] =res_t;// + (aRowData[k+1] * B[nextB+ nColsB]);//B: row k, column tid.
	   nextWriteTo+=workgroupSize;
   }
	
 //if(true)
 {
   

   //We have done our pointwise multiplication
   //If we started with 'n' cols in A, we now have n/512 items left to sum
  int doublWGSize=workgroupSize*2;
   while(remaining > 1)
   {

        //Add every second number
		int halfRem=remaining/2;
		
		int readLeft = workingSpaceStart;
		int writeTo =workingSpaceStart;
		
		
		
		for (int k = 0; k < halfRem; k++)
		{
            toSum[writeTo] =toSum[readLeft] +toSum[readLeft + workgroupSize];
			
			writeTo += workgroupSize;
			readLeft += doublWGSize;
		}
		
		if(remaining % 2 == 1)
		{
			//does not divide into even number
			//Add the last item to first element
            toSum[workingSpaceStart] += toSum[readLeft];
	   }
	   

	   remaining=halfRem;
   }

 }
   // Write the matrix to device memory each 
   // thread writes one element
   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//A[ty * nColsARowsB];//value;
 
}
 
 

__kernel void matrixMul_HighAccuracy_d_256(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __local double* toSum, __global double* C)
{

   
   //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//Work-item (Thread) ID
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
   {
	   int startRead=rowA * nColsARowsB;
	   
	   for(int col = tid; col < nColsARowsB; col += workgroupSize)
	   {
		   aRowData[col]=A[startRead  + col];//A: rowA, column readFromCol
	   }
   }
 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup

	
	//return;
	
 
   //Naive summation (low accuracy when many many small items to sum because of floating point stuff)
   /*
   double value = 0;
   int nextB=bColStartingIndex+tid;
   for (int k = 0; k < nColsARowsB; k++)
   {
      value += aRowData[k] * B[nextB];// B[k * nColsB + tid];//B: row k, column tid
	  nextB+=nColsB;
   }
   C[rowA * nColsB + bColStartingIndex + tid]=value;
   return;
   */
   
   

    //double values[nColsARowsB];

   //High accurate summation (better accuracy when many many small items to sum because of floating point stuff)
   int remaining=nColsARowsB/256;//remaining to sum
 //  C[get_global_id(0)] =nColsARowsB;
 //  return;
   int workingSpaceStart=tid;
   
  // int halfWorkgroupSize=workgroupSize/2;
   int nextB=bColStartingIndex+tid;
   int nextWriteTo=workingSpaceStart;
   for (int k = 0; k < nColsARowsB; )
   {
       //Multiply two entries
       //Multiply another two entries
       //Sum the results and save to the working space
	   //We use halfWorkgroupSize because k is going up in increments of 2
	   //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
	   //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
	   
	   double res0;
		double res_m;
		double res_n;
		double res_o;
		double res_p;
		double res_q;
		double res_r;
		double res_s;
		
		res_s=0;
		#pragma unroll
		for(int s=0;s<2;s++)
		{	   		
		res_r=0;
		#pragma unroll
		for(int r=0;r<2;r++)
		{	   		
			res_q=0;
			#pragma unroll
			for(int q=0;q<2;q++)
			{	
				res_p=0;			
				#pragma unroll
				for(int p=0;p<2;p++)
				{	   
				res_o=0;
				#pragma unroll
				for(int o=0;o<2;o++)
				{
	
				   res_n=0;
				   #pragma unroll
				   for(int n=0;n<2;n++)
				   {
					   res_m=0;
						#pragma unroll
					   for(int m=0;m<2;m++)
					   {
						   res0 =aRowData[k] * B[nextB];
						   nextB+=nColsB; 
						   k++;
						   
						   res0 +=aRowData[k] * B[nextB];
						   nextB+=nColsB; 
						   k++;
						   
							res_m+=res0;
					   }
					   res_n+=res_m;
					}
					res_o+=res_n;
				}
				res_p+=res_o;
			   }
			   res_q+=res_p;
		   }
		   res_r+=res_q;
		}
		res_s+=res_r;
		}
	   //Add the sums
       toSum[nextWriteTo] =res_s;// + (aRowData[k+1] * B[nextB+ nColsB]);//B: row k, column tid.
	   nextWriteTo+=workgroupSize;
   }
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[2] * B[bColStartingIndex+tid +nColsB + nColsB];//[2]
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[3] * B[bColStartingIndex+tid +nColsB + nColsB+ nColsB];//[3]
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+workgroupSize];//[2]+[3]
		//return;
//   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+k*halfWorkgroupSize];
   //return;
	
 //if(true)
 {
   

   //We have done our pointwise multiplication
   //If we started with 'n' cols in A, we now have n/2 items left to sum
   //C[get_global_id(0)]=rowA * nColsARowsB + tid;
   //C[rowA * nColsARowsB + tid]=13;//1000*get_global_id(0);
  // double missedBits=0;
  int doublWGSize=workgroupSize*2;
   while(remaining > 1)
   {

        //Add every second number
        //for (int k = 1; k < remaining; k++)
		int halfRem=remaining/2;
		
		int readLeft = workingSpaceStart;
		int writeTo =workingSpaceStart;
		
		
		
		for (int k = 0; k < halfRem; k++)
		{
            //int offset=k*workgroupSize;
			
			
			//int readRight = readLeft + workgroupSize;
            toSum[writeTo] =toSum[readLeft] +toSum[readLeft + workgroupSize];
			
			writeTo += workgroupSize;
			readLeft += doublWGSize;
		}
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//[0]+[1]+[2]+[3]
		//return;
		
		if(remaining % 2 == 1)
		{
			//does not divide into even number
			//Add the last item to first element
            toSum[workingSpaceStart] += toSum[readLeft];
	   }
	   

	   remaining=halfRem;
   }
  // toSum[workingSpaceStart] += missedBits;
 }
   // Write the matrix to device memory each 
   // thread writes one element
   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//A[ty * nColsARowsB];//value;
 
}
 
 


__kernel void matrixMul_HighAccuracy_d_128(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __local double* toSum, __global double* C)
{

      //int tx = get_global_id(0); //column number for B
   //int ty = get_global_id(0);//row number for A
   
   
   
   //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//Work-item (Thread) ID
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
   {
	   int startRead=rowA * nColsARowsB;
	   
	   for(int col = tid; col < nColsARowsB; col += workgroupSize)
	   {
		   aRowData[col]=A[startRead  + col];//A: rowA, column readFromCol
	   }
   }
 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup

	
	//return;
	
 
   //Naive summation (low accuracy when many many small items to sum because of floating point stuff)
   /*
   double value = 0;
   int nextB=bColStartingIndex+tid;
   for (int k = 0; k < nColsARowsB; k++)
   {
      value += aRowData[k] * B[nextB];// B[k * nColsB + tid];//B: row k, column tid
	  nextB+=nColsB;
   }
   C[rowA * nColsB + bColStartingIndex + tid]=value;
   return;
   */
   
   

    //double values[nColsARowsB];

   //High accurate summation (better accuracy when many many small items to sum because of floating point stuff)
   int remaining=nColsARowsB/128;//remaining to sum
 //  C[get_global_id(0)] =nColsARowsB;
 //  return;
   int workingSpaceStart=tid;
   
  // int halfWorkgroupSize=workgroupSize/2;
   int nextB=bColStartingIndex+tid;
   int nextWriteTo=workingSpaceStart;
   for (int k = 0; k < nColsARowsB; )
   {
       //Multiply two entries
       //Multiply another two entries
       //Sum the results and save to the working space
	   //We use halfWorkgroupSize because k is going up in increments of 2
	   //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
	   //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
	   
	   double res0;
		double res_m;
		double res_n;
		double res_o;
		double res_p;
		double res_q;
		double res_r;
		res_r=0;
		#pragma unroll
		for(int r=0;r<2;r++)
		{	   		
			res_q=0;
			#pragma unroll
			for(int q=0;q<2;q++)
			{	
				res_p=0;			
				#pragma unroll
				for(int p=0;p<2;p++)
				{	   
				res_o=0;
				#pragma unroll
				for(int o=0;o<2;o++)
				{
	
				   res_n=0;
				   #pragma unroll
				   for(int n=0;n<2;n++)
				   {
					   res_m=0;
						#pragma unroll
					   for(int m=0;m<2;m++)
					   {
						   res0 =aRowData[k] * B[nextB];
						   nextB+=nColsB; 
						   k++;
						   
						   res0 +=aRowData[k] * B[nextB];
						   nextB+=nColsB; 
						   k++;
						   
							res_m+=res0;
					   }
					   res_n+=res_m;
					}
					res_o+=res_n;
				}
				res_p+=res_o;
			   }
			   res_q+=res_p;
		   }
		   res_r+=res_q;
		}
	   //Add the sums
       toSum[nextWriteTo] =res_r;// + (aRowData[k+1] * B[nextB+ nColsB]);//B: row k, column tid.
	   nextWriteTo+=workgroupSize;
   }
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[2] * B[bColStartingIndex+tid +nColsB + nColsB];//[2]
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[3] * B[bColStartingIndex+tid +nColsB + nColsB+ nColsB];//[3]
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+workgroupSize];//[2]+[3]
		//return;
//   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+k*halfWorkgroupSize];
   //return;
	
 //if(true)
 {
   

   //We have done our pointwise multiplication
   //If we started with 'n' cols in A, we now have n/2 items left to sum
   //C[get_global_id(0)]=rowA * nColsARowsB + tid;
   //C[rowA * nColsARowsB + tid]=13;//1000*get_global_id(0);
  // double missedBits=0;
  int doublWGSize=workgroupSize*2;
   while(remaining > 1)
   {

        //Add every second number
        //for (int k = 1; k < remaining; k++)
		int halfRem=remaining/2;
		
		int readLeft = workingSpaceStart;
		int writeTo =workingSpaceStart;
		
		
		
		for (int k = 0; k < halfRem; k++)
		{
            //int offset=k*workgroupSize;
			
			
			//int readRight = readLeft + workgroupSize;
            toSum[writeTo] =toSum[readLeft] +toSum[readLeft + workgroupSize];
			
			writeTo += workgroupSize;
			readLeft += doublWGSize;
		}
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//[0]+[1]+[2]+[3]
		//return;
		
		if(remaining % 2 == 1)
		{
			//does not divide into even number
			//Add the last item to first element
            toSum[workingSpaceStart] += toSum[readLeft];
	   }
	   

	   remaining=halfRem;
   }
  // toSum[workingSpaceStart] += missedBits;
 }
   // Write the matrix to device memory each 
   // thread writes one element
   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//A[ty * nColsARowsB];//value;
 
}
 


__kernel void matrixMul_HighAccuracy_d_64(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __local double* toSum, __global double* C)
{

      //int tx = get_global_id(0); //column number for B
   //int ty = get_global_id(0);//row number for A
   
   
   
   //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//Work-item (Thread) ID
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
   {
	   int startRead=rowA * nColsARowsB;
	   
	   for(int col = tid; col < nColsARowsB; col += workgroupSize)
	   {
		   aRowData[col]=A[startRead  + col];//A: rowA, column readFromCol
	   }
   }
 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup

	
	//return;
	
 
   //Naive summation (low accuracy when many many small items to sum because of floating point stuff)
   /*
   double value = 0;
   int nextB=bColStartingIndex+tid;
   for (int k = 0; k < nColsARowsB; k++)
   {
      value += aRowData[k] * B[nextB];// B[k * nColsB + tid];//B: row k, column tid
	  nextB+=nColsB;
   }
   C[rowA * nColsB + bColStartingIndex + tid]=value;
   return;
   */
   
   

    //double values[nColsARowsB];

   //High accurate summation (better accuracy when many many small items to sum because of floating point stuff)
   int remaining=nColsARowsB/64;//remaining to sum
 //  C[get_global_id(0)] =nColsARowsB;
 //  return;
   int workingSpaceStart=tid;
   
  // int halfWorkgroupSize=workgroupSize/2;
   int nextB=bColStartingIndex+tid;
   int nextWriteTo=workingSpaceStart;
   for (int k = 0; k < nColsARowsB; )
   {
       //Multiply two entries
       //Multiply another two entries
       //Sum the results and save to the working space
	   //We use halfWorkgroupSize because k is going up in increments of 2
	   //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
	   //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
	   
	   double res0;
		double res_m;
		double res_n;
		double res_o;
		double res_p;
		double res_q;
		

		res_q=0;
		#pragma unroll
		for(int q=0;q<2;q++)
		{	   

			res_p=0;
			#pragma unroll
			for(int p=0;p<2;p++)
			{	   

			res_o=0;
			#pragma unroll
			for(int o=0;o<2;o++)
			{

			   res_n=0;
			   #pragma unroll
			   for(int n=0;n<2;n++)
			   {
				   res_m=0;
					   
#pragma unroll
				   for(int m=0;m<2;m++)
				   {
					   res0 =aRowData[k] * B[nextB];
					   nextB+=nColsB; 
					   k++;
					   
					   res0 +=aRowData[k] * B[nextB];
					   nextB+=nColsB; 
					   k++;
					   
						res_m+=res0;
				   }
				   res_n+=res_m;
				}
				res_o+=res_n;
			}
			res_p+=res_o;
		   }
		   res_q+=res_p;
	   }
	   //Add the sums
       toSum[nextWriteTo] =res_q;// + (aRowData[k+1] * B[nextB+ nColsB]);//B: row k, column tid.
	   nextWriteTo+=workgroupSize;
   }
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[2] * B[bColStartingIndex+tid +nColsB + nColsB];//[2]
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[3] * B[bColStartingIndex+tid +nColsB + nColsB+ nColsB];//[3]
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+workgroupSize];//[2]+[3]
		//return;
//   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+k*halfWorkgroupSize];
   //return;
	
 //if(true)
 {
   

   //We have done our pointwise multiplication
   //If we started with 'n' cols in A, we now have n/2 items left to sum
   //C[get_global_id(0)]=rowA * nColsARowsB + tid;
   //C[rowA * nColsARowsB + tid]=13;//1000*get_global_id(0);
  // double missedBits=0;
  int doublWGSize=workgroupSize*2;
   while(remaining > 1)
   {

        //Add every second number
        //for (int k = 1; k < remaining; k++)
		int halfRem=remaining/2;
		
		int readLeft = workingSpaceStart;
		int writeTo =workingSpaceStart;
		
		
		
		for (int k = 0; k < halfRem; k++)
		{
            //int offset=k*workgroupSize;
			
			
			//int readRight = readLeft + workgroupSize;
            toSum[writeTo] =toSum[readLeft] +toSum[readLeft + workgroupSize];
			
			writeTo += workgroupSize;
			readLeft += doublWGSize;
		}
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//[0]+[1]+[2]+[3]
		//return;
		
		if(remaining % 2 == 1)
		{
			//does not divide into even number
			//Add the last item to first element
            toSum[workingSpaceStart] += toSum[readLeft];
	   }
	   

	   remaining=halfRem;
   }
  // toSum[workingSpaceStart] += missedBits;
 }
   // Write the matrix to device memory each 
   // thread writes one element
   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//A[ty * nColsARowsB];//value;
 
}
 



__kernel void matrixMul_HighAccuracy_d_32(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __local double* toSum, __global double* C)
{

      //int tx = get_global_id(0); //column number for B
   //int ty = get_global_id(0);//row number for A
   
   
   
   //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//Work-item (Thread) ID
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
   {
	   int startRead=rowA * nColsARowsB;
	   
	   for(int col = tid; col < nColsARowsB; col += workgroupSize)
	   {
		   aRowData[col]=A[startRead  + col];//A: rowA, column readFromCol
	   }
   }
 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup

	
	//return;
	
 
   //Naive summation (low accuracy when many many small items to sum because of floating point stuff)
   /*
   double value = 0;
   int nextB=bColStartingIndex+tid;
   for (int k = 0; k < nColsARowsB; k++)
   {
      value += aRowData[k] * B[nextB];// B[k * nColsB + tid];//B: row k, column tid
	  nextB+=nColsB;
   }
   C[rowA * nColsB + bColStartingIndex + tid]=value;
   return;
   */
   
   

    //double values[nColsARowsB];

   //High accurate summation (better accuracy when many many small items to sum because of floating point stuff)
   int remaining=nColsARowsB/32;//remaining to sum
 //  C[get_global_id(0)] =nColsARowsB;
 //  return;
   int workingSpaceStart=tid;
   
  // int halfWorkgroupSize=workgroupSize/2;
   int nextB=bColStartingIndex+tid;
   int nextWriteTo=workingSpaceStart;
   for (int k = 0; k < nColsARowsB; )
   {
       //Multiply two entries
       //Multiply another two entries
       //Sum the results and save to the working space
	   //We use halfWorkgroupSize because k is going up in increments of 2
	   //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
	   //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
	   
		double res_m;
		double res_n;
		double res_o;
		double res_p;
		double res0;

		res_p=0;
		#pragma unroll
		for(int p=0;p<2;p++)
		{	   

		res_o=0;
		#pragma unroll
		for(int o=0;o<2;o++)
		{

		   res_n=0;
		   #pragma unroll
		   for(int n=0;n<2;n++)
		   {
			   res_m=0;
				   
#pragma unroll
			   for(int m=0;m<2;m++)
			   {
				   res0 =aRowData[k] * B[nextB];
				   nextB+=nColsB; 
				   k++;
				   
				   res0 +=aRowData[k] * B[nextB];
				   nextB+=nColsB; 
				   k++;
				   
					res_m+=res0;
			   }
			   res_n+=res_m;
			}
			res_o+=res_n;
		}
		res_p+=res_o;
	   }
	   
	   //Add the sums
       toSum[nextWriteTo] =res_p;// + (aRowData[k+1] * B[nextB+ nColsB]);//B: row k, column tid.
	   nextWriteTo+=workgroupSize;
   }
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[2] * B[bColStartingIndex+tid +nColsB + nColsB];//[2]
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[3] * B[bColStartingIndex+tid +nColsB + nColsB+ nColsB];//[3]
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+workgroupSize];//[2]+[3]
		//return;
//   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+k*halfWorkgroupSize];
   //return;
	
 //if(true)
 {
   

   //We have done our pointwise multiplication
   //If we started with 'n' cols in A, we now have n/2 items left to sum
   //C[get_global_id(0)]=rowA * nColsARowsB + tid;
   //C[rowA * nColsARowsB + tid]=13;//1000*get_global_id(0);
  // double missedBits=0;
  int doublWGSize=workgroupSize*2;
   while(remaining > 1)
   {

        //Add every second number
        //for (int k = 1; k < remaining; k++)
		int halfRem=remaining/2;
		
		int readLeft = workingSpaceStart;
		int writeTo =workingSpaceStart;
		
		
		
		for (int k = 0; k < halfRem; k++)
		{
            //int offset=k*workgroupSize;
			
			
			//int readRight = readLeft + workgroupSize;
            toSum[writeTo] =toSum[readLeft] +toSum[readLeft + workgroupSize];
			
			writeTo += workgroupSize;
			readLeft += doublWGSize;
		}
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//[0]+[1]+[2]+[3]
		//return;
		
		if(remaining % 2 == 1)
		{
			//does not divide into even number
			//Add the last item to first element
            toSum[workingSpaceStart] += toSum[readLeft];
	   }
	   

	   remaining=halfRem;
   }
  // toSum[workingSpaceStart] += missedBits;
 }
   // Write the matrix to device memory each 
   // thread writes one element
   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//A[ty * nColsARowsB];//value;
 
}
 



// OpenCL Kernel
__kernel void matrixMul_HighAccuracy_d_16(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __local double* toSum, __global double* C)
{

      //int tx = get_global_id(0); //column number for B
   //int ty = get_global_id(0);//row number for A
   
   
   
   //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//Work-item (Thread) ID
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
   {
	   int startRead=rowA * nColsARowsB;
	   
	   for(int col = tid; col < nColsARowsB; col += workgroupSize)
	   {
		   aRowData[col]=A[startRead  + col];//A: rowA, column readFromCol
	   }
   }
 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup

	
	//return;
	
 
   //Naive summation (low accuracy when many many small items to sum because of floating point stuff)
   /*
   double value = 0;
   int nextB=bColStartingIndex+tid;
   for (int k = 0; k < nColsARowsB; k++)
   {
      value += aRowData[k] * B[nextB];// B[k * nColsB + tid];//B: row k, column tid
	  nextB+=nColsB;
   }
   C[rowA * nColsB + bColStartingIndex + tid]=value;
   return;
   */
   
   

    //double values[nColsARowsB];

   //High accurate summation (better accuracy when many many small items to sum because of floating point stuff)
   int remaining=nColsARowsB/16;//remaining to sum
 //  C[get_global_id(0)] =nColsARowsB;
 //  return;
   int workingSpaceStart=tid;
   
  // int halfWorkgroupSize=workgroupSize/2;
   int nextB=bColStartingIndex+tid;
   int nextWriteTo=workingSpaceStart;
   for (int k = 0; k < nColsARowsB; )
   {
       //Multiply two entries
       //Multiply another two entries
       //Sum the results and save to the working space
	   //We use halfWorkgroupSize because k is going up in increments of 2
	   //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
	   //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
	   

	   double res_o=0;
	   #pragma unroll
	   for(int o=0;o<2;o++)
	   {

		   double res_n=0;
		   #pragma unroll
		   for(int n=0;n<2;n++)
		   {
			   double res_m=0;
			   double res0;
#pragma unroll
			   for(int m=0;m<2;m++)
			   {
				   res0 =aRowData[k] * B[nextB];
				   nextB+=nColsB; 
				   k++;
				   
				   res0 +=aRowData[k] * B[nextB];
				   nextB+=nColsB; 
				   k++;
				   
					res_m+=res0;
			   }
			   res_n+=res_m;
		   }
		res_o+=res_n;
	   }
	   
	   
	   //Add the sums
       toSum[nextWriteTo] =res_o;// + (aRowData[k+1] * B[nextB+ nColsB]);//B: row k, column tid.
	   nextWriteTo+=workgroupSize;
   }
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[2] * B[bColStartingIndex+tid +nColsB + nColsB];//[2]
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[3] * B[bColStartingIndex+tid +nColsB + nColsB+ nColsB];//[3]
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+workgroupSize];//[2]+[3]
		//return;
//   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+k*halfWorkgroupSize];
   //return;
	
 //if(true)
 {
   

   //We have done our pointwise multiplication
   //If we started with 'n' cols in A, we now have n/2 items left to sum
   //C[get_global_id(0)]=rowA * nColsARowsB + tid;
   //C[rowA * nColsARowsB + tid]=13;//1000*get_global_id(0);
  // double missedBits=0;
  int doublWGSize=workgroupSize*2;
   while(remaining > 1)
   {

        //Add every second number
        //for (int k = 1; k < remaining; k++)
		int halfRem=remaining/2;
		
		int readLeft = workingSpaceStart;
		int writeTo =workingSpaceStart;
		
		
		
		for (int k = 0; k < halfRem; k++)
		{
            //int offset=k*workgroupSize;
			
			
			//int readRight = readLeft + workgroupSize;
            toSum[writeTo] =toSum[readLeft] +toSum[readLeft + workgroupSize];
			
			writeTo += workgroupSize;
			readLeft += doublWGSize;
		}
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//[0]+[1]+[2]+[3]
		//return;
		
		if(remaining % 2 == 1)
		{
			//does not divide into even number
			//Add the last item to first element
            toSum[workingSpaceStart] += toSum[readLeft];
	   }
	   

	   remaining=halfRem;
   }
  // toSum[workingSpaceStart] += missedBits;
 }
   // Write the matrix to device memory each 
   // thread writes one element
   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//A[ty * nColsARowsB];//value;
 
}
 



// OpenCL Kernel
__kernel void matrixMul_HighAccuracy_d_8(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __local double* toSum, __global double* C)
{

      //int tx = get_global_id(0); //column number for B
   //int ty = get_global_id(0);//row number for A
   
   
   
   //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//Work-item (Thread) ID
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
   {
	   int startRead=rowA * nColsARowsB;
	   
	   for(int col = tid; col < nColsARowsB; col += workgroupSize)
	   {
		   aRowData[col]=A[startRead  + col];//A: rowA, column readFromCol
	   }
   }
 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup

	
	//return;
	
 
   //Naive summation (low accuracy when many many small items to sum because of floating point stuff)
   /*
   double value = 0;
   int nextB=bColStartingIndex+tid;
   for (int k = 0; k < nColsARowsB; k++)
   {
      value += aRowData[k] * B[nextB];// B[k * nColsB + tid];//B: row k, column tid
	  nextB+=nColsB;
   }
   C[rowA * nColsB + bColStartingIndex + tid]=value;
   return;
   */
   
   

    //double values[nColsARowsB];

   //High accurate summation (better accuracy when many many small items to sum because of floating point stuff)
   int remaining=nColsARowsB/8;//remaining to sum
 //  C[get_global_id(0)] =nColsARowsB;
 //  return;
   int workingSpaceStart=tid;
   
  // int halfWorkgroupSize=workgroupSize/2;
   int nextB=bColStartingIndex+tid;
   int nextWriteTo=workingSpaceStart;
   for (int k = 0; k < nColsARowsB; )
   {
       //Multiply two entries
       //Multiply another two entries
       //Sum the results and save to the working space
	   //We use halfWorkgroupSize because k is going up in increments of 2
	   //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
	   //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
	   
	   
	   double res_n=0;
	   #pragma unroll
	   for(int n=0;n<2;n++)
	   {
		   double res_m=0;
		   double res0;
		   #pragma unroll
		   for(int m=0;m<2;m++)
		   {
			   res0 =aRowData[k] * B[nextB];
			   nextB+=nColsB; 
			   k++;
			   
			   res0 +=aRowData[k] * B[nextB];
			   nextB+=nColsB; 
			   k++;
			   
				res_m+=res0;
		   }
		   res_n+=res_m;
	   }
	   
	   
	   //Add the sums
       toSum[nextWriteTo] =res_n;// + (aRowData[k+1] * B[nextB+ nColsB]);//B: row k, column tid.
	   nextWriteTo+=workgroupSize;
   }
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[2] * B[bColStartingIndex+tid +nColsB + nColsB];//[2]
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[3] * B[bColStartingIndex+tid +nColsB + nColsB+ nColsB];//[3]
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+workgroupSize];//[2]+[3]
		//return;
//   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+k*halfWorkgroupSize];
   //return;
	
 //if(true)
 {
   

   //We have done our pointwise multiplication
   //If we started with 'n' cols in A, we now have n/2 items left to sum
   //C[get_global_id(0)]=rowA * nColsARowsB + tid;
   //C[rowA * nColsARowsB + tid]=13;//1000*get_global_id(0);
  // double missedBits=0;
  int doublWGSize=workgroupSize*2;
   while(remaining > 1)
   {

        //Add every second number
        //for (int k = 1; k < remaining; k++)
		int halfRem=remaining/2;
		
		int readLeft = workingSpaceStart;
		int writeTo =workingSpaceStart;
		
		
		
		for (int k = 0; k < halfRem; k++)
		{
            //int offset=k*workgroupSize;
			
			
			//int readRight = readLeft + workgroupSize;
            toSum[writeTo] =toSum[readLeft] +toSum[readLeft + workgroupSize];
			
			writeTo += workgroupSize;
			readLeft += doublWGSize;
		}
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//[0]+[1]+[2]+[3]
		//return;
		
		if(remaining % 2 == 1)
		{
			//does not divide into even number
			//Add the last item to first element
            toSum[workingSpaceStart] += toSum[readLeft];
	   }
	   

	   remaining=halfRem;
   }
  // toSum[workingSpaceStart] += missedBits;
 }
   // Write the matrix to device memory each 
   // thread writes one element
   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//A[ty * nColsARowsB];//value;
 
}
 


/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Each thread gets responsibility for one final entry in C
 * Threads are grouped into workgroups; each workgroup accesses the same row from A
 * A and B should both have their data organised row-at-a-time
 * aRowData should be of size nColsARowsB;
 * toSum is an array used for working - it should be nColsARowsB * nColsB / 2
 * nColsARowsB must be divisible by 8
 */
__kernel void matrixMul_HighAccuracy_d(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __local double* toSum, __global double* C)
{

      //int tx = get_global_id(0); //column number for B
   //int ty = get_global_id(0);//row number for A
   
   
   
   //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//Work-item (Thread) ID
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
   {
	   int startRead=rowA * nColsARowsB;
	   
	   for(int col = tid; col < nColsARowsB; col += workgroupSize)
	   {
		   aRowData[col]=A[startRead  + col];//A: rowA, column readFromCol
	   }
   }
 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup

 
   //Naive summation (low accuracy when many many small items to sum because of floating point stuff)
   /*
   double value = 0;
   int nextB=bColStartingIndex+tid;
   for (int k = 0; k < nColsARowsB; k++)
   {
      value += aRowData[k] * B[nextB];// B[k * nColsB + tid];//B: row k, column tid
	  nextB+=nColsB;
   }
   C[rowA * nColsB + bColStartingIndex + tid]=value;
   return;
   */
   

   //High accurate summation (better accuracy when many many small items to sum because of floating point stuff)
   int remaining=nColsARowsB/8;//remaining to sum
   int workingSpaceStart=tid;
   
   int nextB=bColStartingIndex+tid;
   int nextWriteTo=workingSpaceStart;
   for (int k = 0; k < nColsARowsB; )
   {
       //Multiply two entries
       //Multiply another two entries
       //Sum the results and save to the working space
	   //We use halfWorkgroupSize because k is going up in increments of 2
	   //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
	   //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
	   
	   //First two multiplications
	   double res0 =aRowData[k] * B[nextB];
	   nextB+=nColsB; 
	   k++;
	   
	   res0 +=aRowData[k] * B[nextB];
	   nextB+=nColsB; 
	   k++;
	   
	   //Second two multiplications
	   double res1 =aRowData[k] * B[nextB];
	   nextB+=nColsB; 
	   k++;
	   
	   res1 +=aRowData[k] * B[nextB];
	   nextB+=nColsB; 
	   k++;
	
		res0+=res1;
		
		//Third two multiplications
		//We re-use res1 here
		res1 =aRowData[k] * B[nextB];
	   nextB+=nColsB; 
	   k++;
	   
	   res1 +=aRowData[k] * B[nextB];
	   nextB+=nColsB; 
	   k++;
	
		//Fourth two multiplications
		double res2 =aRowData[k] * B[nextB];
	   nextB+=nColsB; 
	   k++;
	   
	   res2 +=aRowData[k] * B[nextB];
	   nextB+=nColsB; 
	   k++;
	
		res1+=res2;
	
	   //Add the sums
       toSum[nextWriteTo] =res0 + res1;// + (aRowData[k+1] * B[nextB+ nColsB]);//B: row k, column tid.
	   nextWriteTo+=workgroupSize;
   }
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[2] * B[bColStartingIndex+tid +nColsB + nColsB];//[2]
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[3] * B[bColStartingIndex+tid +nColsB + nColsB+ nColsB];//[3]
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+workgroupSize];//[2]+[3]
		//return;
//   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+k*halfWorkgroupSize];
   //return;
	
 //if(true)
 {
   

   //We have done our pointwise multiplication
   //If we started with 'n' cols in A, we now have n/2 items left to sum
   //C[get_global_id(0)]=rowA * nColsARowsB + tid;
   //C[rowA * nColsARowsB + tid]=13;//1000*get_global_id(0);
  // double missedBits=0;
  int doublWGSize=workgroupSize*2;
   while(remaining > 1)
   {

        //Add every second number
        //for (int k = 1; k < remaining; k++)
		int halfRem=remaining/2;
		
		int readLeft = workingSpaceStart;
		int writeTo =workingSpaceStart;
		
		
		
		for (int k = 0; k < halfRem; k++)
		{
            //int offset=k*workgroupSize;
			
			
			//int readRight = readLeft + workgroupSize;
            toSum[writeTo] =toSum[readLeft] +toSum[readLeft + workgroupSize];
			
			writeTo += workgroupSize;
			readLeft += doublWGSize;
		}
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//[0]+[1]+[2]+[3]
		//return;
		
		if(remaining % 2 == 1)
		{
			//does not divide into even number
			//Add the last item to first element
            toSum[workingSpaceStart] += toSum[readLeft];
	   }
	   

	   remaining=halfRem;
   }
  // toSum[workingSpaceStart] += missedBits;
 }
   // Write the matrix to device memory each 
   // thread writes one element
   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//A[ty * nColsARowsB];//value;
 
}
 
 
 
// OpenCL Kernel
__kernel void matrixMul_HighAccuracy_d_4(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __local double* toSum, __global double* C)
{

      //int tx = get_global_id(0); //column number for B
   //int ty = get_global_id(0);//row number for A
   
   
   
   //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//Work-item (Thread) ID
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
   {
	   int startRead=rowA * nColsARowsB;
	   
	   for(int col = tid; col < nColsARowsB; col += workgroupSize)
	   {
		   aRowData[col]=A[startRead  + col];//A: rowA, column readFromCol
	   }
   }
 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup

	
	//return;
	
 
   //Naive summation (low accuracy when many many small items to sum because of floating point stuff)
   /*
   double value = 0;
   int nextB=bColStartingIndex+tid;
   for (int k = 0; k < nColsARowsB; k++)
   {
      value += aRowData[k] * B[nextB];// B[k * nColsB + tid];//B: row k, column tid
	  nextB+=nColsB;
   }
   C[rowA * nColsB + bColStartingIndex + tid]=value;
   return;
   */
   
   

    //double values[nColsARowsB];

   //High accurate summation (better accuracy when many many small items to sum because of floating point stuff)
   int remaining=nColsARowsB/4;//remaining to sum
 //  C[get_global_id(0)] =nColsARowsB;
 //  return;
   int workingSpaceStart=tid;
   
  // int halfWorkgroupSize=workgroupSize/2;
   int nextB=bColStartingIndex+tid;
   int nextWriteTo=workingSpaceStart;
   for (int k = 0; k < nColsARowsB; )
   {
       //Multiply two entries
       //Multiply another two entries
       //Sum the results and save to the working space
	   //We use halfWorkgroupSize because k is going up in increments of 2
	   //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
	   //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
	   
	   //First two multiplications
	   double res0 =aRowData[k] * B[nextB];
	   nextB+=nColsB; 
	   k++;
	   
	   res0 +=aRowData[k] * B[nextB];
	   nextB+=nColsB; 
	   k++;
	   
	   //Second two multiplications
	   double res1 =aRowData[k] * B[nextB];
	   nextB+=nColsB; 
	   k++;
	   
	   res1 +=aRowData[k] * B[nextB];
	   nextB+=nColsB; 
	   k++;
	   
	   //Add the sums
       toSum[nextWriteTo] =res0 + res1;// + (aRowData[k+1] * B[nextB+ nColsB]);//B: row k, column tid.
	   nextWriteTo+=workgroupSize;
   }
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[2] * B[bColStartingIndex+tid +nColsB + nColsB];//[2]
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[3] * B[bColStartingIndex+tid +nColsB + nColsB+ nColsB];//[3]
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+workgroupSize];//[2]+[3]
		//return;
//   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+k*halfWorkgroupSize];
   //return;
	
 //if(true)
 {
   

   //We have done our pointwise multiplication
   //If we started with 'n' cols in A, we now have n/2 items left to sum
   //C[get_global_id(0)]=rowA * nColsARowsB + tid;
   //C[rowA * nColsARowsB + tid]=13;//1000*get_global_id(0);
  // double missedBits=0;
  int doublWGSize=workgroupSize*2;
   while(remaining > 1)
   {

        //Add every second number
        //for (int k = 1; k < remaining; k++)
		int halfRem=remaining/2;
		
		int readLeft = workingSpaceStart;
		int writeTo =workingSpaceStart;
		
		
		
		for (int k = 0; k < halfRem; k++)
		{
            //int offset=k*workgroupSize;
			
			
			//int readRight = readLeft + workgroupSize;
            toSum[writeTo] =toSum[readLeft] +toSum[readLeft + workgroupSize];
			
			writeTo += workgroupSize;
			readLeft += doublWGSize;
		}
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//[0]+[1]+[2]+[3]
		//return;
		
		if(remaining % 2 == 1)
		{
			//does not divide into even number
			//Add the last item to first element
            toSum[workingSpaceStart] += toSum[readLeft];
	   }
	   

	   remaining=halfRem;
   }
  // toSum[workingSpaceStart] += missedBits;
 }
   // Write the matrix to device memory each 
   // thread writes one element
   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//A[ty * nColsARowsB];//value;
 
}
 // OpenCL Kernel
__kernel void matrixMul_HighAccuracy_d_2(__global double* A, __global double* B, int nColsARowsB, int nColsB, int bColStartingIndex, __local double* aRowData, __local double* toSum, __global double* C)
{

      //int tx = get_global_id(0); //column number for B
   //int ty = get_global_id(0);//row number for A
   
   
   
   //Each workgroup gets one row from A (And all columns from B)
   //Each thread gets one column from B
   int rowA = get_group_id(0);//+aRowOffset;
   int workgroupSize = get_local_size(0);
   int tid = get_local_id(0);//Work-item (Thread) ID
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
   {
	   int startRead=rowA * nColsARowsB;
	   
	   for(int col = tid; col < nColsARowsB; col += workgroupSize)
	   {
		   aRowData[col]=A[startRead  + col];//A: rowA, column readFromCol
	   }
   }
 
    barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup

	
	//return;
	
 
   //Naive summation (low accuracy when many many small items to sum because of floating point stuff)
   /*
   double value = 0;
   int nextB=bColStartingIndex+tid;
   for (int k = 0; k < nColsARowsB; k++)
   {
      value += aRowData[k] * B[nextB];// B[k * nColsB + tid];//B: row k, column tid
	  nextB+=nColsB;
   }
   C[rowA * nColsB + bColStartingIndex + tid]=value;
   return;
   */
   
   

    //double values[nColsARowsB];

   //High accurate summation (better accuracy when many many small items to sum because of floating point stuff)
   int remaining=nColsARowsB/2;//remaining to sum
 //  C[get_global_id(0)] =nColsARowsB;
 //  return;
   int workingSpaceStart=tid;
   
   int halfWorkgroupSize=workgroupSize/2;
   int nextB=bColStartingIndex+tid;
   int nextWriteTo=workingSpaceStart;
   for (int k = 0; k < nColsARowsB;)
   {
       //Multiply two entries
       //Multiply another two entries
       //Sum the results and save to the working space
	   //We use halfWorkgroupSize because k is going up in increments of 2
	   //NB accessing data in B is coalesced, because it is row at a time. As each thread is getting an item in the same row, they are all side by side
	   //NB accessing data in aRowData is NOT coalesced (they all access the same item at the same time), but it is in local memory, so I don't know if this has much impact
	   
       toSum[nextWriteTo] =matrixMul_HighAccuracy_d_2_Sub(B,  nColsB, aRowData, &nextB, &k);
	   nextWriteTo+=workgroupSize;
   }
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[2] * B[bColStartingIndex+tid +nColsB + nColsB];//[2]
   //C[rowA * nColsB + bColStartingIndex + tid]=aRowData[3] * B[bColStartingIndex+tid +nColsB + nColsB+ nColsB];//[3]
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+workgroupSize];//[2]+[3]
		//return;
//   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart+k*halfWorkgroupSize];
   //return;
	
 //if(true)
 {
   

   //We have done our pointwise multiplication
   //If we started with 'n' cols in A, we now have n/2 items left to sum
   //C[get_global_id(0)]=rowA * nColsARowsB + tid;
   //C[rowA * nColsARowsB + tid]=13;//1000*get_global_id(0);
  // double missedBits=0;
  int doublWGSize=workgroupSize*2;
   while(remaining > 1)
   {

        //Add every second number
        //for (int k = 1; k < remaining; k++)
		int halfRem=remaining/2;
		
		int readLeft = workingSpaceStart;
		int writeTo =workingSpaceStart;
		
		
		
		for (int k = 0; k < halfRem; k++)
		{
            //int offset=k*workgroupSize;
			
			
			//int readRight = readLeft + workgroupSize;
            toSum[writeTo] =toSum[readLeft] +toSum[readLeft + workgroupSize];
			
			writeTo += workgroupSize;
			readLeft += doublWGSize;
		}
		//C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//[0]+[1]+[2]+[3]
		//return;
		
		if(remaining % 2 == 1)
		{
			//does not divide into even number
			//Add the last item to first element
            toSum[workingSpaceStart] += toSum[readLeft];
	   }
	   

	   remaining=halfRem;
   }
  // toSum[workingSpaceStart] += missedBits;
 }
   // Write the matrix to device memory each 
   // thread writes one element
   C[rowA * nColsB + bColStartingIndex + tid]=toSum[workingSpaceStart];//A[ty * nColsARowsB];//value;
 
}
 