inline int MatrixMul_32x32_Step1_GetIndex(int patch_col, int patch_row, int nCols_patch)
{
    //returns the start point to copy from A or B for matrixMul_32x32_Step1
    return (patch_row * nCols_patch+patch_col)*1024;
}

inline float MatrixMul_32x32_Step1_MultAndOffset_2_f(__local float* aLocal, __local float* bLocal)
{
    //sums two items, and offset the array for this to happen again
    //Using this method in a pyramidal way will give better summation accuracy
    float sum = (*aLocal)*(*bLocal);
    aLocal++;
    bLocal+=32;
    sum += (*aLocal)*(*bLocal);
    return sum;
}

inline float MatrixMul_32x32_Step1_MultAndOffset_4_f(__local float* aLocal, __local float* bLocal)
{
    //sums four items, and offset the array for this to happen again
    //Using this method in a pyramidal way will give better summation accuracy
    return MatrixMul_32x32_Step1_MultAndOffset_2_f(aLocal,bLocal) + MatrixMul_32x32_Step1_MultAndOffset_2_f(aLocal+2,bLocal+64);
}

inline float MatrixMul_32x32_Step1_MultAndOffset_8_f(__local float* aLocal, __local float* bLocal)
{
    //sums 8 items, and offset the array for this to happen again
    //Using this method in a pyramidal way will give better summation accuracy
    return MatrixMul_32x32_Step1_MultAndOffset_4_f(aLocal,bLocal) + MatrixMul_32x32_Step1_MultAndOffset_4_f(aLocal+4,bLocal+128);
}
inline float MatrixMul_32x32_Step1_MultAndOffset_16_f(__local float* aLocal, __local float* bLocal)
{
    //sums 16 items, and offset the array for this to happen again
    //Using this method in a pyramidal way will give better summation accuracy
    return MatrixMul_32x32_Step1_MultAndOffset_8_f(aLocal,bLocal) + MatrixMul_32x32_Step1_MultAndOffset_8_f(aLocal+8,bLocal+256);
}
inline float MatrixMul_32x32_Step1_MultAndOffset_32_f(__local float* aLocal, __local float* bLocal)
{
    //sums 16 items, and offset the array for this to happen again
    //Using this method in a pyramidal way will give better summation accuracy
    return MatrixMul_32x32_Step1_MultAndOffset_16_f(aLocal,bLocal) + MatrixMul_32x32_Step1_MultAndOffset_16_f(aLocal+16,bLocal+512);
}
kernel void MatrixMul_32x32_Step1_Multiplication_f(__global float* A, __global float* B, int nRowsA_patch, int nColsARowsB_patch, int nColsB_patch, __local float* aLocal, __local float* bLocal, __global float* C)
{  
   
   //Performs the first step of matrix multiplication using a 32x32 patch based approach
   //Input:
   //   A,B:                matrices to be multiplied, supplied 32x32 patch at a time ordered row-at-a-time both in inter-patch and intra-patch terms
   //   nRowsA_patch        number of rows in A, divided by 32 (the number of patches that fit into one row of A)
   //   nColsARowsB_patch:  the number of cols in A and rows in B divided by 32 
   //   nColsB_patch:       the number of columns in B divided by 32 (the number of patches that fit into one row of B)
   //   aLocal:             a local array that holds the 32x32 patch from A
   //   bLocal:             a local array that holds the 32x32 patch from B
   //   C:                  the result of each 32x32 matrix multiplication, concatenated into one long array. It should be nColsA * nColsB * nRowsA long
   //Input and workgroup size:
   //   A and B sizes much divide cleanly into 32
   //   workgroup Ids should be 2D
   //       [0] aRows * BCols workgroups:   indicating the 32x32 patch of  the output matrix that this workgroup result is for
   //       [1] bRows:  indicating how much colA and rowB should be offset by (remember for each position there are rowB numbers of additions)
   //   Each workgroup should be 1024 threads in 1D 
   //Global:   [0] Number of elements in the final matrix. 
   //          [1] Number of additions required when summing multiplications (=no 32x32 patches in a column from A or Row from B)
   //Workgroup:[0] each workgroup handles a 32x32 patch
   //          [1] each workgroup calculates only one 32x32 * 32x32 multiplication. The results for each WG of the same end location are handled by the second method
   //Process:
   //   Each workgroup takes one 32x32 patch
   //   Each thread has a 2D id between (0,0) --> (31,31) inclusive
   //   Threads work together to load the patch from A into aLocal (one item per thread)
   //   Threads work together to load the patch from B into bLocal (one item per thread)
   //   Each thread calculates one row from aLocal x one column from bLocal and saves to C
   
    
   //convert group ID into the two-patches to be multiplied
   //--Find the row/column match
   int gid = get_group_id(0);
   int patch_row_A = gid / nColsB_patch;
   int patch_col_B = gid % nColsB_patch;
   //--Add how far along the row/column this workgroup should take a patch from

   int patch_col_A_row_B = get_group_id(1);
   
   
   int threadId=get_local_id(0);
   int localRowA = threadId/32;//row withing localA that this will use
   int localColB = threadId - localRowA *32;//cols within localB that this will use
   
   //Copy data locally so we can access it faster
   //Each thread copies only a subset of items, splitting the workload across
   //all threads in this workgroup
    aLocal[threadId] = A[MatrixMul_32x32_Step1_GetIndex(patch_col_A_row_B,patch_row_A, nColsARowsB_patch) + threadId];
    bLocal[threadId] = B[MatrixMul_32x32_Step1_GetIndex(patch_col_B,patch_col_A_row_B, nColsB_patch) + threadId];
   barrier(CLK_LOCAL_MEM_FENCE); //Sync threads in this workgroup
   



   //Naive summation
   //--offset to the correct row
   aLocal+=localRowA*32;
   bLocal+=localColB;

   //--Cross product of the the 32-item-long vectors 
    float sum = MatrixMul_32x32_Step1_MultAndOffset_32_f(aLocal,bLocal);;//MatrixMul_32x32_Step1_MultAndOffset_32(aLocal,bLocal);

    //Within this group, results are written row-at-a-time for all columns.
    //Across groups, results are ordered groupid(0) fastest (i.e. all for the same 32x32 patch are NOT together), then by group(1)
    /*E.G for a result matrix which is 64x64 (two patches by two patches) these will be saved like so:
    * [index in output]: result matrix patch row, result matrix patch col
    * [0-1024]: 0 0
    * [1024-2048]:  0 1 
    * [2048 - 3000ish]: 1 0
    * [300ish - 4096]: 1 1
    * then repeated the same but for matA source col and matB source row offset by 1
    */
    int offsetC = (get_group_id(1)* get_num_groups(0)  + gid) * 1024;
    C[offsetC + localRowA * 32 + localColB] =sum;
}
inline int MatrixMul_32x32_Step2_GetInitialPos(int nRowsA_patch, int nColsARowsB_patch, int nColsB_patch)
{
   //convert group ID into the two-patches to be multiplied
   //--Find the row/column match
   //int gid = get_group_id(0);
   //int patch_row_A = gid / nColsB_patch;//correct but not used so commented
   //int patch_col_B = gid % nColsB_patch;//correct but not used so commented

   int threadId=get_local_id(0);
   int localRowA = threadId/32;//row withing localA that this will use
   int localColB = threadId - localRowA *32;//cols within localB that this will use

    return get_group_id(0) * 1024+ localRowA * 32 + localColB;
}


kernel void MatrixMul_32x32_Step2_Reduction_f(__global float* C, int nRowsA_patch, int nColsARowsB_patch, int nColsB_patch)
{  
    
   //Performs the second step of matrix multiplication using a 32x32 patch based approach - it takes two patches that need to be added and adds them
   //this step will need be repeated for as many reductions are required (64 cols in A = 1, 128 cols in A = 2, etc)
   //Input:
   //   C:                  result from matrixMul_32x32_Step1 (see that algorithm for details on data ordering) 
   //   nColsARowsB_patch:  the number of cols in A and rows in B divided by 32 (the number of patches that fit into one row of A)
   //   nColsB_patch:       the number of columns in B divided by 32 (the number of patches that fit into one row of B)
   //   reducedC:           reduced result from C. The number of entries will be half the size of C
   //Input and workgroup size:
   //   workgroups Ids should be 1D between 0 and aCols/32
   //   Each workgroup should be 1024 threads
   //Process:
   //   Each workgroup adds one resulting 32x32 patch from C to another (remember for each patch, there is a whole stack of addition that has not been done)
   //   Each thread has a 1D id between 0-1204 which corresponds with (0,0) --> (31,31) inclusive within the patch:
   //       Offset to the start of this patch, plus this local row/column position
   //       Add the next patch that is for the same position
   //       save the sum in reducedC

   
   
   //convert group ID into the two-patches to be multiplied
   int initialPos = MatrixMul_32x32_Step2_GetInitialPos(nRowsA_patch, nColsARowsB_patch, nColsB_patch);
   
    //C[threadId] = (patch_row*32 +localRowA)*nColsB_patch*32 + patch_col*32 + localColB;// = localColB*32+localRowA;
//return;   
   

    //Within this group, results are written row-at-a-time for all columns.
    //Across groups, results are ordered groupid(0) fastest (i.e. all for the same 32x32 patch are NOT together), then by group(1)
    /*E.G for a result matrix which is 64x64 (two patches by two patches) these will be saved like so:
    * [index in output]: result matrix patch row, result matrix patch col
    * [0-1024]: 0 0
    * [1024-2048]:  0 1 
    * [2048 - 3000ish]: 1 0
    * [300ish - 4096]: 1 1
    * then repeated the same but for matA source col and matB source row offset by 1
    */
    int stepSize = get_num_groups(0)*1024;

    C[initialPos] += C[initialPos+stepSize];
}

kernel void MatrixMul_32x32_Step2_Reduction_ReduceOneItem_f(__global float* C, int nRowsA_patch, int nColsARowsB_patch, int nColsB_patch, int stepSize)
{  
    //Just like MatrixMul_32x32_Step2_Reduction_d but stepSize is provided.
//This allows this to be used to reduce just one item, rather than halve the array, when the array has an odd number of patches. 
//e.g. 1,2,3,4 --> 1+3,2+4 for the previous method
//but 1,2,3,4,5 -->  1+5,2,3,4 for this method (now we are an even number of items long and can call the other method to reduce)
   //Performs the second step of matrix multiplication using a 32x32 patch based approach - it takes two patches that need to be added and adds them
   //this step will need be repeated for as many reductions are required (64 cols in A = 1, 128 cols in A = 2, etc)
   //Input:
   //   C:                  result from matrixMul_32x32_Step1 (see that algorithm for details on data ordering) 
   //   nColsARowsB_patch:  the number of cols in A and rows in B divided by 32 (the number of patches that fit into one row of A)
   //   nColsB_patch:       the number of columns in B divided by 32 (the number of patches that fit into one row of B)
   //   reducedC:           reduced result from C. The number of entries will be half the size of C
   //Input and workgroup size:
   //   workgroups Ids should be 1D between 0 and aCols/32
   //   Each workgroup should be 1024 threads
   //Process:
   //   Each workgroup adds one resulting 32x32 patch from C to another (remember for each patch, there is a whole stack of addition that has not been done)
   //   Each thread has a 1D id between 0-1204 which corresponds with (0,0) --> (31,31) inclusive within the patch:
   //       Offset to the start of this patch, plus this local row/column position
   //       Add the next patch that is for the same position
   //       save the sum in reducedC

   
   
	int initialPos = MatrixMul_32x32_Step2_GetInitialPos(nRowsA_patch, nColsARowsB_patch, nColsB_patch);

    C[initialPos] += C[initialPos+stepSize];
//C[initialPos] = initialPos+stepSize;
}