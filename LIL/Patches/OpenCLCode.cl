//------------------------------PATCH SEGMENTATION CODE
__kernel void PatchSSD(__global uchar* allPatches_T1, __global uchar* allPatches_T2, int nPatches, __global int* startFrom, __global uchar* kernIm_T1, __global uchar* kernIm_T2, int kernImXWidth, int kernImYHeight, int kernImZDepth, __global int* kernInternalIndices, __local uchar* kernelArr, __global float* resultWeight, __global int* resultPatchIndex, int groupOffset, __local int* ssds, __local int* indices, int keepNBestResults,uchar skipNBestResults, uchar useT2Char, uchar patchSize)
{

	//DEFINITIONS:
	//    Kernel means a 5x5x5 block of bytes from the that we want to compare to many others. There is only one
	//    Patch  means a 5x5x5 block of bytes that is compared with kernel. There are many
	//DATA:
	//    allPatches is a giant list of patches as one column. The data order is patch(fastest),x,y,z
	//        -So moving through:
	//            [0] is (0,0,0) of patch [0]
	//            [1] is (0,0,0) of patch [1]
	//            [nPatches] is (1,0,0) of patch [0]
	//            [nPatches+1] is (1,0,0) of patch [1]
	//            [nPatches*2] is (2,0,0) of patch [0]
	//            [nPatches*nPatches] is (0,0,1) of patch [0] (I think)
	//    nPatches is how many patches are held in allPatches (nPatches.Length/125)
	//    startFrom[groupId] is the patch number in allPatches which we compare from            
	//    kernIm is a 3D image from which we take our kernel.  The data order is z fastest, then y, then x
	//  kernimXWidth etc are the sizes of kernIm
	//    kernInternalIndices[groupId] gives the location within kern from which to start taking the kernel. This is the top-left-back of this kernel, not the centre
	//    kernelArr is a local copy of the kernel, read from kernIm at the specified position
	//    resultWeight holds the 1/SSD for each kernal-patch comparison
	//    resultPatchIndex holds the patch index that corresponds to the weights in resultWeight (needed because we filter our results)
	//    resultOffset states where in result the first SSD should be written to
	//    groupOffset should be added to groupID to get the global groupID (allows for 'chunking')
	//    ssds is working space to filter SSDs down to the n Best
	//    indices are the indices corresponding to SSDs in ssds
	//	  skipNBestResults indicated the number of best results that we skip over. e.g. if this is 3, we ignore the three lowest cost results
	//	  keepNBestResults indicated the number of best results to return. e.g. if this is 3, we return 3 results
	//WORKGROUP:
	//The workgroup size  = how many patches are compared
	//LOGIC:
	//Each group reads the kernel given into local memory
	//Each thread:
	//  is assigned one patch to compare the kernel to
	//  compares the kernel to that 5x5x5 item using SSD
	//  saves the result to the central position of that kernel in the output image


	//Read the kernel from global to local memory
	int threadId = get_local_id(0);
	int groupId = get_group_id(0) + groupOffset;
	{
	int        ssd = 0;
	__global uchar* patchSource = allPatches_T1;
	__global uchar* kernSource = kernIm_T1;
	int compareToPatchIndex = startFrom[groupId] + threadId;
	
	for (int iIm = 0; iIm < 2; iIm++)
	{
		int nVoxInPatch = (patchSize*patchSize*patchSize);
		if (threadId < nVoxInPatch)
		{
			//threads 0-nVoxInPatch read one item each

			//--Kernel info
			int strideYGlobalKern = kernImXWidth; //How far to step to get the next Y at the same X coord. next item is same Z, same Y, different X. So increment by XWidth for next Y
			int strideZGlobalKern = kernImYHeight*strideYGlobalKern; //How far to step to get the next Z at the same X and Y coords. next item is same Z, same Y, different X. So increment by XWidth * YHeight for next Z


																	 //We know where the first thread should read from: kernInternalIndices
																	 //Figure out how this thread is offset from that position
			int zOffset = threadId / (patchSize*patchSize);
			int yOffset = threadId / patchSize - zOffset * patchSize;
			int xOffset = threadId - zOffset * (patchSize*patchSize) - yOffset * patchSize;
			/*int xGlobalKern = (xOffset + kernX);
			int yGlobalKern = (yOffset + kernY);
			int zGlobalKern = (zOffset + kernZ);
			int indexOfGlobal = zGlobalKern * strideZGlobalKern + yGlobalKern * strideYGlobalKern + xGlobalKern;*/

			//Add it all up to give us where we want to read from
			int indexOfGlobal = kernInternalIndices[groupId] + zOffset * strideZGlobalKern + yOffset * strideYGlobalKern + xOffset;
			//result[threadId + resultOffset + get_local_size(0) * groupId] = indexOfGlobal;
			//return;
			kernelArr[threadId] = kernSource[indexOfGlobal];

		}
		//result[threadId + resultOffset + get_local_size(0) * groupId] = -1;
		//return;

		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads


		int nextIndex = compareToPatchIndex;

		//#pragma unroll
		for (int i = 0; i < nVoxInPatch; i++)
		{
			//We are reading from global memory but all threads will be in sync and the memory should be coalesced because of its ordering
			int diff = kernelArr[i] - patchSource[nextIndex];
			ssd += diff*diff;

			nextIndex += nPatches;
		}

		if (useT2Char == 1)
		{
			patchSource = allPatches_T2;
			kernSource = kernIm_T2;
			barrier(CLK_LOCAL_MEM_FENCE); //Sync threads before we loop around and load new stuff into local memory
		}
		else
		{
			break;
		}
	}
	//ssd = allPatches[startFrom * nPatches + threadId]; 

	ssds[threadId] = ssd;//convert to weight and save to working space
	indices[threadId] = compareToPatchIndex;
}
	//Sort the SSds from lowest to highest
	//This is just odd-even (bubble) sort because it's simple to write right now
	{
	int nItems = get_local_size(0);
	int leftIndex = threadId * 2;
	int rightIndex = threadId * 2 + 1;
	for (int i = 0; i < nItems / 2; i++)//we need nItems/2 loops to guarantee it's sorted (NB it does two things per loop!)
	{
		//Left is even, right is odd
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads
		if (threadId < nItems / 2)
		{
			int leftSSD = ssds[leftIndex];
			int rightSSD = ssds[rightIndex];
			if (leftSSD > rightSSD)
			{

				ssds[leftIndex] = rightSSD;
				ssds[rightIndex] = leftSSD;

				{
					int temp = indices[leftIndex];
					indices[leftIndex] = indices[rightIndex];
					indices[rightIndex] = temp;
				}
			}
		}
		//Left is ODD, right is even
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads
		if (threadId < nItems / 2 - 1)//the last thread shouldn't execute because it will go past the bounds of the array
		{
			leftIndex++;
			rightIndex++;

			int leftSSD = ssds[leftIndex];
			int rightSSD = ssds[rightIndex];
			if (leftSSD > rightSSD)
			{


				ssds[leftIndex] = rightSSD;
				ssds[rightIndex] = leftSSD;

				{
					int temp = indices[leftIndex];
					indices[leftIndex] = indices[rightIndex];
					indices[rightIndex] = temp;
				}
			}
			leftIndex--;
			rightIndex--;
		}
	}
	}
	barrier(CLK_LOCAL_MEM_FENCE); //Sync threads

								  //SSDs are now sorted
								  //Save final result as 1/SSD (weight) to global memory to return
	
	if (threadId < keepNBestResults)
	{
		
		int resultIndex = threadId + keepNBestResults  * (groupId - groupOffset);
		resultWeight[resultIndex] = 1.0f / (ssds[threadId+skipNBestResults] + 0.000001f);;//add epsilon prevents NaN etc due to zeros
		resultPatchIndex[resultIndex] = indices[threadId+skipNBestResults];// allPatches[startFrom + threadId];
	}
}



__kernel void CalcWeightSum(__global float* patchWeights,int nPatchesInWeights, __global float* filter, __global float* result_weightSums, int voxPerPatch)
{
    //PURPOSE
    //    For a collection of patches, calculates the sum of all weights
    //DEFINITIONS:
    //    Patch  means a 5x5x5 block of bytes. There are many
    //    Template image is the image the patches have come from
    //    Sample image is the image that patches have been found to match some part of
    //DATA:
    //    patchWeights is a list of weights to apply to each patch. It is nPatchesInWeights * nGroups long 
    //    nPatchesInWeights is how many patches belong to each group
    //    filter is a kernel, like a gaussian kernel, that the patches are multiplied by after being added
    //    result_patchSums is where the final weighted kernel sum is placed (one kernel per workgroup), so it's voxPerPatch * nGroups
    //LOGIC:
    //    Each group is assigned a collection of patches (count per group = nPatchesInWeights) from the template that have all been found to match a patch from the sample image
    //    Each group should be voxPerPatch threads (e.g. 125 for a 5x5x5 patch)
    //    Each thread:
    //      is assigned one x,y,z position between (0,0,0) and (5,5,5) for a 5x5x5 patch
    //      for i patches:
    //          thread reads this position to give the intensity of that voxel from patch i
    //          multiplies this value by its appropriate weight
    //          keeps a running sum
    //      multiplies the final sum by the kernel value at this voxel index (kernel being like a gaussian)
    //      outputs the final sum. Because groups will have overlapping regions, this output is combined into an image in a follow up method

    //Params
    int threadId = get_local_id(0);
    int groupId = get_group_id(0);
    float sumOfWeights=0;
    int index =groupId * nPatchesInWeights;//index for weights/index array.
    for(int iPatch=0;iPatch<nPatchesInWeights;iPatch++)
    {
        //Read the current patch in from global memory
        //NB we have voxPerPatch threads - exactly one per voxel in the patch
        //so each thread holds only one voxel value
        sumOfWeights+= patchWeights[index];
        index++;
    }

    //Apply gaussian/whatever
    float filterVal=filter[threadId];
    sumOfWeights*= filterVal;

    index=groupId * voxPerPatch + threadId;//reuse index. now refers to the save index
    result_weightSums[index] = sumOfWeights;
}

__kernel void WeightAndSumBestPatches(__global float* allPatches, int nPatchesInAllPatches, __global float* patchWeights,__global int* patchIndices, int nPatchesInWeights, __global float* filter, __global float* result_patchSums, int patchSize)
{
    //PURPOSE
    //    For a collection of  patches, multiplies each by its appropriate weight, and adds them
    //DEFINITIONS:
    //    Patch  means a 5x5x5 block of bytes. There are many
    //    Template image is the image the patches have come from
    //    Sample image is the image that patches have been found to match some part of
    //DATA:
    //    allPatches is a giant list of patches as one column. The data order is patch(fastest),x,y,z
    //        -So moving through:
    //            [0] is (0,0,0) of patch [0]
    //            [1] is (0,0,0) of patch [1]
    //            [nPatchesInAllPatches] is (1,0,0) of patch [0]
    //            [nPatchesInAllPatches+1] is (1,0,0) of patch [1]
    //            [nPatchesInAllPatches*2] is (2,0,0) of patch [0]
    //            [nPatchesInAllPatches*nPatchesInAllPatches] is (0,0,1) of patch [0] (I think)
    //    nPatchesInAllPatches is how many patches are held in allPatches (nPatchesInAllPatches.Length/patchSize)
    //    patchWeights is a list of weights to apply to each patch. It is nPatchesInWeights * nGroups long
    //    patchIndices is a list of patchIDs that correspond to patchWeights. 
    //    nPatchesInWeights is how many patches belong to each group
    //    filter is a kernel, like a gaussian kernel, that the patches are multiplied by after being added
    //    result_patchSums is where the final weighted kernel sum is placed (one kernel per workgroup), so it's patchSize * nGroups
    //LOGIC:
    //    Each group is assigned a collection of patches (count per group = nPatchesInWeights) from the template that have all been found to match a patch from the sample image
    //    Each group should be patchSize threads for a 5x5x5 patch
    //    Each thread:
    //      is assigned one x,y,z position between (0,0,0) and (5,5,5) for a 5x5x5 patch
    //      for i patches:
    //          thread reads this position to give the intensity of that voxel from patch i
    //          multiplies this value by its appropriate weight
    //          keeps a running sum
    //      multiplies the final sum by the kernel value at this voxel index (kernel being like a gaussian)
    //      outputs the final sum. Because groups will have overlapping regions, this output is combined into an image in a follow up method

    //Params
    int threadId = get_local_id(0);
    int groupId = get_group_id(0);
    
    
    
    float sumOfWeightsTimeIntensity=0;
    
    int index =groupId * nPatchesInWeights;//index for weights/index array.
    for(int iPatch=0;iPatch<nPatchesInWeights;iPatch++)
    {
        //Read the current patch in from global memory
        //NB we have (patchSize*patchSize*patchSize) threads - exactly one per voxel in the patch
        //so each thread holds only one voxel value
        
        float patchWeight = patchWeights[index];
        int patchIndex = patchIndices[index] + threadId*nPatchesInAllPatches;//index in allPatches. Offset by threadId*nPatchesInAllPatches to get the correct voxel for this thread
        float weightedVoxel = allPatches[patchIndex] * patchWeight;//Multiply by the appropriate weight
        sumOfWeightsTimeIntensity+=weightedVoxel;
        
        index++;
    }
    //Apply gaussian/whatever
    float filterVal=filter[threadId];
    sumOfWeightsTimeIntensity *= filterVal;
    

    index=groupId * (patchSize*patchSize*patchSize) + threadId;//reuse index. now refers to the save index
    result_patchSums[index] = sumOfWeightsTimeIntensity;

}

__kernel void ReconstructImageStage1(__global float* patchSums,__global int* patchInternalIndices,  int ImXWidth, int ImYHeight, int ImZDepth,  __global float* Im_Intensities, int noPatchesTotal,int patchInternalIndicesOffset, int patchSize)
{

    //DEFINITIONS:
    //    Patch  means a 5x5x5 block of FLOATS that may overlap with others
    //DATA:
    //    patchSums is a collection of patch intensity*weight OR similar (this kernel can be used for weights only too). Ordered x,y,z,patch.
    //    Im_Intensities is a series of reconstructed images that will later be combined. Values are sum of all intensity*weights. Order is x (fastest),y,z,image
    //    ImXWidth etc are the sizes of the resulting images
    //    patchInternalIndices[i] gives the location within the result images from which to start start writing the resulting patch. This is the top-left-back of this kernel, not the centre
    //WORKGROUP:
    //The workgroup size  = no voxels in a patch
    //The number of workgroups should be:
    //  * < the number of patches to read
    //  * <= MaxMemoryAllocationSize / (ImXWidth * ImYHeight * ImZDepth), to prevent the result arrays being too large
    //LOGIC:
    //Each group loops through all the patches it is assigned
    //For each patch:
    //      Each thread:
    //          convert it's assigned xyz patch position (between 0,0,0, and 5,5,5) into its final saveTo location in the image
    //          reads its  assigned specific x,y,z position in the patchSums
    //          save this to Im_Intensities
   
 int noWG =get_num_groups(0);
    int groupId = get_group_id(0);
    int threadId = get_local_id(0);
    int patchFrom = noPatchesTotal*groupId/noWG;   
    int patchTo = noPatchesTotal*(groupId+1)/noWG;//this slightly more awkward way of counting will give the right result if the no patches doesn't divide evenly across workgroups
	int patchSliceSize = patchSize * patchSize;
	int voxPerPatch = (patchSize*patchSize*patchSize);
    //Do intial work to calculate final save-to position
        //--Kernel info
        int strideYGlobal = ImXWidth; //How far to step to get the next Y at the same X coord. next item is same Z, same Y, different X. So increment by XWidth for next Y
        int strideZGlobal = ImYHeight*strideYGlobal; //How far to step to get the next Z at the same X and Y coords. next item is same Z, same Y, different X. So increment by XWidth * YHeight for next Z

        //We know where kernel (0,0,0) should be written to
        //--Figure out how this thread is offset from that position
        int zOffset = threadId / patchSliceSize;
        int yOffset = threadId / patchSize - zOffset * patchSize;
        int xOffset = threadId - zOffset * patchSliceSize - yOffset * patchSize;
        int totalOffset=zOffset * strideZGlobal + yOffset * strideYGlobal + xOffset;
        //--Offset this further because each group writes to its own image
        totalOffset += groupId * ImXWidth * ImYHeight * ImZDepth;
        
        int readFrom = voxPerPatch * patchFrom + threadId;
    patchInternalIndices+=patchInternalIndicesOffset;//offset the patchInternalIndices positions to account for chunking
    for(int iPatch = patchFrom; iPatch < patchTo; iPatch++)
    {
        //Calculate the final save-to position
        int indexOfGlobal = patchInternalIndices[iPatch] + totalOffset;
        
        //read assigned specific x,y,z position in the patchSums and add to the intensities image
        Im_Intensities[indexOfGlobal] += patchSums[readFrom];
		
        readFrom+= voxPerPatch;
		barrier(CLK_LOCAL_MEM_FENCE); //Sync threads as results from one patch can overlap with the next patch
    }
}
__kernel void ReconstructImageStage2(int voxelsPerImage, int noImagesInSeries, __global float* imageSeries, __global float* result_reconstructedImage)
{
    //PURPOSE:
    //imageSeries holds a series of images
    //This adds those series together to get one result  for imageSeries
    //This result is saved at the start of imageSeries
    //DEFINITIONS:
    //    
    //DATA:
    //    imageSeries is a series of reconstructed images that will  be combined. Values are sum of all intensity*weights. Order is x (fastest),y,z,image
    //    voxelsPerImage is how many voxels are in each image in imageSeries
    //    noImagesInSeries gives the number of images that are in imageSeries
    //WORKGROUP:
    //The workgroup size  = ImXWidth
    //The number of workgroups is ImYHeight * ImZDepth
    //LOGIC:
    //Each unique thread is assigned one x,y,z position
    //These threads loop through all images, summing the value at that x,y,z position to its total
    //The sum is then placed at that x,y,z position in the first image in the series
    
   
    //Do initial work to calculate final save-to position
        int ImXWidth = get_local_size(0);
        int readAndSaveFromOrig = get_group_id(0) * ImXWidth + get_local_id(0);
        int readFromNext=readAndSaveFromOrig;
        float sum=0;
        for(int iImage = 0; iImage < noImagesInSeries; iImage++)
    {
        //read assigned specific x,y,z position in the patchSums and add to the running total
        sum+= imageSeries[readFromNext];
        readFromNext+=voxelsPerImage;
    }
    
    //save to the final location in the imageSeries
    result_reconstructedImage[readAndSaveFromOrig] += sum;
}

