//--------------------------------------------------------------
__kernel void CalcDistToNodes(__global float* nodeFeatures, __global uchar* samples, local float* workingSpace, __global int* result_best_indices)
{
   //PURPOSE
   //    For a collection of samples, calculates the closest node for each
   //DEFINITIONS:
   //    Node means a node in an some with n features
   //    Sample means an item with n features
   //    Feature means a floating point number
   //DATA:
   //    nodeFeatures is a list of features in each node. It is ordered NODE at a time. i.e. node0_feat0 node0_feat1 node0_feat2 ... , node1_feat0 node1_feat1 etc. It is n_nodes * n_features long
   //    samples is a list of features for each sample. It is ordered SAMPLE at a time. This is not the same ordering as nodeFeatures! i.e. sample0_feat0 sample0_feat1 sample0_feat2 ... sample(n-1)_feat0node, sample1_feat0, sample1_feat1 etc. It is n_features * n_samples long
   //    result_best_indices is a list of the best nodes, one value per sample
   //    workingSpace is nFeatures items long
   //LOGIC:
   //    Each group is assigned a single sample
   //    Each group should be nFeatures threads
   //    Each thread:
   //      is assigned one feature
   //      reads the sample[igroup][ithread]
   //      wait for sync
   //      for inode nodes:
   //          thread reads feature[ithread] from samples (local)
   //          thread reads feature[ithread]  from nodeFeatures[inode] (global)
   //          calculates the square difference
   //          atomically adds this to sum for inode
   //          if ithread == 0
   //              decides whether this is smaller than the best so far. If it is, we keep it
   //      if ithread == 0
   //          saves the best so far to result_best_indices[igroup]

   //Params
   int threadId = get_local_id(0);
   int groupId = get_group_id(0);
   int indexOfNodeFeature = threadId;
   //NB you can get NoFeatures with get_local_size if you want a non-hardcoded version


   int bestIndex = -1;
   float shortestSqDist = INFINITY;

   //Copy the sample locally
   float sampleFeature = (float)samples[groupId * %NoFeatures% +threadId];


   //sync
   workingSpace = 0;
   barrier(CLK_LOCAL_MEM_FENCE);

   //%NoFeatures% and %NoNodes% must be replaced by the code that opens this file in order for this to compile
//#pragma unroll //unrolling this is a bad idea for large numbers of nodes because the script gets seriously long
   for (int iNode = 0; iNode < %NoNodes%; iNode++)
   {
      float diff = nodeFeatures[indexOfNodeFeature] - sampleFeature;

      workingSpace[threadId] = diff*diff;

      barrier(CLK_LOCAL_MEM_FENCE);
      if (threadId == 0)
      {
         //sum the scores for this node
         float sqDist = 0;
#pragma unroll
         for (int iFeat = 0; iFeat < %NoFeatures%; iFeat++)
         {
            sqDist += workingSpace[iFeat];
         }

         //Do we have a smaller distance?
         if (sqDist < shortestSqDist)
         {
            shortestSqDist = sqDist;
            bestIndex = iNode;
         }
      }

      indexOfNodeFeature += %NoFeatures%;
      barrier(CLK_LOCAL_MEM_FENCE);

   }

   //Save the result
   if (threadId == 0)
   {
      result_best_indices[groupId] = bestIndex;
   }
}