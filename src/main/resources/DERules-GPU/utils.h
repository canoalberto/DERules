/**** RANDOM GENERATORS ***/

__device__ float rndFloat(curandState* globalState, int id) 
{
    curandState localState = globalState[id];
    float RANDOM = curand_uniform(&localState);
    globalState[id] = localState; 
    return RANDOM;
}

__device__ int rndInt(curandState* globalState, int id, int max) 
{
    curandState localState = globalState[id];
    float RANDOM = curand_uniform(&localState);
    globalState[id] = localState; 
    return RANDOM*max;
}

__device__ float rndFloat(curandState* globalState, int id, int max) 
{
    curandState localState = globalState[id];
    float RANDOM = curand_uniform(&localState);
    globalState[id] = localState; 
    return RANDOM*max;
}

__device__ float rndFloat(curandState* globalState, int id, int min,int max) 
{
    curandState localState = globalState[id];
    float RANDOM = curand_uniform(&localState);
    globalState[id] = localState; 
    return min+RANDOM*(max-min);
}

#define cudaCheckError() { \
	cudaError_t e=cudaGetLastError(); \
	if(e!=cudaSuccess) { \
		printf("CUDA error %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
		exit(0);\
	} \
}
