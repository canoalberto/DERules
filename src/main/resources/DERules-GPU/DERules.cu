#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <curand_kernel.h>
#include "utils.h"
#include "jni/moa_classifiers_evolutionary_DERules.h"

#define THREAD_BLOCK 128

/**** HOST AND DEVICE VARIABLES ****/
int h_numberInstances, h_numberAttributes, h_numberClasses, h_numberRulesClass;
float *d_population, *d_offspring, *h_population, *h_minValue, *h_maxValue, *d_minValue, *d_maxValue, *d_classifier, *h_classifier;
float *d_dataset, *d_instance, *d_votes;
int *d_elitistIndividuals;
unsigned char *d_coverage;
curandState* d_devStates;
bool elitism;
static jfloatArray classifierArrayGlobal;

int seed, numberGenerations, h_populationSize, h_genotypeLength;
const float h_crossoverProbability=0.9f, h_F=0.8f;

__constant__ int d_populationSize, d_genotypeLength, d_numberAttributes, d_numberClasses, d_numberRulesClass;
__constant__ float d_crossoverProbability, d_F;

/*** SEEDS ****/
__global__ void setup_kernel (curandState* state, unsigned long seed, unsigned int size)
{
	int generatorID = blockIdx.x*blockDim.x + threadIdx.x;

	if(generatorID < size)
		curand_init(seed, generatorID, 0, &state[generatorID]);
}

/**** DIFFERENTIAL EVOLUTION: INITIALIZATION ***/
__global__ void kernel_Initialization(curandState* globalState, float* population, float* minValue, float* maxValue, int* elitistIndividuals, bool elitism)
{
	int generatorID = (blockIdx.x * d_numberClasses * d_numberAttributes * d_populationSize) + (blockIdx.y * d_numberAttributes * d_populationSize) + (blockIdx.z * d_populationSize) + threadIdx.x;
	int gen = (blockIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (blockIdx.z * 2 * d_populationSize) + threadIdx.x;

	curandState localState = globalState[generatorID];

	if(elitism == false || elitistIndividuals[blockIdx.x * d_numberClasses + blockIdx.y] != threadIdx.x)
	{
		population[gen             ] = minValue[blockIdx.z]+curand_uniform(&localState)*(maxValue[blockIdx.z]-minValue[blockIdx.z]);
		population[gen + blockDim.x] = minValue[blockIdx.z]+curand_uniform(&localState)*(maxValue[blockIdx.z]-minValue[blockIdx.z]);
	}

	globalState[generatorID] = localState;
}

/**** DIFFERENTIAL EVOLUTION: EVALUATION : COVERAGE ***/
__global__ void kernel_Coverage(float* population, float* dataset, unsigned char* coverage, int numberInstances, int actualRuleClassSet)
{
	int gen = (actualRuleClassSet * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (blockIdx.z * 2 * d_populationSize) + threadIdx.x;

	float minValue = population[gen];
	float maxValue = population[gen + blockDim.x];

	if(minValue <= maxValue && (dataset[blockIdx.x*d_numberAttributes + blockIdx.z] < minValue || dataset[blockIdx.x*d_numberAttributes + blockIdx.z] > maxValue))
		coverage[(actualRuleClassSet * d_numberClasses * numberInstances * d_populationSize) + (blockIdx.y * numberInstances * d_populationSize) + (blockIdx.x * d_populationSize) + threadIdx.x] = 1;  // 1 means DO NOT cover, 0 initialized by default means yes.
}

/**** DIFFERENTIAL EVOLUTION: EVALUATION : FITNESS ***/
__global__ void kernel_Fitness(float* population, float* dataset, unsigned char* coverage, int numberInstances)
{
	unsigned int tp = 0, fp = 0, tn = 0, fn = 0;

	for(int i = 0; i < numberInstances; i++)
	{
		unsigned char covers = coverage[(blockIdx.x * d_numberClasses * numberInstances * d_populationSize) + (blockIdx.y * numberInstances * d_populationSize) + (i * d_populationSize) + threadIdx.x];

		if(covers == 0) // IF COVERS
		{
			if(blockIdx.y == dataset[i*d_numberAttributes + d_numberAttributes-1])
				tp++;
			else
				fp++;
		}
		else // IF DOESNT COVER
		{
			if(blockIdx.y != dataset[i*d_numberAttributes + d_numberAttributes-1])
				tn++;
			else
				fn++;
		}
	}

	float se = tp+fn != 0 ? (tp/(float)(tp+fn)) : 1;
	float sp = tn+fp != 0 ? (tn/(float)(tn+fp)) : 1;

	//printf("%d %d %d: %d %d %d %d %f\n", blockIdx.x, blockIdx.y, threadIdx.x, tp, tn, fp, fn, se * sp);

	// Set the fitness to the individual
	population[(blockIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (d_numberAttributes-1)*2*d_populationSize + d_populationSize + threadIdx.x] = se * sp;
}

/**** DIFFERENTIAL EVOLUTION: MUTATION, CROSSOVER ***/
__global__ void kernel_Generation(curandState* globalState, float* population, float* offspring, float* minValue, float* maxValue, int* elitistIndividuals)
{
	int generatorID = (blockIdx.x * d_numberClasses * d_numberAttributes * d_populationSize) + (blockIdx.y * d_numberAttributes * d_populationSize) + (blockIdx.z * d_populationSize) + threadIdx.x;
	int gen = (blockIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (blockIdx.z * 2 * d_populationSize) + threadIdx.x;

	int a = rndInt(globalState, generatorID, d_populationSize);
	int b = rndInt(globalState, generatorID, d_populationSize);

	while(a == threadIdx.x){a = rndInt(globalState, generatorID, d_populationSize);}
	while(b == a && b == threadIdx.x){b=rndInt(globalState, generatorID, d_populationSize);}

	// DE mutation and crossover
	if(d_crossoverProbability > rndFloat(globalState, generatorID))
	{
		// DE crossover rand-best
		unsigned int agen = (blockIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (blockIdx.z * 2 * d_populationSize) + a;
		unsigned int bgen = (blockIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (blockIdx.z * 2 * d_populationSize) + b;
		unsigned int bestgen = (blockIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (blockIdx.z * 2 * d_populationSize) + elitistIndividuals[blockIdx.x * d_numberClasses + blockIdx.y];

		offspring[gen] = max(population[gen] + d_F*(population[bestgen] - population[gen]) + d_F*(population[agen] - population[bgen]), minValue[blockIdx.z]);
		offspring[gen + blockDim.x] = min(population[gen + blockDim.x] + d_F*(population[bestgen + blockDim.x] - population[gen + blockDim.x]) + d_F*(population[agen + blockDim.x] - population[bgen + blockDim.x]), maxValue[blockIdx.z]);
	}
	else
	{
		// DE reproduction
		offspring[gen] = population[gen];
		offspring[gen + blockDim.x] = population[gen + blockDim.x];
	}
}

/**** DIFFERENTIAL EVOLUTION: SELECTION ***/
__global__ void kernel_Selection(float *population, float *offspring)
{
	unsigned int gen = (blockIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (blockIdx.z * 2 * d_populationSize) + threadIdx.x;
	unsigned int fitnessIndex = (blockIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (d_numberAttributes-1)*2*d_populationSize + d_populationSize + threadIdx.x;

	//printf("%d %d %d: %f %f\n", blockIdx.x, blockIdx.y, threadIdx.x, population[fitnessIndex], offspring[fitnessIndex]);

	if(offspring[fitnessIndex] > population[fitnessIndex])
	{
		population[gen] = offspring[gen];
		population[gen + blockDim.x] = offspring[gen + blockDim.x];
	}
}

__global__ void kernel_BuildClassifier(float *population, float *classifier, int* elitistIndividuals)
{
	float bestFitness = -FLT_MAX;
	unsigned int bestIndividual = 0;

	for(int i = 0; i < d_populationSize; i++)
	{
		if(population[(blockIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (d_numberAttributes-1)*2*d_populationSize + d_populationSize + i] > bestFitness)
		{
			bestFitness = population[(blockIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (d_numberAttributes-1)*2*d_populationSize + d_populationSize + i];
			bestIndividual = i;
		}
	}

	classifier[(blockIdx.y * d_numberRulesClass * d_genotypeLength) + (blockIdx.x * d_genotypeLength) + 2*threadIdx.x    ] = population[(blockIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (threadIdx.x * 2 * d_populationSize) + bestIndividual];
	classifier[(blockIdx.y * d_numberRulesClass * d_genotypeLength) + (blockIdx.x * d_genotypeLength) + 2*threadIdx.x + 1] = population[(blockIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (blockIdx.y * d_genotypeLength * d_populationSize) + (threadIdx.x * 2 * d_populationSize) + d_populationSize + bestIndividual];

	if(threadIdx.x == 0)
		elitistIndividuals[blockIdx.x * d_numberClasses + blockIdx.y] = bestIndividual;
}

__global__ void kernel_IdentifyElitist(float *population, int* elitistIndividuals)
{
	float bestFitness = -FLT_MAX;
	unsigned int bestIndividual = 0;

	for(int i = 0; i < d_populationSize; i++)
	{
		if(population[(threadIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (threadIdx.y * d_genotypeLength * d_populationSize) + (d_numberAttributes-1)*2*d_populationSize + d_populationSize + i] > bestFitness)
		{
			bestFitness = population[(threadIdx.x * d_numberClasses * d_genotypeLength * d_populationSize) + (threadIdx.y * d_genotypeLength * d_populationSize) + (d_numberAttributes-1)*2*d_populationSize + d_populationSize + i];
			bestIndividual = i;
		}
	}

	elitistIndividuals[threadIdx.x * d_numberClasses + threadIdx.y] = bestIndividual;
}

JNIEXPORT void JNICALL Java_moa_classifiers_evolutionary_DERules_contextualize
(JNIEnv *env, jobject obj, jint jseed, jint jpopulationSize, jint jnumberGenerations, jint jnumberRulesClass, jint jnumberInstances, jint jnumberAttributes, jint jnumberClasses)
{
	seed = jseed;
	h_populationSize = jpopulationSize;
	numberGenerations = jnumberGenerations;
	h_numberRulesClass = jnumberRulesClass;
	h_numberInstances = jnumberInstances;
	h_numberAttributes = jnumberAttributes;
	h_numberClasses = jnumberClasses;
	h_genotypeLength = 2*h_numberAttributes; // [min,max] pairs for each attribute + number active conditions + fitness value
	elitism = false;

	cudaMalloc((void**)&d_dataset,    h_numberInstances  * h_numberAttributes * sizeof(float));
	cudaMalloc((void**)&d_minValue,   h_numberAttributes * sizeof(float));
	cudaMalloc((void**)&d_maxValue,   h_numberAttributes * sizeof(float));
	cudaMalloc((void**)&d_devStates,  h_numberRulesClass * h_numberClasses * h_numberAttributes * h_populationSize * sizeof(curandState));
	cudaMalloc((void**)&d_coverage,   h_numberRulesClass * h_numberClasses * h_numberInstances  * h_populationSize * sizeof(unsigned char));
	cudaMalloc((void**)&d_population, h_numberRulesClass * h_numberClasses * h_populationSize * h_genotypeLength * sizeof(float));
	cudaMalloc((void**)&d_offspring,  h_numberRulesClass * h_numberClasses * h_populationSize * h_genotypeLength * sizeof(float));
	cudaMalloc((void**)&d_classifier, h_numberRulesClass * h_numberClasses * h_genotypeLength * sizeof(float));
	cudaMalloc((void**)&d_instance,   h_numberAttributes * sizeof(float));
	cudaMalloc((void**)&d_votes,      h_numberClasses    * sizeof(float));
	cudaMalloc((void**)&d_elitistIndividuals,   h_numberRulesClass * h_numberClasses * sizeof(int));

	cudaMallocHost((void**)&h_population, h_numberRulesClass * h_numberClasses * h_populationSize * h_genotypeLength * sizeof(float));
	cudaMallocHost((void**)&h_classifier, h_numberRulesClass * h_numberClasses * h_genotypeLength * sizeof(float));
	cudaMallocHost((void**)&h_minValue, h_numberAttributes * sizeof(float));
	cudaMallocHost((void**)&h_maxValue, h_numberAttributes * sizeof(float));

	cudaMemcpyToSymbol(d_numberRulesClass, &h_numberRulesClass, sizeof(int));
	cudaMemcpyToSymbol(d_numberAttributes, &h_numberAttributes, sizeof(int));
	cudaMemcpyToSymbol(d_numberClasses,    &h_numberClasses, sizeof(int));
	cudaMemcpyToSymbol(d_populationSize,   &h_populationSize, sizeof(int));
	cudaMemcpyToSymbol(d_genotypeLength,   &h_genotypeLength, sizeof(int));
	cudaMemcpyToSymbol(d_crossoverProbability, &h_crossoverProbability, sizeof(float));
	cudaMemcpyToSymbol(d_F, &h_F, sizeof(float));

	unsigned int setup_size = h_numberRulesClass * h_numberClasses * h_populationSize * h_numberAttributes;
	setup_kernel <<< (int) ceil(setup_size / (float) THREAD_BLOCK) , THREAD_BLOCK>>> (d_devStates, seed, setup_size);
}

JNIEXPORT void JNICALL Java_moa_classifiers_evolutionary_DERules_copyDataset
(JNIEnv *env, jobject obj, jobject algorithm, jint jnumberInstances)
{
	h_numberInstances = jnumberInstances;

	for(int j = 0; j < h_numberAttributes; j++)
	{
		h_minValue[j] = FLT_MAX;
		h_maxValue[j] = -FLT_MAX;
	}

	jclass cls = env->GetObjectClass(algorithm);
	jmethodID getDataset = env->GetMethodID(cls, "getDataset", "()[F");
	jfloatArray datasetArray = (jfloatArray) env->CallObjectMethod(algorithm, getDataset);
	float *dataset = (float*) env->GetFloatArrayElements(datasetArray, 0);

	for(int i = 0; i < h_numberInstances; i++)
	{
		for(int j = 0; j < h_numberAttributes; j++)
		{
			float val = dataset[i*h_numberAttributes + j];
			h_minValue[j] = val < h_minValue[j] ? val : h_minValue[j];
			h_maxValue[j] = val > h_maxValue[j] ? val : h_maxValue[j];
		}
	}

	cudaMemcpy(d_dataset,  dataset,    h_numberInstances  * h_numberAttributes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_minValue, h_minValue, h_numberAttributes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_maxValue, h_maxValue, h_numberAttributes * sizeof(float), cudaMemcpyHostToDevice);

	env->ReleaseFloatArrayElements(datasetArray, dataset, 0);
}

JNIEXPORT jfloatArray JNICALL Java_moa_classifiers_evolutionary_DERules_train
(JNIEnv *env, jobject obj)
{
	dim3 grid_initialization(h_numberRulesClass, h_numberClasses, h_numberAttributes-1);
	kernel_Initialization <<< grid_initialization , h_populationSize >>> (d_devStates, d_population, d_minValue, d_maxValue, d_elitistIndividuals, elitism);

	cudaMemset(d_coverage, 0, h_numberRulesClass * h_numberClasses * h_numberInstances * h_populationSize * sizeof(unsigned char));

	dim3 grid_coverage(h_numberInstances, h_numberClasses, h_numberAttributes-1);

	for(int i = 0; i < h_numberRulesClass; i++)
		kernel_Coverage <<< grid_coverage , h_populationSize >>> (d_population, d_dataset, d_coverage, h_numberInstances, i);

	dim3 grid_fitness(h_numberRulesClass, h_numberClasses);

	kernel_Fitness <<< grid_fitness , h_populationSize >>> (d_population, d_dataset, d_coverage, h_numberInstances);

	dim3 block_Elitist(h_numberRulesClass, h_numberClasses);

	kernel_IdentifyElitist <<< 1 , block_Elitist >>> (d_population, d_elitistIndividuals);

	dim3 grid_generation(h_numberRulesClass, h_numberClasses, h_numberAttributes-1);
	dim3 grid_selection(h_numberRulesClass, h_numberClasses, h_numberAttributes);

	/*printf("\nDataset\n");
	for(int i = 0; i < h_numberAttributes-1; i++)
	{
		printf("%d [%.3f,%.3f] ", i, h_minValue[i], h_maxValue[i]);
	}
	printf("\n");*/

	for(int generation = 0; generation < numberGenerations; generation++)
	{
		kernel_Generation <<< grid_generation , h_populationSize >>> (d_devStates, d_population, d_offspring, d_minValue, d_maxValue, d_elitistIndividuals);

		cudaMemset(d_coverage, 0, h_numberRulesClass * h_numberClasses * h_numberInstances * h_populationSize * sizeof(unsigned char));

		for(int i = 0; i < h_numberRulesClass; i++)
			kernel_Coverage <<< grid_coverage , h_populationSize >>> (d_offspring, d_dataset, d_coverage, h_numberInstances, i);

		kernel_Fitness <<< grid_fitness , h_populationSize >>> (d_offspring, d_dataset, d_coverage, h_numberInstances);

		kernel_Selection <<< grid_selection , h_populationSize >>> (d_population, d_offspring);

		/*dim3 grid_buildClassifier(h_numberRulesClass, h_numberClasses);
		kernel_BuildClassifier <<< grid_buildClassifier , h_numberAttributes >>> (d_population, d_classifier, d_elitistIndividuals);
		cudaMemcpy(h_classifier, d_classifier, h_numberClasses * h_numberRulesClass * h_genotypeLength * sizeof(float),cudaMemcpyDeviceToHost);
		printf("\nClassifier Generation %d\n", generation);
		for(int predictedClass = 0; predictedClass < h_numberClasses; predictedClass++)
		{
			printf("Class: %d\n", predictedClass);
			for(int rule = 0; rule < h_numberRulesClass; rule++)
			{
				printf("Rule: ");
				for(int i = 0; i < h_numberAttributes-1; i++)
					if(h_classifier[predictedClass*h_numberRulesClass*h_genotypeLength + rule*h_genotypeLength + 2*i] <= h_classifier[predictedClass*h_numberRulesClass*h_genotypeLength + rule*h_genotypeLength + 2*i + 1])
						printf("%d [%.3f,%.3f] ", i, h_classifier[predictedClass*h_numberRulesClass*h_genotypeLength + rule*h_genotypeLength + 2*i], h_classifier[predictedClass*h_numberRulesClass*h_genotypeLength + rule*h_genotypeLength + 2*i + 1]);
				printf(" Fitness: %.3f\n", h_classifier[predictedClass*h_numberRulesClass*h_genotypeLength + rule*h_genotypeLength + h_genotypeLength-1]);
			}
		}*/

		kernel_IdentifyElitist <<< 1 , block_Elitist >>> (d_population, d_elitistIndividuals);
	}

	dim3 grid_buildClassifier(h_numberRulesClass, h_numberClasses);

	kernel_BuildClassifier <<< grid_buildClassifier , h_numberAttributes >>> (d_population, d_classifier, d_elitistIndividuals);

	cudaMemcpy(h_classifier, d_classifier, h_numberClasses * h_numberRulesClass * h_genotypeLength * sizeof(float),cudaMemcpyDeviceToHost);

	cudaCheckError();

	if (classifierArrayGlobal == NULL)
	{
		jfloatArray classifierArray;
		classifierArray = env->NewFloatArray(h_numberClasses * h_numberRulesClass * h_genotypeLength);
		classifierArrayGlobal = (jfloatArray) env->NewGlobalRef(classifierArray);
	}

    env->SetFloatArrayRegion(classifierArrayGlobal, 0, h_numberClasses * h_numberRulesClass * h_genotypeLength, h_classifier);

	elitism = true;

	return classifierArrayGlobal;

	/*printf("\nDataset\n");
    for(int i = 0; i < h_numberAttributes-1; i++)
    {
    	printf("%d [%.3f,%.3f] ", i, h_minValue[i], h_maxValue[i]);
    }
    printf("\n");

    printf("\nClassifier\n");
    for(int predictedClass = 0; predictedClass < h_numberClasses; predictedClass++)
    {
    	printf("Class: %d\n", predictedClass);

    	for(int rule = 0; rule < h_numberRulesClass; rule++)
    	{
    		printf("Rule: ");

            for(int i = 0; i < h_numberAttributes-1; i++)
                if(h_classifier[predictedClass*h_numberRulesClass*h_genotypeLength + rule*h_genotypeLength + 2*i] <= h_classifier[predictedClass*h_numberRulesClass*h_genotypeLength + rule*h_genotypeLength + 2*i + 1])
                    printf("%d [%.3f,%.3f] ", i, h_classifier[predictedClass*h_numberRulesClass*h_genotypeLength + rule*h_genotypeLength + 2*i], h_classifier[predictedClass*h_numberRulesClass*h_genotypeLength + rule*h_genotypeLength + 2*i + 1]);

            printf(" Fitness: %.3f\n", h_classifier[predictedClass*h_numberRulesClass*h_genotypeLength + rule*h_genotypeLength + h_genotypeLength-1]);
    	}
    }

    printf("\n");
    fflush(stdout);

    cudaCheckError();*/
}

JNIEXPORT void JNICALL Java_moa_classifiers_evolutionary_DERules_release
(JNIEnv *env, jobject obj)
{
	cudaFree(d_dataset);
	cudaFree(d_coverage);
	cudaFree(d_minValue);
	cudaFree(d_maxValue);
	cudaFree(d_population);
	cudaFree(d_offspring);
	cudaFree(d_instance);
	cudaFree(d_elitistIndividuals);
	cudaFree(d_devStates);

	cudaFreeHost(h_population);
	cudaFreeHost(h_classifier);
	cudaFreeHost(h_minValue);
	cudaFreeHost(h_maxValue);
}
