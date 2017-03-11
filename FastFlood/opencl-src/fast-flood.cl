__kernel void calculateLayer(
		__global float *neurons, __global float *weights,
		__constant int *layerInfos, __constant int *neuronOffsets, const int layer){
	int gid = get_global_id(0);
	
	const int lastLayerSize = layerInfos[2 * (layer - 1)];
	const int lastLayerOffset = layerInfos[2 * (layer - 1) + 1];
	// neuron offset in weights array
	const int neuronOffset = neuronOffsets[layerInfos[2 * layer + 1] + gid];
	
	// calculate input sum for neuron
	float sum = 0;
	for(int i = 0; i < lastLayerSize; i++)
		sum += neurons[lastLayerOffset + i] * weights[neuronOffset + i];
	// bias neuron weight
	sum += weights[neuronOffset + lastLayerSize];
	// calculate output for neuron via sigmoid function
	neurons[layerInfos[2 * layer + 1] + gid] = 1 / (1 + powr(M_E_F, -sum));
}

__kernel void calculateFirstLayer(
		__global float *inputs, __global float *neurons, __global float *weights,
		__constant int *layerInfos, __constant int *neuronOffsets){
	int gid = get_global_id(0);
	
	const int inputLayerSize = layerInfos[0];
	// neuron offset in weights array
	const int neuronOffset = neuronOffsets[gid];
	
	// calculate input sum for neuron
	float sum = 0;
	for(int i = 0; i < inputLayerSize; i++)
		sum += inputs[i] * weights[neuronOffset + i];
	// bias neuron weight
	sum += weights[neuronOffset + inputLayerSize];
	// calculate output for neuron via sigmoid function
	neurons[gid] = 1 / (1 + powr(M_E_F, -sum));
}


__constant int RANDOM_MAX_VALUE = 1 << 16;
__constant float RANDOM_MAPPING = 1 / (RANDOM_MAX_VALUE * 0.5f);

__kernel void randomFloatArray(__global float *array, const int seed){
	int gid = get_global_id(0);
	
	// source for random:
	// http://stackoverflow.com/questions/3062746/special-simple-random-number-generator/3062783#3062783
	ushort random = ((uint)(seed + gid) * 1103515245 + 12345) % RANDOM_MAX_VALUE;
	
	// always between -1 and 1 (random times RANDOM_MAPPING is between 0 and 2)
	array[gid] = 1 - (random * RANDOM_MAPPING);
}
