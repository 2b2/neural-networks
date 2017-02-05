
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
