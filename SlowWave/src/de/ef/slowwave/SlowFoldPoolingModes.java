package de.ef.slowwave;

class SlowFoldPoolingModes{
	
	private SlowFoldPoolingModes(){}
	
	
	
	static float directOutputSum(
			float inputs[], float filter[],
			int lowerBoundX, int upperBoundX, int lowerBoundY, int upperBoundY,
			int x, int y, int width, int height, int size,
			int filterWidth, int filterHeight, int filterSize, int filterDepth,
			int filterWidthPadding, int filterHeightPadding){
		
		float outputSum = 0f;
		for(int fY = lowerBoundY, rY = y + (fY - filterHeightPadding); fY < upperBoundY; fY++, rY++){
			for(int fX = lowerBoundX, rX = x + (fX - filterWidthPadding); fX < upperBoundX; fX++, rX++){
				for(int fZ = 0; fZ < filterDepth; fZ++){
					outputSum +=
						inputs[rX + (rY * width) + (fZ * size)]
						* filter[fX + (fY * filterWidth) + (fZ * filterSize)];
				}
			}
		}
		return outputSum;
	}
	
	static float averageOutputSum(
			float inputs[], float filter[],
			int lowerBoundX, int upperBoundX, int lowerBoundY, int upperBoundY,
			int x, int y, int width, int height, int size,
			int filterWidth, int filterHeight, int filterSize, int filterDepth,
			int filterWidthPadding, int filterHeightPadding){
		
		float outputSum = 0f;
		for(int fY = lowerBoundY, rY = y + (fY - filterHeightPadding); fY < upperBoundY; fY++, rY++){
			for(int fX = lowerBoundX, rX = x + (fX - filterWidthPadding); fX < upperBoundX; fX++, rX++){
				for(int fZ = 0; fZ < filterDepth; fZ++){
					outputSum +=
						(
							inputs[2 * rX + (2 * rY * width) + (fZ * size)]
							+ inputs[2 * rX + ((2 * rY + 1) * width) + (fZ * size)]
							+ inputs[(2 * rX + 1) + (2 * rY * width) + (fZ * size)]
							+ inputs[(2 * rX + 1) + ((2 * rY + 1) * width) + (fZ * size)]
						) / 4
						* filter[fX + (fY * filterWidth) + (fZ * filterSize)];
				}
			}
		}
		return outputSum;
	}
	
	static float maximumOutputSum(
			float inputs[], float filter[],
			int lowerBoundX, int upperBoundX, int lowerBoundY, int upperBoundY,
			int x, int y, int width, int height, int size,
			int filterWidth, int filterHeight, int filterSize, int filterDepth,
			int filterWidthPadding, int filterHeightPadding){
		
		float outputSum = 0f;
		for(int fY = lowerBoundY, rY = y + (fY - filterHeightPadding); fY < upperBoundY; fY++, rY++){
			for(int fX = lowerBoundX, rX = x + (fX - filterWidthPadding); fX < upperBoundX; fX++, rX++){
				for(int fZ = 0; fZ < filterDepth; fZ++){
					outputSum +=
						Math.max(
							inputs[2 * rX + (2 * rY * width) + (fZ * size)],
							Math.max(
								inputs[2 * rX + ((2 * rY + 1) * width) + (fZ * size)],
								Math.max(
									inputs[(2 * rX + 1) + (2 * rY * width) + (fZ * size)],
									inputs[(2 * rX + 1) + ((2 * rY + 1) * width) + (fZ * size)]
								)
							)
						)
						* filter[fX + (fY * filterWidth) + (fZ * filterSize)];
				}
			}
		}
		return outputSum;
	}
	
	
	static float directOutputErrorSum(
			float inputs[], float errors[], int layerOffset,
			int lowerBoundX, int upperBoundX, int lowerBoundY, int upperBoundY,
			int offsetX, int offsetY, int offsetZ, int width, int height, int size){
		
		float outputErrorSum = 0f;
		for(int y = lowerBoundY; y < upperBoundY; y++){
			for(int x = lowerBoundX; x < upperBoundX; x++){
				outputErrorSum +=
					inputs[x + (y * width) + (offsetZ * size)]
					* errors[(x - offsetX) + ((y - offsetY) * width) + layerOffset];
			}
		}
		return outputErrorSum;
	}
	
	static float averageOutputErrorSum(
			float inputs[], float errors[], int layerOffset,
			int lowerBoundX, int upperBoundX, int lowerBoundY, int upperBoundY,
			int offsetX, int offsetY, int offsetZ, int height, int width, int size){
		
		float outputErrorSum = 0f;
		for(int y = lowerBoundY; y < upperBoundY; y++){
			for(int x = lowerBoundX; x < upperBoundX; x++){
				outputErrorSum +=
					(
						inputs[2 * x + (2 * y * width) + (offsetZ * size)]
						+ inputs[2 * x + ((2 * y + 1) * width) + (offsetZ * size)]
						+ inputs[(2 * x + 1) + (2 * y * width) + (offsetZ * size)]
						+ inputs[(2 * x + 1) + ((2 * y + 1) * width) + (offsetZ * size)]
					) / 4
					* errors[(x - offsetX) + ((y - offsetY) * width) + layerOffset];
			}
		}
		return outputErrorSum;
	}
	
	static float maximumOutputErrorSum(
			float inputs[], float errors[], int layerOffset,
			int lowerBoundX, int upperBoundX, int lowerBoundY, int upperBoundY,
			int offsetX, int offsetY, int offsetZ, int height, int width, int size){
		
		float outputErrorSum = 0f;
		for(int y = lowerBoundY; y < upperBoundY; y++){
			for(int x = lowerBoundX; x < upperBoundX; x++){
				outputErrorSum +=
					Math.max(
						inputs[2 * x + (2 * y * width) + (offsetZ * size)],
						Math.max(
							inputs[2 * x + ((2 * y + 1) * width) + (offsetZ * size)],
							Math.max(
								inputs[(2 * x + 1) + (2 * y * width) + (offsetZ * size)],
								inputs[(2 * x + 1) + ((2 * y + 1) * width) + (offsetZ * size)]
							)
						)
					)
					* errors[(x - offsetX) + ((y - offsetY) * width) + layerOffset];
			}
		}
		return outputErrorSum;
	}
}
