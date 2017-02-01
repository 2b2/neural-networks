package de.ef.neuralnetworks.util;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;
import java.util.function.DoubleFunction;

import de.ef.neuralnetworks.NeuralNetwork;

public final class NeuralNetworkTraining{
	
	private NeuralNetworkTraining(){}
	
	
	public final static int DEFAULT_VALIDATION_PERCENT = 20, MAX_VALIDATION_PERCENT = 50;
	
	
	
	public static void train(
			NeuralNetwork network, Map<double[], double[]> dataSets,
			DoubleFunction<Boolean> completed) throws IOException{
		
		NeuralNetworkTraining.train(network, dataSets, completed, DEFAULT_VALIDATION_PERCENT);
	}
	
	public static void train(
			NeuralNetwork network, Map<double[], double[]> dataSet,
			DoubleFunction<Boolean> completed, int validationPercent) throws IOException{
		
		if(validationPercent < 0 || validationPercent > MAX_VALIDATION_PERCENT)
			throw new IllegalArgumentException("Validation percentage not possible: " + validationPercent);
		int validationSize = (int)(dataSet.size() * (validationPercent / 100.0));
		
		List<Entry<double[], double[]>> dataSetList = new ArrayList<>(dataSet.size());
		for(Entry<double[], double[]> entry : dataSet.entrySet())
			dataSetList.add(entry);
		
		List<Entry<double[], double[]>> trainingSet = null, validationSet = null;
		
		Random random = new Random();
		double totalError;
		long index = 0;
		do{
			// TODO make times to shuffle configurable
			if(index++ % 200 == 0){
				Collections.shuffle(dataSetList, random);
				validationSet = dataSetList.subList(0, validationSize);
				trainingSet = dataSetList.subList(validationSize, dataSetList.size());
			}
			
			// train
			for(Entry<double[], double[]> entry : trainingSet)
				network.train(entry.getKey(), entry.getValue());
			
			// validate
			totalError = 0;
			for(Entry<double[], double[]> entry : validationSet){
				double outputs[] = network.calculate(entry.getKey());
				double expectedOutputs[] = entry.getValue();
				
				double error = 0;
				for(int i = 0; i < outputs.length; i++)
					error += expectedOutputs[i] - outputs[i];
				totalError += error / outputs.length;
			}
		}while(completed.apply(totalError) == false);
	}
}
