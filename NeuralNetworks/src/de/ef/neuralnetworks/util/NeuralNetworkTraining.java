package de.ef.neuralnetworks.util;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;
import java.util.function.BiFunction;
import java.util.function.Predicate;

import de.ef.neuralnetworks.NeuralNetwork;

public final class NeuralNetworkTraining{
	
	private NeuralNetworkTraining(){}
	
	
	public final static int DEFAULT_VALIDATION_PERCENT = 20, MAX_VALIDATION_PERCENT = 50;
	
	
	
	public static <I, O> void train(
			NeuralNetwork<I, O> network, Map<I, O> dataSets,
			BiFunction<O, O, Double> errorCalculator, Predicate<Double> completed) throws IOException{
		
		NeuralNetworkTraining.train(network, dataSets, errorCalculator, completed, DEFAULT_VALIDATION_PERCENT);
	}
	
	public static <I, O> void train(
			NeuralNetwork<I, O> network, Map<I, O> dataSet,
			BiFunction<O, O, Double> errorCalculator, Predicate<Double> completed,
			int validationPercent) throws IOException{
		
		if(validationPercent < 0 || validationPercent > MAX_VALIDATION_PERCENT)
			throw new IllegalArgumentException("Validation percentage not possible: " + validationPercent);
		int validationSize = (int)(dataSet.size() * (validationPercent / 100.0));
		
		List<Entry<I, O>> dataSetList = new ArrayList<>(dataSet.size());
		for(Entry<I, O> entry : dataSet.entrySet())
			dataSetList.add(entry);
		
		List<Entry<I, O>> trainingSet = null, validationSet = null;
		
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
			for(Entry<I, O> entry : trainingSet)
				network.train(entry.getKey(), entry.getValue());
			
			// validate
			totalError = 0;
			for(Entry<I, O> entry : validationSet){
				O output = network.calculate(entry.getKey());
				O expectedOutput = entry.getValue();
				
				totalError += errorCalculator.apply(output, expectedOutput);
			}
		}while(completed.test(totalError) == false);
	}
}
