package de.ef.neuralnetworks;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.function.DoubleFunction;

/**
 * {@code NeuralNetworkComparator} is a {@link java.util.Comparator Comparator}
 * which uses a {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork} to
 * figure out if two values are the same or which one is greater.
 * <p>
 * The functionality of this class could be useful to for example
 * sort data into a tree structure to have faster access rates.
 * </p>
 * <p>
 * The calculations are made by two neural-networks that are provided
 * on instantiation. One has to issue if two inputs are the same and the
 * other has to find out if one input is less or more than the other. This class
 * itself just gives some functionality for the
 * {@link java.util.Comparator#compare compare} method and provides
 * training routines.
 * </p>
 * 
 * @param <T> the {@link de.ef.neuralnetworks.NeuralNetworkData NeuralNetworkData}
 *            type that has to be compared
 * 
 * @author Erik Fritzsche
 * @version 1.0
 * @since 1.0
 */
public class NeuralNetworkComparator<T extends NeuralNetworkData>
	implements Comparator<T>{
	
	/**
	 * Constant to indicate the result of the comparison, the closer the
	 * output of the neural network the more likely the result.
	 */
	public final static double NOT_EQUAL = 0.0, EQUAL = 1.0, LESS_THAN = 0.0, GREATER_THAN = 1.0;
	
	
	/**
	 * Constant to indicate the cutoff point of the result.
	 */
	private final static double E_THRESHOLD = 0.5, GT_THRESHOLD = 0.5;
	
	/**
	 * Internal cache so that the array is not created every time it is needed.
	 */
	private final static double
		NE_OUTPUT[] = {NOT_EQUAL}, E_OUTPUT[] = {EQUAL},
		LT_OUTPUT[] = {LESS_THAN}, GT_OUTPUT[] = {GREATER_THAN};
	
	
	
	private final NeuralNetwork equal, compare;
	
	
	public NeuralNetworkComparator(NeuralNetwork equal, NeuralNetwork compare){
		this.equal = equal;
		this.compare = compare;
	}
	
	
	@Override
	public int compare(T a, T b){
		try{
			double aData[] = a.getData(), bData[] = b.getData();
			double inputs[] = new double[aData.length + bData.length];
			
			System.arraycopy(aData, 0, inputs, 0, aData.length);
			System.arraycopy(bData, 0, inputs, aData.length, bData.length);
			
			if(this.equal.calculate(inputs)[0] > E_THRESHOLD)
				return 0;
			
			if(this.compare.calculate(inputs)[0] > GT_THRESHOLD)
				return 1;
			return -1;
		}
		catch(IOException e){
			throw new RuntimeException(e);
		}
	}
	
	
	public void train(Map<Long, ? extends Collection<T>> trainingSets, DoubleFunction<Boolean> completed) throws IOException{
		List<TrainingData> allData = new LinkedList<>();
		
		int totalCount = 0;
		for(Entry<Long, ? extends Collection<T>> container : trainingSets.entrySet())
			for(Iterator<T> i = container.getValue().iterator(); i.hasNext(); totalCount++)
				allData.add(new TrainingData(i.next().getData(), container.getKey()));
		
		List<OrderedPair> allPairs, trainingPairs = null, validationPairs = null;
		
		allPairs = new ArrayList<>(totalCount * totalCount);
		for(TrainingData a : allData)
			for(TrainingData b : allData)
				allPairs.add(new OrderedPair(a, b));

		// TODO make validation percentage configurable
		int validationSize = (int)(allPairs.size() * (20 / 100.0));
		
		Map<String, Double> expected = new HashMap<>(allPairs.size());
		
		Random random = new Random();
		double error;
		long index = 0;
		do{
			// TODO make times to shuffle configurable
			if(index++ % 200 == 0){
				Collections.shuffle(allPairs, random);
				validationPairs = allPairs.subList(0, validationSize);
				trainingPairs = allPairs.subList(validationSize, allPairs.size());
			}
			/* error = */this.train(trainingPairs, expected);
			error = this.validate(validationPairs, expected);
		}while(completed.apply(error) == false);
	}
	
	protected double train(Collection<OrderedPair> pairs, Map<String, Double> expected) throws IOException{
		double totalError = 0;
		
		int count = 0;
		for(OrderedPair pair : pairs){
			count++;
			if(pair.same == true)
				totalError += this.equal.train(pair.data, E_OUTPUT);
			else{
				totalError += this.equal.train(pair.data, NE_OUTPUT);
				count++;
				
				Double expectedResult = expected.get(pair.id);
				if(expectedResult == null){
					// if expected result is null, reversed result is too
					expectedResult = this.compare.calculate(pair.data)[0];
					if(expectedResult > GT_THRESHOLD)
						totalError += this.compare.train(pair.data, GT_OUTPUT);
					else
						totalError += this.compare.train(pair.data, LT_OUTPUT);
					
					expectedResult = this.compare.calculate(pair.data)[0]; // recalculate after training
					expected.put(pair.id, expectedResult);
					expected.put(pair.reverseId, 1 - expectedResult);
				}
				else{
					// if expected result is not null, reversed result is too
					expectedResult =
						(expectedResult + this.compare.calculate(pair.data)[0]) / 2;
					
					Double reversedResult = expected.get(pair.reverseId);
					
					if(expectedResult > GT_THRESHOLD && reversedResult <= GT_THRESHOLD)
						totalError += this.compare.train(pair.data, GT_OUTPUT);
					else if(expectedResult <= GT_THRESHOLD && reversedResult > GT_THRESHOLD)
						totalError += this.compare.train(pair.data, LT_OUTPUT);
					else if(expectedResult > GT_THRESHOLD && reversedResult > GT_THRESHOLD)
						totalError += this.compare.train(
							pair.data,
							expectedResult < reversedResult ? LT_OUTPUT : GT_OUTPUT
						);
					else
						totalError += this.compare.train(
							pair.data,
							expectedResult > reversedResult ? GT_OUTPUT : LT_OUTPUT
						);
					
					expectedResult =
						(expected.get(pair.id) + this.compare.calculate(pair.data)[0]) / 2; // recalculate after training
					expected.put(pair.id, expectedResult);
					expected.put(pair.reverseId, (reversedResult + (1 - expectedResult)) / 2);
				}
			}
		}
		return totalError / count;
	}
	
	protected double validate(Collection<OrderedPair> pairs, Map<String, Double> expected) throws IOException{
		double totalError = 0;
		for(OrderedPair pair : pairs){
			if(pair.same == true)
				totalError += (EQUAL - this.equal.calculate(pair.data)[0]);
			else{
				Double expectedResult = expected.get(pair.id);
				// if for some reason there is no expected result for this pair
				// skip and add highest error to total-error
				if(expectedResult == null){
					totalError += 1.0;
					continue;
				}
				if(expectedResult > GT_THRESHOLD)
					totalError += (GREATER_THAN - this.compare.calculate(pair.data)[0]);
				else
					totalError += (this.compare.calculate(pair.data)[0]);
			}
		}
		return totalError / pairs.size();
	}
	
	
	
	private static class TrainingData{
		
		private final double data[];
		private final long containerId;
		
		
		public TrainingData(double data[], long containerId){
			this.data = data;
			this.containerId = containerId;
		}
	}
	
	
	private static class OrderedPair{
		
		private final String id, reverseId;
		private final boolean same;
		private final double data[];
		
		
		public OrderedPair(TrainingData a, TrainingData b){
			this.data = new double[a.data.length + b.data.length];
			
			System.arraycopy(a.data, 0, this.data, 0, a.data.length);
			System.arraycopy(b.data, 0, this.data, a.data.length, b.data.length);
			
			this.id =
				Long.toHexString(a.containerId) + "#" + Long.toHexString(b.containerId);
			this.reverseId =
				Long.toHexString(b.containerId) + "#" + Long.toHexString(a.containerId);
			
			this.same = (a.containerId == b.containerId);
		}
		
		
		@Override
		public int hashCode(){
			// TODO cache hash if it has any performance benefit
			return this.id.hashCode();
		}
		
		@Override
		public boolean equals(Object other){
			if(other instanceof OrderedPair == false)
				return false;
			
			return this.id == ((OrderedPair)other).id;
		}
	}
}
