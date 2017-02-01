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
import java.util.function.BiFunction;
import java.util.function.Predicate;

/**
 * {@code NeuralNetworkComparator} is a {@link java.util.Comparator Comparator}
 * which uses {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetworks} to
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
 * @param I type of values to compare
 * 
 * @author Erik Fritzsche
 * @version 2.0
 * @since 1.0
 */
public class NeuralNetworkComparator<I>
	implements Comparator<I>{
	
	/**
	 * Constant to indicate the result of the comparison, the closer the
	 * output of the neural network the more likely the result.
	 */
	private final static float NOT_EQUAL = 0f, EQUAL = 1f, LESS_THAN = 0f, GREATER_THAN = 1f;
	
	/**
	 * Constant to indicate the cutoff point of the result.
	 */
	private final static float E_THRESHOLD = 0.5f, GT_THRESHOLD = 0.5f;
	
	
	
	private final NeuralNetwork<Object, Float> equal, compare;
	private final BiFunction<I, I, Object> inputConverter;
	
	
	/**
	 * Constructs a new {@code NeuralNetworkComparator}.
	 * 
	 * @param equal tests two inputs for equality (a output of {@code 1} means equal)
	 * @param compare checks for the bigger value (a output of {@code 1} means the first value is bigger than the second)
	 * @param inputConverter combines two inputs into one object consumable by {@code equal} and {@code compare}
	 */
	@SuppressWarnings("unchecked")
	public <C> NeuralNetworkComparator(NeuralNetwork<C, Float> equal, NeuralNetwork<C, Float> compare, BiFunction<I, I, C> inputConverter){
		this.equal = (NeuralNetwork<Object, Float>)equal;
		this.compare = (NeuralNetwork<Object, Float>)compare;
		
		this.inputConverter = (BiFunction<I, I, Object>)inputConverter;
	}
	
	
	@Override
	public int compare(I a, I b){
		try{
			Object input = this.inputConverter.apply(a, b); 
			
			if(this.equal.calculate(input) > E_THRESHOLD)
				return 0;
			
			if(this.compare.calculate(input) > GT_THRESHOLD)
				return 1;
			return -1;
		}
		catch(IOException e){
			throw new RuntimeException(e);
		}
	}
	
	
	public void train(Map<Long, ? extends Collection<I>> trainingSets, Predicate<Double> completed) throws IOException{
		List<TrainingData> allData = new LinkedList<>();
		
		int totalCount = 0;
		for(Entry<Long, ? extends Collection<I>> container : trainingSets.entrySet())
			for(Iterator<I> i = container.getValue().iterator(); i.hasNext(); totalCount++)
				allData.add(new TrainingData(i.next(), container.getKey()));
		
		List<OrderedPair> allPairs, trainingPairs = null, validationPairs = null;
		
		allPairs = new ArrayList<>(totalCount * totalCount);
		for(TrainingData a : allData)
			for(TrainingData b : allData)
				allPairs.add(new OrderedPair(a, b));

		// TODO make validation percentage configurable
		int validationSize = (int)(allPairs.size() * (20 / 100.0));
		
		Map<String, Float> expected = new HashMap<>(allPairs.size());
		
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
		}while(completed.test(error) == false);
	}
	
	protected double train(Collection<OrderedPair> pairs, Map<String, Float> expected) throws IOException{
		double totalError = 0;
		
		int count = 0;
		for(OrderedPair pair : pairs){
			count++;
			if(pair.same == true)
				totalError += this.equal.train(pair.data, EQUAL);
			else{
				totalError += this.equal.train(pair.data, NOT_EQUAL);
				count++;
				
				Float expectedResult = expected.get(pair.id);
				if(expectedResult == null){
					// if expected result is null, reversed result is too
					expectedResult = this.compare.calculate(pair.data);
					if(expectedResult > GT_THRESHOLD)
						totalError += this.compare.train(pair.data, GREATER_THAN);
					else
						totalError += this.compare.train(pair.data, LESS_THAN);
					
					expectedResult = this.compare.calculate(pair.data); // recalculate after training
					expected.put(pair.id, expectedResult);
					expected.put(pair.reverseId, 1 - expectedResult);
				}
				else{
					// if expected result is not null, reversed result is too
					expectedResult =
						(expectedResult + this.compare.calculate(pair.data)) / 2;
					
					Float reversedResult = expected.get(pair.reverseId);
					
					if(expectedResult > GT_THRESHOLD && reversedResult <= GT_THRESHOLD)
						totalError += this.compare.train(pair.data, GREATER_THAN);
					else if(expectedResult <= GT_THRESHOLD && reversedResult > GT_THRESHOLD)
						totalError += this.compare.train(pair.data, LESS_THAN);
					else if(expectedResult > GT_THRESHOLD && reversedResult > GT_THRESHOLD)
						totalError += this.compare.train(
							pair.data,
							expectedResult < reversedResult ? LESS_THAN : GREATER_THAN
						);
					else
						totalError += this.compare.train(
							pair.data,
							expectedResult > reversedResult ? GREATER_THAN : LESS_THAN
						);
					
					expectedResult =
						(expected.get(pair.id) + this.compare.calculate(pair.data)) / 2; // recalculate after training
					expected.put(pair.id, expectedResult);
					expected.put(pair.reverseId, (reversedResult + (1 - expectedResult)) / 2);
				}
			}
		}
		return totalError / count;
	}
	
	protected double validate(Collection<OrderedPair> pairs, Map<String, Float> expected) throws IOException{
		double totalError = 0;
		for(OrderedPair pair : pairs){
			if(pair.same == true)
				totalError += (EQUAL - this.equal.calculate(pair.data));
			else{
				Float expectedResult = expected.get(pair.id);
				// if for some reason there is no expected result for this pair
				// skip and add highest error to total-error
				if(expectedResult == null){
					totalError += 1.0;
					continue;
				}
				if(expectedResult > GT_THRESHOLD)
					totalError += (GREATER_THAN - this.compare.calculate(pair.data));
				else
					totalError += (this.compare.calculate(pair.data));
			}
		}
		return totalError / pairs.size();
	}
	
	
	
	private class TrainingData{
		
		private final I data;
		private final long containerId;
		
		
		public TrainingData(I data, long containerId){
			this.data = data;
			this.containerId = containerId;
		}
	}
	
	private class OrderedPair{
		
		private final String id, reverseId;
		private final boolean same;
		private final Object data;
		
		
		public OrderedPair(TrainingData a, TrainingData b){
			this.data =
				NeuralNetworkComparator.this.inputConverter.apply(a.data, b.data);
			
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
		@SuppressWarnings("unchecked")
		public boolean equals(Object other){
			return this.id == ((OrderedPair)other).id;
		}
	}
}
