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
import java.util.function.Function;
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
 * @version 3.0
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
	
	
	
	private final ConvolutionalNeuralNetwork<Object, Object, Object> equal, compare;
	private final BiFunction<Object, Object, Object> inputConverter;
	private final Function<Object, Float> outputConverter;
	
	private final Object notEqualCache, equalCache, lessThanCache, greaterThanCache;
	
	
	/**
	 * Constructs a new {@code NeuralNetworkComparator}.
	 * 
	 * @param equal tests two inputs for equality (a output of {@code 1} means equal)
	 * @param compare checks for the bigger value (a output of {@code 1} means the first value is bigger than the second)
	 * @param inputConverter combines two inputs into one object consumable by {@code equal} and {@code compare}
	 */
	@SuppressWarnings("unchecked")
	public <C, O> NeuralNetworkComparator(
			ConvolutionalNeuralNetwork<I, C, O> equal, ConvolutionalNeuralNetwork<I, C, O> compare,
			BiFunction<C, C, C> inputConverter, Function<O, Float> outputConverter, Function<Float, O> reverseOutputConverter){
		this.equal = (ConvolutionalNeuralNetwork<Object, Object, Object>)equal;
		this.compare = (ConvolutionalNeuralNetwork<Object, Object, Object>)compare;
		
		this.inputConverter = (BiFunction<Object, Object, Object>)inputConverter;
		this.outputConverter = (Function<Object, Float>)outputConverter;

		this.notEqualCache    = reverseOutputConverter.apply(NOT_EQUAL);
		this.equalCache       = reverseOutputConverter.apply(EQUAL);
		this.lessThanCache    = reverseOutputConverter.apply(LESS_THAN);
		this.greaterThanCache = reverseOutputConverter.apply(GREATER_THAN);
	}
	
	
	@Override
	public int compare(I a, I b){
		try{
			Object aFiltered = this.equal.calculateFilters(a), bFiltered = this.equal.calculateFilters(b);
			Object input = this.inputConverter.apply(aFiltered, bFiltered); 
			
			if(this.outputConverter.apply(this.equal.calculateFullyConnected(input)) > E_THRESHOLD)
				return 0;
			
			aFiltered = this.compare.calculateFilters(a);
			bFiltered = this.compare.calculateFilters(b);
			input = this.inputConverter.apply(aFiltered, bFiltered);
			if(this.outputConverter.apply(this.compare.calculateFullyConnected(input)) > GT_THRESHOLD)
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
				totalError += this.equal.train(pair.data, this.equalCache);
			else{
				totalError += this.equal.train(pair.data, this.notEqualCache);
				count++;
				
				Float expectedResult = expected.get(pair.id);
				if(expectedResult == null){
					// if expected result is null, reversed result is too
					expectedResult = this.outputConverter.apply(this.compare.calculate(pair.data));
					if(expectedResult > GT_THRESHOLD)
						totalError += this.compare.train(pair.data, this.greaterThanCache);
					else
						totalError += this.compare.train(pair.data, this.lessThanCache);
					
					expectedResult = this.outputConverter.apply(this.compare.calculate(pair.data)); // recalculate after training
					expected.put(pair.id, expectedResult);
					expected.put(pair.reverseId, 1 - expectedResult);
				}
				else{
					// if expected result is not null, reversed result is too
					expectedResult =
						(expectedResult + this.outputConverter.apply(this.compare.calculate(pair.data))) / 2;
					
					Float reversedResult = expected.get(pair.reverseId);
					
					if(expectedResult > GT_THRESHOLD && reversedResult <= GT_THRESHOLD)
						totalError += this.compare.train(pair.data, this.greaterThanCache);
					else if(expectedResult <= GT_THRESHOLD && reversedResult > GT_THRESHOLD)
						totalError += this.compare.train(pair.data, this.lessThanCache);
					else if(expectedResult > GT_THRESHOLD && reversedResult > GT_THRESHOLD)
						totalError += this.compare.train(
							pair.data,
							expectedResult < reversedResult ? this.lessThanCache : this.greaterThanCache
						);
					else
						totalError += this.compare.train(
							pair.data,
							expectedResult > reversedResult ? this.greaterThanCache : this.lessThanCache
						);
					
					expectedResult =
						(
							expected.get(pair.id) + this.outputConverter.apply(this.compare.calculate(pair.data))
						) / 2; // recalculate after training
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
				totalError += (EQUAL - this.outputConverter.apply(this.equal.calculate(pair.data)));
			else{
				Float expectedResult = expected.get(pair.id);
				// if for some reason there is no expected result for this pair
				// skip and add highest error to total-error
				if(expectedResult == null){
					totalError += 1.0;
					continue;
				}
				if(expectedResult > GT_THRESHOLD)
					totalError += (GREATER_THAN - this.outputConverter.apply(this.compare.calculate(pair.data)));
				else
					totalError += this.outputConverter.apply(this.compare.calculate(pair.data));
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
