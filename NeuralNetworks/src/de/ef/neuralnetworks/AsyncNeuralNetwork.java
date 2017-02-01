package de.ef.neuralnetworks;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;

/**
 * The class {@code AsyncNeuralNetwork} makes a
 * {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork}
 * asynchronously accessible.
 * <p>
 * The following configurations are possible:
 * <ul>
 * <li>a single neural-network and a single producer thread</li>
 * <li>a single neural-network<b>*</b> and {@code n} producer threads</li>
 * <li>{@code n} neural-networks and {@code n} producer threads</li>
 * </ul>
 * <b>*</b>: In this configuration the neural-network has to be thread-safe.
 * </p>
 * 
 * @param I input type
 * @param O output type
 * 
 * @author Erik Fritzsche
 * @version 2.0
 * @since 1.0
 */
public class AsyncNeuralNetwork<I, O>{
	
	private final NeuralNetwork<I, O>[] networks;
	private final BlockingQueue<AsyncDataContainer> queue;
	private final ExecutorService producer;
	
	
	/**
	 * Configures the {@code AsyncNeuralNetworks} to have
	 * one neural-network and one producer thread.
	 * 
	 * @param network the asynchronously accessed
	 * {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork}
	 */
	public AsyncNeuralNetwork(NeuralNetwork<I, O> network){
		this(network, 1);
	}
	
	/**
	 * Configures the {@code AsyncNeuralNetworks} to have
	 * one neural-network and {@code producerCount} producer threads.
	 * <p>
	 * <b>Important:</b> The implementation  of the
	 * {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork}
	 * <u>must</u> be <b>thread-safe</b>.
	 * </p>
	 * 
	 * @param network the asynchronously accessed <u>thread-safe</u>
	 * {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork}
	 * @param producerCount number of used producer threads
	 */
	@SuppressWarnings("unchecked")
	public AsyncNeuralNetwork(NeuralNetwork<I, O> network, int producerCount){
		this(new NeuralNetwork[]{network}, producerCount);
	}
	
	/**
	 * Configures the {@code AsyncNeuralNetworks} to have
	 * {@code networks.length} neural-networks and
	 * {@code networks.length} producer threads.
	 * 
	 * @param networks the asynchronously accessed
	 * {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetworks}
	 */
	public AsyncNeuralNetwork(NeuralNetwork<I, O>[] networks){
		this(networks, networks.length);
	}
	
	// internal constructor and configuration
	private AsyncNeuralNetwork(NeuralNetwork<I, O>[] networks, int producerCount){
		this.networks = networks;
		this.queue = new LinkedBlockingQueue<AsyncDataContainer>();
		this.producer = Executors.newFixedThreadPool(
			producerCount,
			new ThreadFactory(){
				int index = 0;
				@Override
				public Thread newThread(Runnable r){
					Thread t = new Thread(r, "async-neural-network-producer-" + (index++));
					t.setDaemon(true);
					return t;
				}
			}
		);
		// start up all producer thread
		for(int i = 0; i < producerCount; i++){
			NeuralNetwork<I, O> network = this.networks[i % this.networks.length];
			this.producer.execute(
				() -> this.asyncCalculate(network)
			);
		}
	}
	
	
	/**
	 * Asynchronously invokes the
	 * {@link de.ef.neuralnetworks.NeuralNetwork#calculate(double[]) NeuralNetwork.calculate}
	 * function.
	 * 
	 * @param input the state of the neurons inside the first layer
	 * 
	 * @return a {@link java.util.concurrent.Future Future} with
	 * the output state of the neurons inside the last layer
	 */
	public Future<O> calculate(I input){
		CompletableFuture<O> outputFuture = new CompletableFuture<O>();
		this.queue.add(new AsyncDataContainer(input, outputFuture));
		// TODO may hide that Future<Double[]> is CompletableFuture<Double[]>
		return (Future<O>)outputFuture;
	}
	
	
	// TODO may overwork
	private void asyncCalculate(NeuralNetwork<I, O> network){
		while(this.producer.isShutdown() == false){
			AsyncDataContainer container;
			try{
				container = this.queue.take();
				
				O output = network.calculate(container.input);
				container.outputFuture.complete(output);
			}catch(Throwable t){
				this.producer.shutdown();
				throw new RuntimeException(t);
			}
		}
	}
	
	
	
	private class AsyncDataContainer{
		
		private final I input;
		private final CompletableFuture<O> outputFuture;
		
		
		public AsyncDataContainer(I input, CompletableFuture<O> outputFuture){
			this.input = input;
			this.outputFuture = outputFuture;
		}
	}
}
