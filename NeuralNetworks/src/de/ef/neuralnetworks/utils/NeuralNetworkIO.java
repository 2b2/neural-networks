package de.ef.neuralnetworks.utils;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;

import de.ef.neuralnetworks.NeuralNetwork;
import de.ef.neuralnetworks.NeuralNetworkConstructor;

// TODO comment
// version: 1.1, date: 07.06.2016, author: Erik Fritzsche
public final class NeuralNetworkIO{
	
	private NeuralNetworkIO(){}
	
	
	
	public static void write(NeuralNetwork network, OutputStream output) throws IOException{
		write(network, true, output);
	}
	
	public static void write(NeuralNetwork network, boolean doublePrecision, OutputStream output) throws IOException{
		DataOutputStream dataOutput = new DataOutputStream(output);
		
		boolean bias = network.hasBias();
		byte init = 0;
		if(doublePrecision == true){
			init |= 0b0001;
		}
		if(bias == true){
			init |= 0b0010;
		}
		dataOutput.writeByte(init);
		
		dataOutput.writeInt(network.getLayerCount());
		for(int layer = 0; layer < network.getLayerCount(); layer++){
			dataOutput.writeInt(network.getNeuronCount(layer));
		}
		
		int layerSize, lastLayerSize = -1;
		for(int layer = 0; layer < network.getLayerCount(); layer++){
			layerSize = network.getNeuronCount(layer);
			for(int neuron = 0; neuron < layerSize; neuron++){
				int weightCount = (lastLayerSize != -1 ? lastLayerSize + (bias == true ? 1 : 0) : 0);
				for(int weight = 0; weight < weightCount; weight++){
					if(doublePrecision == true){
						dataOutput.writeDouble(network.getWeight(layer, neuron, weight));
					}
					else{
						dataOutput.writeFloat((float)network.getWeight(layer, neuron, weight));
					}
				}
			}
			lastLayerSize = layerSize;
		}
		dataOutput.flush();
	}
	
	public static NeuralNetwork read(NeuralNetworkConstructor constructor, InputStream input) throws IOException{
		DataInputStream dataInput = new DataInputStream(input);
		
		byte init = dataInput.readByte();
		boolean doublePrecision = (init & 0b0001) == 0b0001;
		boolean bias = (init & 0b0010) == 0b0010;
		
		int layerCount = dataInput.readInt();
		int layerSizes[] = new int[layerCount];
		for(int layer = 0; layer < layerCount; layer++){
			layerSizes[layer] = dataInput.readInt();
		}
		
		NeuralNetwork network = constructor.createNeuralNetwork(
			layerSizes[0],
			Arrays.copyOfRange(layerSizes, 1, layerSizes.length - 1),
			layerSizes[layerSizes.length - 1]
		);
		
		int lastLayerSize = -1;
		for(int layer = 0; layer < layerCount; layer++){
			for(int neuron = 0; neuron < layerSizes[layer]; neuron++){
				int weightCount = (lastLayerSize != -1 ? lastLayerSize + (bias == true ? 1 : 0) : 0);
				for(int weight = 0; weight < weightCount; weight++){
					if(doublePrecision == true){
						network.setWeight(layer, neuron, weight, dataInput.readDouble());
					}
					else{
						network.setWeight(layer, neuron, weight, dataInput.readFloat());
					}
				}
			}
			lastLayerSize = layerSizes[layer];
		}
		
		return network;
	}
}