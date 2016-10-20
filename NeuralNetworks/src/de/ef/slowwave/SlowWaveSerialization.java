package de.ef.slowwave;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

class SlowWaveSerialization{
	
	private SlowWaveSerialization(){}
	
	
	
	static void write(SlowWave network, ObjectOutputStream output) throws IOException{
		// write header byte with bit flags 0b0001 for double precision and 0b0010 for bias neuron
		output.writeByte(0b0011);
		
		SlowWave.Neuron[][] layers = network.getLayers();
		
		output.writeInt(layers.length); // write layer count as integer
		for(int layer = 0; layer < layers.length; layer++){
			output.writeInt(layers[layer].length); // write count of neurons in each layer as integer
		}
		
		int neuronCount, lastNeuronCount = -1;
		for(int layer = 0; layer < layers.length; layer++){
			neuronCount = layers[layer].length;
			for(int neuron = 0; neuron < neuronCount; neuron++){
				// plus one to lastNeuronCount for bias neuron
				int weightCount = (lastNeuronCount != -1 ? lastNeuronCount + 1 : 0);
				for(int weight = 0; weight < weightCount; weight++){
					output.writeDouble(layers[layer][neuron].getWeight(weight)); // write each weight as double
				}
			}
			lastNeuronCount = neuronCount;
		}
	}
	
	
	static SlowWave.Neuron[][] readLayers(ObjectInputStream input) throws IOException{
		byte header = input.readByte(); // read header byte
		// extract bit flags (double precision and bias neuron)
		// if one of the flags is not set correctly throw exception
		if((header & 0b0001) != 0b0001 || (header & 0b0010) != 0b0010){
			throw new IOException("Incorrect serialization header: " + header);
		}
		
		SlowWave.Neuron[][] layers = new SlowWave.Neuron[input.readInt()][]; // read layer count
		
		for(int layer = 0; layer < layers.length; layer++){
			layers[layer] = new SlowWave.Neuron[input.readInt()]; // read each neuron count
		}
		
		return layers;
	}
	
	static void read(SlowWave network, ObjectInputStream input) throws IOException{
		SlowWave.Neuron[][] layers = network.getLayers();
		
		int lastNeuronCount = -1;
		for(int layer = 0; layer < layers.length; layer++){
			for(int neuron = 0; neuron < layers[layer].length; neuron++){
				// plus one to lastNeuronCount for bias neuron
				int weightCount = (lastNeuronCount != -1 ? lastNeuronCount + 1 : 0);
				for(int weight = 0; weight < weightCount; weight++){
					layers[layer][neuron].setWeight(weight, input.readDouble()); // read each weight
				}
			}
			lastNeuronCount = layers[layer].length;
		}
	}
}