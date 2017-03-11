package de.ef.slowwave;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * {@code SlowWaveSerialization} is a helper class for
 * {@link de.ef.slowwave.SlowWave SlowWave} to serialize
 * and deserialize it.
 * 
 * @author Erik Fritzsche
 * @version 1.0
 * @since 1.0
 */
class SlowWaveSerialization{
	
	final static byte LEARNING_RATE_PROPERTY = 0;
	
	
	
	private SlowWaveSerialization(){}
	
	
	
	static void write(SlowWave network, ObjectOutputStream output) throws IOException{
		// write header byte with bit flags 0b0001 for double precision, 0b0010 for bias neuron and 0b0100 for extra properties
		output.writeByte(0b0111);
		
		// write properties
		// property count as short
		output.writeShort(1);
		// learning rate property
		output.writeByte(LEARNING_RATE_PROPERTY);
		output.writeByte(Double.BYTES);
		output.writeDouble(network.learningRate);
		
		SlowWave.Neuron layers[][] = network.layers;
		
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
	
	
	static void read(SlowWave network, ObjectInputStream input) throws IOException{
		byte header = input.readByte(); // read header byte
		// extract bit flags (double precision and bias neuron)
		// if one of the flags is not set correctly throw exception
		if((header & 0b0001) != 0b0001 || (header & 0b0010) != 0b0010){
			throw new IOException("Incorrect serialization header: " + header);
		}
		
		// keep track of all set properties
		boolean learningRateSet = false;
		
		// read properties if present
		if((header & 0b0100) == 0b0100){
			int count = input.readShort();
			
			for(int i = 0; i < count; i++){
				byte property = input.readByte();
				int size = 0, next, currentBit = 0;
				do{
					next = (input.readByte() & 0xFF);
					size |= next << currentBit;
					currentBit += 7;
				}while((next & 0b1000_0000) == 0b1000_0000);
				
				switch(property){
					case LEARNING_RATE_PROPERTY:
						if(size != Double.BYTES)
							throw new IOException("Size of property does not match double size: " + size);
						learningRateSet = true; network.learningRate = input.readDouble(); break;
					default: // just read bytes and throw property away
						for(int j = 0; j < size; j++) input.readByte();
				}
			}
		}
		// set all uninitialized properties
		if(learningRateSet == false)
			network.learningRate = SlowWave.DEFAULT_LEARNING_RATE;
		
		
		SlowWave.Neuron layers[][] = new SlowWave.Neuron[input.readInt()][]; // read layer count
		
		for(int layer = 0; layer < layers.length; layer++){
			layers[layer] = new SlowWave.Neuron[input.readInt()]; // read each neuron count
		}
		
		network.layers = layers;
		
		int lastNeuronCount = -1;
		for(int layer = 0; layer < layers.length; layer++){
			for(int neuron = 0; neuron < layers[layer].length; neuron++){
				// plus one to lastNeuronCount for bias neuron
				int weightCount = (lastNeuronCount != -1 ? lastNeuronCount + 1 : 0);
				// create Neuron with enclosing SlowWave instance
				layers[layer][neuron] = network.new Neuron(new double[weightCount]);
				for(int weight = 0; weight < weightCount; weight++){
					layers[layer][neuron].setWeight(weight, input.readDouble()); // read each weight
				}
			}
			lastNeuronCount = layers[layer].length;
		}
	}
}
