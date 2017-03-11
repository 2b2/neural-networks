package de.ef.slowwave.test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;

import de.ef.slowwave.SlowWave;

public class SerializationTest{

	public SerializationTest(){}
	
	
	@Test
	public void test(){
		SlowWave s = new SlowWave();
		
		Map<String, Object> properties = new HashMap<>();
		properties.put("learning.rate", 0.49515155);
		
		s.init(3, new int[]{2}, 1, properties);
		
		try{
			ByteArrayOutputStream bytes = new ByteArrayOutputStream();
			ObjectOutputStream out = new ObjectOutputStream(bytes);
			
			out.writeObject(s);
			
			ObjectInputStream in =
				new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()));
			
			SlowWave sRead = (SlowWave)in.readObject();
			
			Field learningRateField = SlowWave.class.getDeclaredField("learningRate");
			learningRateField.setAccessible(true);
			Assert.assertEquals((double)learningRateField.get(s), (double)learningRateField.get(sRead), 0.0);
			
			Field layersField = SlowWave.class.getDeclaredField("layers");
			layersField.setAccessible(true);
			
			Class<?> neuronClass = Class.forName("de.ef.slowwave.SlowWave$Neuron");
			Field weightsField = neuronClass.getDeclaredField("weights");
			weightsField.setAccessible(true);
			
			Object layers[][] = (Object[][])layersField.get(s);
			Object layersRead[][] = (Object[][])layersField.get(sRead);
			
			Assert.assertEquals(layers.length, layersRead.length);
			for(int i = 0; i < layers.length; i++){
				Assert.assertEquals(layers[i].length, layersRead[i].length);
				for(int j = 0; j < layers[i].length; j++){
					double weights[] = (double[])weightsField.get(layers[i][j]);
					double weightsRead[] = (double[])weightsField.get(layersRead[i][j]);
					
					Assert.assertArrayEquals(weights, weightsRead, 0.0);
				}
			}
		}catch(Throwable t){
			t.printStackTrace();
			Assert.fail("Exception while testing: " + t.getMessage());
		}
	}
}
