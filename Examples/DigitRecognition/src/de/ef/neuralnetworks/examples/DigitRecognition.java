package de.ef.neuralnetworks.examples;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.DoubleFunction;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import javax.imageio.ImageIO;

import de.ef.neuralnetworks.NeuralNetwork;
import de.ef.neuralnetworks.NeuralNetworkFactory;
import de.ef.neuralnetworks.util.NeuralNetworkTraining;
import de.ef.neuralnetworks.util.image.MonochromeImageData;

public class DigitRecognition{
	
	private final static double OUTPUTS[][] = {
		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
	};
	
	
	public static void main(String ... args) throws IOException, ClassNotFoundException{
		File dataSet = new File("../../Datasets/digits.dataset.zip");
		
		if(dataSet.exists() == false)
			throw new FileNotFoundException(
				"Dataset not found. (Expected at " + dataSet.getAbsolutePath() + ")"
			);
		
		NeuralNetwork network;
		
		File comparatorData = new File("./comp.dat");
		if(comparatorData.exists() == false){
			System.out.println("Creating new neural-networks...");
			String config =
				"{\"implementation\": \"SlowWave\", \"layers\": [256, 32, 10]}";
			network = NeuralNetworkFactory.create(config);
		}
		else{
			System.out.println("Loading neural-networks from file...");
			try(ObjectInputStream input =
					new ObjectInputStream(new FileInputStream(comparatorData))){
				network = (NeuralNetwork)input.readObject();
			}
		}
		
		DigitRecognition recognition = new DigitRecognition(new ZipFile(dataSet), network);
		
		ExecutorService executor = Executors.newSingleThreadExecutor();
		
		AtomicBoolean interrupt = new AtomicBoolean(false);
		AtomicInteger index = new AtomicInteger(0);
		
		System.out.println("Starting training...");
		executor.execute(
			() -> recognition.train(e -> {
				if(index.getAndIncrement() % 100 == 0)
					System.out.println(e);
				return interrupt.get();
			})
		);
		
		// wait for any key press
		System.out.println("Press any key to interrupt.");
		System.in.read();
		interrupt.set(true);
		
		// after training finished start next task and save neural-networks to disk
		/*executor.execute(() -> {
			System.out.println("Saving neural-networks...");
			try(ObjectOutputStream output =
					new ObjectOutputStream(new FileOutputStream(comparatorData))){
				output.writeObject(equal);
				output.writeObject(compare);
			}
			catch(Throwable t){
				t.printStackTrace();
				System.exit(1);
			}
			System.out.println("Finished!");
			System.exit(0);
		});*/
	}
	
	
	
	private final ZipFile dataSet;
	private Map<double[], double[]> data;
	
	private NeuralNetwork network;
	
	
	public DigitRecognition(ZipFile dataSet, NeuralNetwork network){
		this.dataSet = dataSet;
		
		this.network = network;
	}
	
	
	public void train(DoubleFunction<Boolean> completed){
		if(this.data == null){
			Map<Byte, BufferedImage> images = new HashMap<>();
			
			try{
				Enumeration<? extends ZipEntry> entries = dataSet.entries();
				while(entries.hasMoreElements()){
					ZipEntry entry = entries.nextElement();
					
					if(entry.isDirectory() == false){
						String name =
							entry.getName().substring(0, entry.getName().indexOf('/'));
						images.put(Byte.valueOf(name), ImageIO.read(dataSet.getInputStream(entry)));
					}
				}
			}
			catch(IOException e){
				throw new RuntimeException(e);
			}
			
			Map<double[], double[]> data = new HashMap<>();
			for(Entry<Byte, BufferedImage> image : images.entrySet()){
				data.put(
					new MonochromeImageData(image.getValue(), Color.WHITE.getRGB(), 16, 16, true)
						.getData(),
					DigitRecognition.OUTPUTS[image.getKey()]
				);
			}
			
			this.data = data;
		}
		
		try{
			NeuralNetworkTraining.train(this.network, this.data, completed);
		}
		catch(IOException e){
			throw new RuntimeException(e);
		}
	}
}
