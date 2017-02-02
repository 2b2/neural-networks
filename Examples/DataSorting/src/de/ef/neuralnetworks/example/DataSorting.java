package de.ef.neuralnetworks.example;

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
import java.util.LinkedList;
import java.util.List;
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
import de.ef.neuralnetworks.NeuralNetworkComparator;
import de.ef.neuralnetworks.NeuralNetworkData;
import de.ef.neuralnetworks.NeuralNetworkFactory;
import de.ef.neuralnetworks.util.image.MonochromeImageData;

/**
 * This example tries to sort a data-set consisting of the digits 0-9.
 * <p>
 * The whole thing with data sorting is just an experiment that has a high
 * possibility to fail. Maybe some data formats work better then others,
 * but binary images of handwritten digits do not seam to work at all.
 * </p>
 * 
 * @author Erik Fritzsche
 */
public class DataSorting{
	
	public static void main(String ... args) throws IOException, ClassNotFoundException{
		File dataSet = new File("../../Datasets/digits.dataset.zip");
		
		if(dataSet.exists() == false)
			throw new FileNotFoundException(
				"Dataset not found. (Expected at " + dataSet.getAbsolutePath() + ")"
			);
		
		NeuralNetwork equal, compare;
		
		File comparatorData = new File("./comp.dat");
		if(comparatorData.exists() == false){
			System.out.println("Creating new neural-networks...");
			String config =
				"{\"implementation\": \"SlowWave\", \"layers\": [256, 32, 1]}";
			equal = NeuralNetworkFactory.create(config);
			compare = NeuralNetworkFactory.create(config);
		}
		else{
			System.out.println("Loading neural-networks from file...");
			try(ObjectInputStream input =
					new ObjectInputStream(new FileInputStream(comparatorData))){
				equal = (NeuralNetwork)input.readObject();
				compare = (NeuralNetwork)input.readObject();
			}
		}
		
		DataSorting sorting = new DataSorting(new ZipFile(dataSet), equal, compare);
		
		ExecutorService executor = Executors.newSingleThreadExecutor();
		
		AtomicBoolean interrupt = new AtomicBoolean(false);
		AtomicInteger index = new AtomicInteger(0);
		
		System.out.println("Starting training...");
		executor.execute(
			() -> sorting.train(e -> {
				if(index.getAndIncrement() % 10 == 0)
					System.out.println(e);
				return interrupt.get();
			})
		);
		
		// wait for any key press
		System.out.println("Press any key to interrupt.");
		System.in.read();
		interrupt.set(true);
		
		// after training finished start next task and save neural-networks to disk
		executor.execute(() -> {
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
		});
	}
	
	
	
	private final ZipFile dataSet;
	private Map<Long, List<NeuralNetworkData>> data;
	
	private NeuralNetworkComparator<NeuralNetworkData> comparator;
	
	
	public DataSorting(ZipFile dataSet, NeuralNetwork equal, NeuralNetwork compare){
		this.dataSet = dataSet;
		
		this.comparator = new NeuralNetworkComparator<>(equal, compare);
	}
	
	
	public void train(DoubleFunction<Boolean> completed){
		if(this.data == null){
			Map<Long, BufferedImage> images = new HashMap<>();
			
			try{
				Enumeration<? extends ZipEntry> entries = this.dataSet.entries();
				while(entries.hasMoreElements()){
					ZipEntry entry = entries.nextElement();
					
					if(entry.isDirectory() == false){
						String name =
							entry.getName().substring(0, entry.getName().indexOf('/'));
						images.put(Long.valueOf(name), ImageIO.read(this.dataSet.getInputStream(entry)));
					}
				}
			}
			catch(IOException e){
				throw new RuntimeException(e);
			}
			
			Map<Long, List<NeuralNetworkData>> data = new HashMap<>();
			for(Entry<Long, BufferedImage> image : images.entrySet()){
				List<NeuralNetworkData> container = data.get(image.getKey());
				if(container == null)
					data.put(image.getKey(), container = new LinkedList<>());
				// FIXME just testing stuff here
				if(container.size() < 5)
					container.add(new MonochromeImageData(image.getValue(), Color.WHITE.getRGB(), 16, 16, true));
			}
			
			this.data = data;
		}
		
		try{
			this.comparator.train(this.data, completed);
		}
		catch(IOException e){
			throw new RuntimeException(e);
		}
	}
}