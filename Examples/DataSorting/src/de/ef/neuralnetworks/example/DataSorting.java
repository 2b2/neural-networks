package de.ef.neuralnetworks.example;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import javax.imageio.ImageIO;

import de.ef.neuralnetworks.NeuralNetwork;
import de.ef.neuralnetworks.NeuralNetworkComparator;
import de.ef.neuralnetworks.pipeline.Pipeline;
import de.ef.neuralnetworks.pipeline.PipelineBuilder;
import de.ef.neuralnetworks.pipeline.image.GrayscaleImageConverter;
import de.ef.neuralnetworks.pipeline.image.ImageObjects;
import de.ef.slowwave.pipeline.ByteArrayBufferFactory;
import de.ef.slowwave.pipeline.ByteArrayBufferFactory.FixedByteArrayBufferFactory;

/**
 * This example tries to sort a data-set consisting of the digits 0-9.
 * <p>
 * The whole thing with data sorting is just an experiment that has a high
 * possibility to fail. Maybe some data formats work better then others,
 * but binary images of handwritten digits do not seam to work at all.
 * </p>
 * <p>
 * <u>Update:</u> There could be hope, with a different configuration of
 * learning rate and hidden layer size/count it seams to make progress while
 * learning.
 * </p>
 * 
 * @author Erik Fritzsche
 */
public class DataSorting{
	
	private final static int INPUT_WIDTH = 32, INPUT_HEIGHT = 32, INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT;
	
	
	@SuppressWarnings("unchecked")
	public static void main(String ... args) throws IOException, ClassNotFoundException, InstantiationException, IllegalAccessException{
		File dataSet = new File("../../Datasets/digits.dataset.zip");
		
		if(dataSet.exists() == false)
			throw new FileNotFoundException(
				"Dataset not found. (Expected at " + dataSet.getAbsolutePath() + ")"
			);
		
		NeuralNetwork<double[], double[]> equal, compare;
		
		File comparatorData = new File("./comp.dat");
		if(comparatorData.exists() == false){
			System.out.println("Creating new neural-networks...");

			equal = NeuralNetwork.load("de.ef.slowwave.SlowWave");
			compare = NeuralNetwork.load("de.ef.slowwave.SlowWave");
			
			Map<String, Object> properties = new HashMap<>();
			properties.put("learning.rate", 0.1);
			
			equal.init(INPUT_SIZE, new int[]{128, 128}, 1, properties);
			compare.init(INPUT_SIZE, new int[]{128, 128}, 1, properties);
		}
		else{
			System.out.println("Loading neural-networks from file...");
			
			try(ObjectInputStream input =
					new ObjectInputStream(new FileInputStream(comparatorData))){
				equal = (NeuralNetwork<double[], double[]>)input.readObject();
				compare = (NeuralNetwork<double[], double[]>)input.readObject();
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
	private Map<Long, List<double[]>> data;
	
	private NeuralNetworkComparator<double[]> comparator;
	private Pipeline<BufferedImage, double[]> inputPipeline;
	
	
	public DataSorting(ZipFile dataSet, NeuralNetwork<double[], double[]> equal, NeuralNetwork<double[], double[]> compare)
			throws IOException, ClassNotFoundException, InstantiationException, IllegalAccessException{
		this.dataSet = dataSet;
		
		this.comparator = new NeuralNetworkComparator<>(
			equal, compare, (a, b) -> {
				double[] combined = new double[a.length + b.length];
				System.arraycopy(a, 0, combined, 0, a.length);
				System.arraycopy(b, 0, combined, a.length, b.length);
				return combined;
			}, d -> (float)d[0], f -> new double[]{(double)f}
		);
		
		ImageObjects imageObjects = ImageObjects.load("de.ef.slowwave.pipeline.image.SlowWaveImageObjects");
		imageObjects.init(null);
		this.inputPipeline =
			new PipelineBuilder<BufferedImage, double[]>()
			.root(
				GrayscaleImageConverter.fromBufferedImage(new ByteArrayBufferFactory(1), false)
			)
			.pipe(
				imageObjects.foregroundObjectExtractor(
					INPUT_WIDTH, INPUT_HEIGHT, 127, new FixedByteArrayBufferFactory(INPUT_SIZE, 1)
				)
			)
			.exit(
				GrayscaleImageConverter.toDoubleArray((s, c) -> new double[s])
			)
			.build();
	}
	
	
	public void train(Predicate<Double> completed){
		if(this.data == null){
			List<Entry<Byte, BufferedImage>> images = new ArrayList<>();
			
			try{
				Enumeration<? extends ZipEntry> entries = this.dataSet.entries();
				while(entries.hasMoreElements()){
					ZipEntry entry = entries.nextElement();
					
					if(entry.isDirectory() == false){
						String name =
							entry.getName().substring(0, entry.getName().indexOf('/'));
						images.add(
							new SimpleEntry<>(Byte.valueOf(name), ImageIO.read(this.dataSet.getInputStream(entry)))
						);
					}
				}
			}
			catch(IOException e){
				throw new RuntimeException(e);
			}
			
			Map<Long, List<double[]>> data = new HashMap<>();
			for(Entry<Byte, BufferedImage> image : images){
				List<double[]> container = data.get(image.getKey());
				if(container == null)
					data.put((long)image.getKey(), container = new LinkedList<>());
				container.add(this.inputPipeline.process(image.getValue()));
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
