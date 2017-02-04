package de.ef.neuralnetworks.examples;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Predicate;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import javax.imageio.ImageIO;

import de.ef.neuralnetworks.NeuralNetwork;
import de.ef.neuralnetworks.NeuralNetworkContext;
import de.ef.neuralnetworks.NeuralNetworkContextFactory;
import de.ef.neuralnetworks.pipeline.Pipeline;
import de.ef.neuralnetworks.pipeline.PipelineBuilder;
import de.ef.neuralnetworks.pipeline.image.GrayscaleImageConverter;
import de.ef.neuralnetworks.util.NeuralNetworkTraining;
import de.ef.slowwave.pipeline.ByteArrayBufferFactory;
import de.ef.slowwave.pipeline.ByteArrayBufferFactory.FixedByteArrayBufferFactory;
import de.ef.slowwave.pipeline.image.SlowWaveForegroundObjectExtractor;

public class DigitRecognition{
	
	private final static int INPUT_WIDTH = 32, INPUT_HEIGHT = 32, INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT;
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
	
	
	@SuppressWarnings("unchecked")
	public static void main(String ... args) throws IOException, ClassNotFoundException{
		File dataSet = new File("../../Datasets/digits.dataset.zip");
		
		if(dataSet.exists() == false)
			throw new FileNotFoundException(
				"Dataset not found. (Expected at " + dataSet.getAbsolutePath() + ")"
			);
		
		NeuralNetwork<double[], double[]> network;
		
		File networkData = new File("./comp.dat");
		if(networkData.exists() == false){
			System.out.println("Creating new neural-networks...");
			Class.forName("de.ef.slowwave.SlowWaveContext");
			NeuralNetworkContext context = NeuralNetworkContextFactory.create("SlowWave");
			
			Map<String, Object> properties = new HashMap<>();
			properties.put("layers.input.size", INPUT_SIZE);
			properties.put("layers.output.size", 10);
			properties.put("layers.hidden.count", 2);
			properties.put("layers.hidden[0].size", 128);
			properties.put("layers.hidden[1].size", 128); // probably unnecessary, one hidden layer with 128 neurons works too
			                                              // but this seams to learn faster and better
			
			network = context.createNeuralNetwork(double[].class, double[].class, properties);
		}
		else{
			System.out.println("Loading neural-networks from file...");
			try(ObjectInputStream input =
					new ObjectInputStream(new FileInputStream(networkData))){
				network = (NeuralNetwork<double[], double[]>)input.readObject();
			}
		}
		
		DigitRecognition recognition = new DigitRecognition(new ZipFile(dataSet), network);
		
		ExecutorService executor = Executors.newSingleThreadExecutor();
		
		AtomicBoolean interrupt = new AtomicBoolean(false);
		AtomicInteger index = new AtomicInteger(0);
		
		System.out.println("Starting training...");
		executor.execute(
			() -> recognition.train(e -> {
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
					new ObjectOutputStream(new FileOutputStream(networkData))){
				output.writeObject(network);
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
	private List<Entry<double[], double[]>> data;
	
	private NeuralNetwork<double[], double[]> network;
	private Pipeline<BufferedImage, double[]> inputPipeline;
	
	
	public DigitRecognition(ZipFile dataSet, NeuralNetwork<double[], double[]> network){
		this.dataSet = dataSet;
		
		this.network = network;
		this.inputPipeline =
			new PipelineBuilder<BufferedImage, double[]>()
			.root(
				GrayscaleImageConverter.fromBufferedImage(new ByteArrayBufferFactory(1), false)
			)
			.pipe(
				new SlowWaveForegroundObjectExtractor(
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
				Enumeration<? extends ZipEntry> entries = dataSet.entries();
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
			
			List<Entry<double[], double[]>> data = new ArrayList<>();
			for(Entry<Byte, BufferedImage> image : images){
				data.add(
					new SimpleEntry<>(
						this.inputPipeline.process(image.getValue()), DigitRecognition.OUTPUTS[image.getKey()]
					)
				);
			}
			
			this.data = data;
		}
		
		try{
			BiFunction<double[], double[], Double> errorCalculator = (o, e) -> {
				double error = 0;
				for(int i = 0; i < o.length; i++)
					error += Math.abs(o[i] - e[i]);
				return error / o.length;
			};
			
			NeuralNetworkTraining.train(this.network, this.data, errorCalculator, completed);
		}
		catch(IOException e){
			throw new RuntimeException(e);
		}
	}
}
