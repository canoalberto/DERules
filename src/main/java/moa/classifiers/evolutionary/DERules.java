package moa.classifiers.evolutionary;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;

public class DERules extends AbstractClassifier implements MultiClassClassifier {

	private static final long serialVersionUID = 1L;

	public IntOption seed = new IntOption("seed", 'r', "Seed", 123456, 1, Integer.MAX_VALUE);

	public IntOption populationSize = new IntOption("populationSize", 'p', "Population size", 128, 2, 1000);

	public IntOption numberGenerations = new IntOption("numberGenerations", 'g', "Number generations", 50, 1, 1000);

	public IntOption numberRulesClass = new IntOption("numberRulesClass", 'n', "Number rules per class", 5, 1, 100);

	public IntOption windowSize = new IntOption("windowSize", 'w', "Number of instances per chunk", 1000, 1, Integer.MAX_VALUE);

	protected InstancesHeader context;

	protected Instances window;

	protected float[] classifier;

	private double[] numberInstancesClass;

	protected boolean model;

	private native void contextualize(int seed, int populationSize, int numberGenerations, int numberRulesClass, int numberInstances, int numberAttributes, int numberClasses);

	protected native void copyDataset(DERules algorithm, int numberInstances);

	protected native float[] train();

	@Override
	public void setModelContext(InstancesHeader context) {

		try {
			System.loadLibrary("DERules");
		} catch (Exception e) {
			System.err.println("Can't load DERules GPU library. Please make sure to include the library path");
			System.exit(0);
		}

		contextualize(seed.getValue(), populationSize.getValue(), numberGenerations.getValue(), numberRulesClass.getValue(), windowSize.getValue(), context.numAttributes(), context.numClasses());

		this.context = context;
		this.classifier = new float[context.numClasses() * numberRulesClass.getValue() * 2 * context.numAttributes()];
	}

	@Override
	public void resetLearningImpl() {
	}

	@Override
	public void trainOnInstanceImpl(Instance instance) {

		if(window == null)
			window = new Instances(instance.dataset(), 0);

		window.add(instance);

		if(window.size() == windowSize.getValue())
		{
			copyDataset(this, window.size());
			classifier = train();
			model = true;
			window.delete();
		}
	}

	@Override
	public double[] getVotesForInstance(Instance instance)
	{
		double[] votes = new double[instance.numClasses()];
		
		if(model)
		{
			for(int i = 0; i < instance.numClasses(); i++)
			{
				for(int j = 0; j < numberRulesClass.getValue(); j++)
				{
					boolean covers = true;

					for(int k = 0; k < instance.numAttributes()-1; k++)
					{
						float minValue = classifier[(i * numberRulesClass.getValue() * 2 * instance.numAttributes()) + (j * 2 * instance.numAttributes()) + 2*k];
						float maxValue = classifier[(i * numberRulesClass.getValue() * 2 * instance.numAttributes()) + (j * 2 * instance.numAttributes()) + 2*k + 1];

						if(minValue <= maxValue && (instance.value(k) < minValue || instance.value(k) > maxValue))
						{
							covers = false;
							break;
						}
					}

					if(covers)
						votes[i] += classifier[(i * numberRulesClass.getValue() * 2 * instance.numAttributes()) + (j * 2 * instance.numAttributes()) + 2*instance.numAttributes() - 1];
				}
			}
		}

		double votesSum = 0;

		for(int i = 0; i < instance.numClasses(); i++)
			votesSum += votes[i];

		if(votesSum == 0) // default prediction based on relative frequency if no rules are triggered
			return numberInstancesClass;
		else
			return votes;
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		Measurement[] measurements = new Measurement[2];

		measurements[0] = new Measurement("NumberRules", numberRulesClass.getValue() * numberRulesClass.getValue());
		
		int numberConditions = 0;

		if(model)
		{
			for(int i = 0; i < context.numClasses(); i++)
			{
				for(int j = 0; j < numberRulesClass.getValue(); j++)
				{
					for(int k = 0; k < context.numInputAttributes(); k++)
					{
						float minValue = classifier[(i * numberRulesClass.getValue() * 2 * context.numAttributes()) + (j * 2 * context.numAttributes()) + 2*k];
						float maxValue = classifier[(i * numberRulesClass.getValue() * 2 * context.numAttributes()) + (j * 2 * context.numAttributes()) + 2*k + 1];

						if(minValue <= maxValue)
							numberConditions++;
					}
				}
			}
		}

		measurements[1] = new Measurement("NumberConditions", numberConditions);

		return measurements;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	public boolean isRandomizable() {
		return true;
	}

	public float[] getDataset()
	{
		float[] dataset = new float[window.size() * window.numAttributes()];
		numberInstancesClass = new double[window.numClasses()];

		for(int i = 0; i < window.size(); i++)
		{
			for(int j = 0; j < window.numAttributes(); j++)
				dataset[i*window.numAttributes() + j] = (float) window.get(i).value(j);

			numberInstancesClass[(int) window.get(i).classValue()]++;
		}

		return dataset;				
	}
}