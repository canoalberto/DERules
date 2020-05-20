package moa.classifiers.evolutionary;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.GaussianNumericAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NominalAttributeClassObserver;
import moa.core.AutoExpandVector;
import moa.core.Measurement;

public class DERulesNB extends DERules {

	private static final long serialVersionUID = 1L;

	private NaiveBayes naiveBayes = new NaiveBayes();

	@Override
	public void trainOnInstanceImpl(Instance instance) {

		if(window == null)
			window = new Instances(instance.dataset(), 0);

		window.add(instance);

		if(window.size() == windowSize.getValue())
		{
			copyDataset(this, window.size());
			classifier = train();

			naiveBayes.resetLearning();

			for(int i = 0; i < window.size(); i++)
				naiveBayes.trainOnInstance(window.get(i));

			model = true;
			window.delete();
		}
	}

	@Override
	public double[] getVotesForInstance(Instance instance)
	{
		boolean fired = false;
		double[] votes = new double[instance.numClasses()];

		if(!model)	return votes;

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
				{
					fired = true;
					votes[i] += classifier[(i * numberRulesClass.getValue() * 2 * instance.numAttributes()) + (j * 2 * instance.numAttributes()) + 2*instance.numAttributes() - 1];
				}
			}
		}

		if(fired == true)
		{
			votes = normalize(votes);
			votes = naiveBayes.doNaiveBayesPredictionLog(instance, votes);
			votes = exponential(votes);
			votes = normalize(votes);
			return votes;
		}
		else
		{
			double[] naiveBayesVotes;
			naiveBayesVotes = naiveBayes.doNaiveBayesPredictionLog(instance);
			naiveBayesVotes = exponential(naiveBayesVotes);
			naiveBayesVotes = normalize(naiveBayesVotes);
			return naiveBayesVotes;
		}
	}
	protected double[] normalize(double[] votes) {
		double sum=0;
		for (int i = 0; i < votes.length; i++) {
			sum = sum + votes[i];
		}
		for (int j = 0; j < votes.length; j++) {
			votes[j] = votes[j] / sum;
		}
		return votes;
	}

	protected double[] exponential(double[] votes) {
		for (int i = 0; i < votes.length; i++) {
			votes[i] = Math.exp(votes[i]);
		}
		return votes;
	}

	private class NaiveBayes extends AbstractClassifier implements MultiClassClassifier {

		private static final long serialVersionUID = 1L;

		@Override
		public String getPurposeString() {
			return "Naive Bayes classifier: performs classic bayesian prediction while making naive assumption that all inputs are independent.";
		}

		protected double[] observedClassDistribution;

		protected AutoExpandVector<AttributeClassObserver> attributeObservers;

		@Override
		public void resetLearningImpl() {
			this.observedClassDistribution = null;
			this.attributeObservers = new AutoExpandVector<AttributeClassObserver>();
		}

		@Override
		public void trainOnInstanceImpl(Instance inst) {

			if(observedClassDistribution == null)
				observedClassDistribution = new double[inst.numClasses()];

			this.observedClassDistribution[(int) inst.classValue()]++;

			for (int i = 0; i < inst.numAttributes() - 1; i++) {
				int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
				AttributeClassObserver obs = this.attributeObservers.get(i);
				if (obs == null) {
					obs = inst.attribute(instAttIndex).isNominal() ? newNominalClassObserver() : newNumericClassObserver();
					this.attributeObservers.set(i, obs);
				}
				obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
			}
		}

		@Override
		public double[] getVotesForInstance(Instance inst) {
			return doNaiveBayesPrediction(inst, this.observedClassDistribution, this.attributeObservers);
		}

		@Override
		protected Measurement[] getModelMeasurementsImpl() {
			return null;
		}

		@Override
		public void getModelDescription(StringBuilder out, int indent) {
		}

		@Override
		public boolean isRandomizable() {
			return false;
		}

		protected AttributeClassObserver newNominalClassObserver() {
			return new NominalAttributeClassObserver();
		}

		protected AttributeClassObserver newNumericClassObserver() {
			return new GaussianNumericAttributeClassObserver();
		}

		public double[] doNaiveBayesPrediction(Instance inst, double[] observedClassDistribution, AutoExpandVector<AttributeClassObserver> attributeObservers) {
			double[] votes = new double[inst.numClasses()];
			double observedClassSum = sumOfValues(observedClassDistribution);
			for (int classIndex = 0; classIndex < votes.length; classIndex++) {
				votes[classIndex] = observedClassDistribution[classIndex] / observedClassSum;
				for (int attIndex = 0; attIndex < inst.numAttributes() - 1; attIndex++) {
					int instAttIndex = modelAttIndexToInstanceAttIndex(attIndex, inst);
					AttributeClassObserver obs = attributeObservers.get(attIndex);
					if ((obs != null) && !inst.isMissing(instAttIndex)) {
						votes[classIndex] *= obs.probabilityOfAttributeValueGivenClass(inst.value(instAttIndex), classIndex);
					}
				}
			}
			return votes;
		}

		public double[] doNaiveBayesPredictionLog(Instance inst) {
			AttributeClassObserver obs;
			double[] votes = new double[inst.numClasses()];
			double observedClassSum = sumOfValues(observedClassDistribution);
			for (int classIndex = 0; classIndex < votes.length; classIndex++) {
				votes[classIndex] = Math.log10(observedClassDistribution[classIndex] / observedClassSum);
				for (int attIndex = 0; attIndex < inst.numAttributes() - 1; attIndex++) {
					int instAttIndex = modelAttIndexToInstanceAttIndex(attIndex, inst);
					obs = attributeObservers.get(attIndex);
					if ((obs != null) && !inst.isMissing(instAttIndex)) {
						votes[classIndex] += Math.log10(obs.probabilityOfAttributeValueGivenClass(inst.value(instAttIndex), classIndex));
					}
				}
			}
			return votes;
		}

		public double[] doNaiveBayesPredictionLog(Instance inst, double[] ruleVotes) {
			AttributeClassObserver obs;
			double[] votes = new double[inst.numClasses()];
			double observedClassSum = sumOfValues(observedClassDistribution);
			for (int classIndex = 0; classIndex < votes.length; classIndex++) {
				votes[classIndex] = Math.log10(observedClassDistribution[classIndex] / observedClassSum);
				for (int attIndex = 0; attIndex < inst.numAttributes() - 1; attIndex++) {
					int instAttIndex = modelAttIndexToInstanceAttIndex(attIndex, inst);
					obs = attributeObservers.get(attIndex);
					if ((obs != null) && !inst.isMissing(instAttIndex)) {
						votes[classIndex] += Math.log10(obs.probabilityOfAttributeValueGivenClass(inst.value(instAttIndex), classIndex));
					}
				}
				votes[classIndex] += Math.log10(ruleVotes[classIndex]);
			}
			return votes;
		}


		private double sumOfValues(double[] array) {
			double sum = 0;
			for(int i = 0; i < array.length; i++) sum += array[i];
			return sum;
		}
	}
}