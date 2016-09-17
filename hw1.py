import numpy as np
from random import choice
import matplotlib.pyplot as plt
class Learner():
	def __init__(self, algorithm, label_gen, experts):
		if algorithm == "WMA":
			self.algorithm = 0
			self.algorithm_name = algorithm
		elif algorithm == "RWMA":
			self.algorithm = 1
			self.algorithm_name = algorithm
		else:
			print "check algorithm spelling"

		if label_gen == "stochastic":
			self.labels = 0
			self.label_name = label_gen
		elif label_gen == "deterministic":
			self.labels = 1
			self.label_name = label_gen
		elif label_gen == "adversarial":
			self.labels = 2
			self.label_name = label_gen
		else:
			print "check label generator spelling"

		self.experts = experts
		self.weights = [1]*experts
		self.loss = []
		self.expert_loss = plot_data = [[] for _ in range(experts)]
		self.regret = plot_data = [[] for _ in range(experts)]
		
	def expertAdvice(self,i):
		h = [0]*self.experts
		h[0] = 1
		h[1] = -1
		if i%2 == 0:	#odd
			h[2] = -1
		else: 			#even
			h[2] = 1
		return h

	def predictor(self, h, ind):
		if self.algorithm == 0:
			dotp = np.dot(h,self.weights)
		else:
			dotp = np.dot(h[ind],self.weights[ind])
		if dotp < 0:
			y_hat = -1
		else:
			y_hat = 1
		return y_hat

	def true_label(self, h):
		if self.labels == 0:
			y  = choice([-1,1])
		elif self.labels ==1:
			#always winning 
			y  = 1
		else:
			# adversary
			y = np.dot(h,self.weights)
			if y > 0:
				y = -1
			else:
				y = 1
		return y

	def learn(self,T, eta):
		self.weights = [1]*self.experts
		phi = 0
		ind = 0
		for i in range(T):
			hypothesis = self.expertAdvice(i)
			if self.algorithm == 1:
				phi = np.sum(self.weights)
				ind = np.argmax(np.random.multinomial(1, np.divide(self.weights,phi))) 
			y_hat = self.predictor(hypothesis,ind)
			y = self.true_label(hypothesis)
			loss = self.calculateLoss(y,y_hat)
			self.loss.append(loss)
			for ind in range(self.experts):
				loss_exp = self.calculateLoss(y,hypothesis[ind])
				self.expert_loss[ind].append(loss_exp)
				regret = (np.sum(self.loss) - np.sum(self.expert_loss[ind]))/(i+1.)
				self.regret[ind].append(regret)
			self.weights = self.updateWeights(hypothesis,y,eta)
			"""print "y_hat", y_hat
			print "expert 3 prediction",self.expertAdvice(i)[2]
			print "y_actual", y
			print "loss", loss
			print "expert3 loss", self.expert_loss[2][-1]
			print "weights", self.weights"""
		print "Total loss", np.sum(self.loss)

	def calculateLoss(self, y,y_hat):
		l = y_hat-y
		if l ==0:
			l = 0
		else:
			l = 1
		return l

	def updateWeights(self,h,y,eta):
		for i in range(self.experts):
			loss = self.calculateLoss(y,h[i])
			if loss != 0:
				self.weights[i] = self.weights[i]*(1-eta)
		return self.weights

	def plotLossGraph(self,T):
		time = np.arange(T)
		f, axarr = plt.subplots(3, sharex=True)
		for i in range(self.experts):
			axarr[i].set_title('Loss Plot for {} and {} case'.format(self.algorithm_name, self.label_name))
			axarr[i].plot(time,self.loss, label='Algorithm loss')
			axarr[i].plot(time,self.expert_loss[i], label='Expert {}'.format(i))
			axarr[i].set_ylim([-0.5, 1.5])
			axarr[i].legend()
			axarr[i].set_ylabel('loss')
		axarr[i].set_xlabel('time')
		
	def plotRegretGraph(self,T):
		time = np.arange(T)
		f, axarr = plt.subplots(3, sharex=True)
		for i in range(self.experts):
			axarr[i].set_title('Average Regret Plot for {} and {} case'.format(self.algorithm_name, self.label_name))
			axarr[i].plot(time,self.regret[i], label='Expert{}'.format(i))
			axarr[i].set_ylabel('avg regret')
			axarr[i].legend()
		axarr[i].set_xlabel('time')
		
def main():
	T = 100
	eta = 0.1
	experts = 3
	allplots = False
	"""
	WMA, RWMA
	stochastic,deterministic,adversarial
	"""

	param1 = ["WMA","RWMA"]
	param2 = ["stochastic","deterministic","adversarial"]

	if allplots:
		for i in param1:
			for j in param2:
				predict = Learner(i,j, experts)
				predict.learn(T,eta)
				predict.plotLossGraph(T)
				predict.plotRegretGraph(T)
	else:
		predict = Learner("WMA","stochastic", experts)
		predict.learn(T,eta)
		predict.plotLossGraph(T)
		predict.plotRegretGraph(T)

	plt.show()

if __name__ == "__main__":
	main()